"""Structured object-slot supervision and local decoder training."""

from __future__ import annotations

import copy
import math
import operator

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from .train_visual_latent_model import PhaseTrainingResult
from .visual_dataset import IMAGE_SIZE
from .visual_latent_data import (
    frame_indices_for_episode_ids,
    frames_to_tensor,
    renderer_object_masks,
)
from .visual_latent_model import SpatialConvAutoencoder


def normalize_object_slot_targets(
    states: np.ndarray,
    world_bounds: np.ndarray,
) -> np.ndarray:
    """Return image-normalized centre and wrapped heading unit vectors."""

    values = np.asarray(states)
    bounds = np.asarray(world_bounds, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 4
        or values.size == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("states must be finite non-empty [..., 4]")
    if (
        bounds.shape != (4,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] >= bounds[1]
        or bounds[2] >= bounds[3]
    ):
        raise ValueError(
            "world_bounds must contain finite "
            "(min_x, max_x, min_y, max_y)"
        )
    min_x, max_x, min_y, max_y = bounds
    scale = min(
        (IMAGE_SIZE - 1) / (max_x - min_x),
        (IMAGE_SIZE - 1) / (max_y - min_y),
    )
    used_width = (max_x - min_x) * scale
    used_height = (max_y - min_y) * scale
    offset_x = ((IMAGE_SIZE - 1) - used_width) / 2.0
    offset_y = ((IMAGE_SIZE - 1) - used_height) / 2.0
    positions = np.asarray(values[..., :2], dtype=np.float64)
    centre_x = (
        2.0 * (offset_x + (positions[..., 0] - min_x) * scale)
        / (IMAGE_SIZE - 1)
        - 1.0
    )
    centre_y = (
        2.0 * (offset_y + (max_y - positions[..., 1]) * scale)
        / (IMAGE_SIZE - 1)
        - 1.0
    )
    heading = np.asarray(values[..., 2], dtype=np.float64)
    targets = np.stack(
        (centre_x, centre_y, np.sin(heading), np.cos(heading)),
        axis=-1,
    )
    if not np.all(np.isfinite(targets)):
        raise ValueError("object slot targets must be finite")
    return targets.astype(np.float32)


def normalized_affine_to_raw(
    normalized_weight: np.ndarray,
    normalized_bias: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an affine map over normalized values to raw input space."""

    weight = np.asarray(normalized_weight, dtype=np.float64)
    bias = np.asarray(normalized_bias, dtype=np.float64)
    mean_values = np.asarray(mean, dtype=np.float64)
    std_values = np.asarray(std, dtype=np.float64)
    if (
        weight.ndim != 2
        or weight.shape[0] == 0
        or weight.shape[1] == 0
        or bias.shape != (weight.shape[0],)
        or mean_values.shape != (weight.shape[1],)
        or std_values.shape != (weight.shape[1],)
        or not np.all(np.isfinite(weight))
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(mean_values))
        or not np.all(np.isfinite(std_values))
        or np.any(std_values <= 0.0)
    ):
        raise ValueError(
            "affine weight, bias, mean, and positive std must have "
            "compatible finite shapes"
        )
    raw_weight = weight / std_values[None, :]
    raw_bias = bias - weight @ (mean_values / std_values)
    return raw_weight, raw_bias


class VisualObjectSlotFrameDataset(Dataset):
    """Return aligned image, object mask, and physical object-slot target."""

    def __init__(
        self,
        dataset: dict[str, np.ndarray],
        selected_episode_ids: np.ndarray,
    ) -> None:
        self.frame_indices = frame_indices_for_episode_ids(
            dataset,
            selected_episode_ids,
        )
        self.frame_indices.setflags(write=False)
        self._frames = np.asarray(dataset["frames"])
        self._targets = normalize_object_slot_targets(
            np.asarray(dataset["states"]),
            np.asarray(dataset["scene_world_bounds"]),
        )

    def __len__(self) -> int:
        return int(self.frame_indices.size)

    def __getitem__(
        self,
        item: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("object-slot dataset index must be an integer")
        try:
            position = operator.index(item)
        except TypeError:
            raise TypeError(
                "object-slot dataset index must be an integer"
            ) from None
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("object-slot dataset index out of range")
        frame_index = int(self.frame_indices[position])
        frame = self._frames[frame_index]
        return (
            frames_to_tensor(frame),
            renderer_object_masks(frame),
            torch.from_numpy(self._targets[frame_index].copy()),
        )


def _validate_slot_parameters(
    *,
    patch_size: int,
    hidden_size: int,
    initial_alpha: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    foreground_loss_weight: float,
    mask_loss_weight: float,
    centre_loss_weight: float,
    heading_loss_weight: float,
    seed: int,
) -> None:
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, int)
        or patch_size <= 0
        or patch_size > IMAGE_SIZE
        or patch_size % 2 == 0
    ):
        raise ValueError("patch_size must be a positive odd image-sized integer")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise ValueError("hidden_size must be a positive integer")
    if (
        not math.isfinite(initial_alpha)
        or initial_alpha <= 0.0
        or initial_alpha >= 1.0
    ):
        raise ValueError("initial_alpha must be between zero and one")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    for name, value, positive in (
        ("learning_rate", learning_rate, True),
        ("foreground_loss_weight", foreground_loss_weight, False),
        ("mask_loss_weight", mask_loss_weight, False),
        ("centre_loss_weight", centre_loss_weight, False),
        ("heading_loss_weight", heading_loss_weight, False),
    ):
        if (
            not math.isfinite(value)
            or value < 0.0
            or (positive and value == 0.0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be finite and {qualifier}")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def initialize_object_slot_autoencoder(
    source: SpatialConvAutoencoder,
    *,
    locator: str = "spatial_attention",
    centre_weight: np.ndarray | None = None,
    centre_bias: np.ndarray | None = None,
    patch_size: int,
    hidden_size: int,
    initial_alpha: float,
    seed: int,
) -> SpatialConvAutoencoder:
    """Copy a source model and freeze everything except one local slot head."""

    if not isinstance(source, SpatialConvAutoencoder):
        raise ValueError("source must be a spatial autoencoder")
    if source.object_residual_decoder or source.object_slot_decoder:
        raise ValueError("source must not already have an object decoder")
    if locator not in {"spatial_attention", "global_affine"}:
        raise ValueError(
            "locator must be 'spatial_attention' or 'global_affine'"
        )
    if locator == "global_affine":
        weight = np.asarray(centre_weight)
        bias = np.asarray(centre_bias)
        if (
            weight.shape != (2, source.latent_dim)
            or bias.shape != (2,)
            or not np.all(np.isfinite(weight))
            or not np.all(np.isfinite(bias))
        ):
            raise ValueError(
                "global affine centre weight and bias must be finite "
                f"[2, {source.latent_dim}] and [2]"
            )
    else:
        if centre_weight is not None or centre_bias is not None:
            raise ValueError(
                "centre weight and bias require global affine locator"
            )
        weight = np.empty((0, 0), dtype=np.float32)
        bias = np.empty((0,), dtype=np.float32)
    _validate_slot_parameters(
        patch_size=patch_size,
        hidden_size=hidden_size,
        initial_alpha=initial_alpha,
        epochs=1,
        batch_size=1,
        learning_rate=1.0,
        foreground_loss_weight=1.0,
        mask_loss_weight=1.0,
        centre_loss_weight=1.0,
        heading_loss_weight=1.0,
        seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate = SpatialConvAutoencoder(
            latent_channels=source.latent_channels,
            base_channels=source.base_channels,
            object_initial_alpha=initial_alpha,
            object_slot_decoder=True,
            object_slot_patch_size=patch_size,
            object_slot_hidden_size=hidden_size,
            object_slot_locator=locator,
        )
    candidate.encoder_convolutions.load_state_dict(
        source.encoder_convolutions.state_dict()
    )
    candidate.decoder_convolutions.load_state_dict(
        source.decoder_convolutions.state_dict()
    )
    for module in (
        candidate.encoder_convolutions,
        candidate.decoder_convolutions,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if locator == "global_affine":
        assert candidate.object_center is not None
        with torch.no_grad():
            candidate.object_center.weight.copy_(
                torch.as_tensor(weight, dtype=torch.float32)
            )
            candidate.object_center.bias.copy_(
                torch.as_tensor(bias, dtype=torch.float32)
            )
    candidate.eval()
    return candidate


def _validate_objective_inputs(
    model: SpatialConvAutoencoder,
    images: torch.Tensor,
    object_masks: torch.Tensor,
    slot_targets: torch.Tensor,
    *,
    foreground_loss_weight: float,
    mask_loss_weight: float,
    centre_loss_weight: float,
    heading_loss_weight: float,
) -> None:
    if (
        not isinstance(model, SpatialConvAutoencoder)
        or not model.object_slot_decoder
    ):
        raise ValueError("model must have an object slot decoder")
    if (
        images.ndim != 4
        or images.shape[1] != 3
        or tuple(images.shape[2:]) != (IMAGE_SIZE, IMAGE_SIZE)
    ):
        raise ValueError("images must have shape [B, 3, 64, 64]")
    if tuple(object_masks.shape) != (
        images.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise ValueError("object_masks must have shape [B, 1, 64, 64]")
    if tuple(slot_targets.shape) != (images.shape[0], 4):
        raise ValueError("slot_targets must have shape [B, 4]")
    if not bool(torch.all(torch.isfinite(images))) or not bool(
        torch.all(torch.isfinite(slot_targets))
    ):
        raise ValueError("images and slot_targets must be finite")
    if not bool(torch.all((object_masks == 0) | (object_masks == 1))):
        raise ValueError("object_masks must be binary")
    heading_norms = torch.linalg.vector_norm(slot_targets[:, 2:], dim=1)
    if not bool(torch.all(torch.abs(heading_norms - 1.0) <= 1e-4)):
        raise ValueError("slot target headings must be unit vectors")
    for name, value in (
        ("foreground_loss_weight", foreground_loss_weight),
        ("mask_loss_weight", mask_loss_weight),
        ("centre_loss_weight", centre_loss_weight),
        ("heading_loss_weight", heading_loss_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    object_count = torch.sum(object_masks)
    background_count = torch.sum(1.0 - object_masks)
    if float(object_count) == 0.0 or float(background_count) == 0.0:
        raise ValueError(
            "object-slot objective requires non-empty object and background "
            "regions"
        )


def object_slot_objective(
    model: SpatialConvAutoencoder,
    images: torch.Tensor,
    object_masks: torch.Tensor,
    slot_targets: torch.Tensor,
    *,
    foreground_loss_weight: float,
    mask_loss_weight: float,
    centre_loss_weight: float,
    heading_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return reconstruction, localization, centre, and heading supervision."""

    _validate_objective_inputs(
        model,
        images,
        object_masks,
        slot_targets,
        foreground_loss_weight=foreground_loss_weight,
        mask_loss_weight=mask_loss_weight,
        centre_loss_weight=centre_loss_weight,
        heading_loss_weight=heading_loss_weight,
    )
    components = model.decode_object_slot_components(model.encode(images))
    full_mse = torch.mean(torch.square(components.composite - images))
    expanded_object = object_masks.expand_as(images)
    foreground_object_mse = torch.sum(
        torch.square(components.foreground - images) * expanded_object
    ) / torch.sum(expanded_object)
    alpha_bce = functional.binary_cross_entropy(
        components.alpha,
        object_masks,
        reduction="none",
    )
    background_masks = 1.0 - object_masks
    balanced_alpha_bce = (
        torch.sum(alpha_bce * object_masks) / torch.sum(object_masks)
        + torch.sum(alpha_bce * background_masks)
        / torch.sum(background_masks)
    )
    centre_mse = torch.mean(
        torch.square(components.slot[:, :2] - slot_targets[:, :2])
    )
    heading_mse = torch.mean(
        torch.square(components.slot[:, 2:] - slot_targets[:, 2:])
    )
    total = (
        full_mse
        + foreground_loss_weight * foreground_object_mse
        + mask_loss_weight * balanced_alpha_bce
        + centre_loss_weight * centre_mse
        + heading_loss_weight * heading_mse
    )
    return total, {
        "full_mse": full_mse,
        "foreground_object_mse": foreground_object_mse,
        "balanced_alpha_bce": balanced_alpha_bce,
        "centre_mse": centre_mse,
        "heading_mse": heading_mse,
    }


def _mean_object_slot_loss(
    model: SpatialConvAutoencoder,
    loader: DataLoader,
    *,
    foreground_loss_weight: float,
    mask_loss_weight: float,
    centre_loss_weight: float,
    heading_loss_weight: float,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, masks, targets in loader:
            loss, _ = object_slot_objective(
                model,
                images,
                masks,
                targets,
                foreground_loss_weight=foreground_loss_weight,
                mask_loss_weight=mask_loss_weight,
                centre_loss_weight=centre_loss_weight,
                heading_loss_weight=heading_loss_weight,
            )
            batch_count = int(images.shape[0])
            loss_sum += float(loss) * batch_count
            sample_count += batch_count
    if sample_count == 0:
        raise ValueError("object-slot validation data must not be empty")
    return loss_sum / sample_count


def train_object_slot_decoder(
    source: SpatialConvAutoencoder,
    dataset: dict[str, np.ndarray],
    *,
    split_episode_ids: dict[str, np.ndarray],
    locator: str = "spatial_attention",
    centre_weight: np.ndarray | None = None,
    centre_bias: np.ndarray | None = None,
    patch_size: int = 11,
    hidden_size: int = 64,
    initial_alpha: float = 0.01,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    foreground_loss_weight: float = 1.0,
    mask_loss_weight: float = 0.01,
    centre_loss_weight: float = 1.0,
    heading_loss_weight: float = 0.1,
    seed: int = 0,
) -> PhaseTrainingResult:
    """Train one local object slot and restore its best validation state."""

    _validate_slot_parameters(
        patch_size=patch_size,
        hidden_size=hidden_size,
        initial_alpha=initial_alpha,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        foreground_loss_weight=foreground_loss_weight,
        mask_loss_weight=mask_loss_weight,
        centre_loss_weight=centre_loss_weight,
        heading_loss_weight=heading_loss_weight,
        seed=seed,
    )
    train_data = VisualObjectSlotFrameDataset(
        dataset,
        split_episode_ids["train"],
    )
    validation_data = VisualObjectSlotFrameDataset(
        dataset,
        split_episode_ids["validation"],
    )
    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("object-slot train and validation data must not be empty")
    model = initialize_object_slot_autoencoder(
        source,
        locator=locator,
        centre_weight=centre_weight,
        centre_bias=centre_bias,
        patch_size=patch_size,
        hidden_size=hidden_size,
        initial_alpha=initial_alpha,
        seed=seed,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("object-slot decoder has no trainable parameters")
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_epoch = 0
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch_index in range(epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, masks, targets in train_loader:
            optimizer.zero_grad()
            loss, _ = object_slot_objective(
                model,
                images,
                masks,
                targets,
                foreground_loss_weight=foreground_loss_weight,
                mask_loss_weight=mask_loss_weight,
                centre_loss_weight=centre_loss_weight,
                heading_loss_weight=heading_loss_weight,
            )
            if not torch.isfinite(loss):
                raise ValueError("object-slot training loss is non-finite")
            loss.backward()
            optimizer.step()
            batch_count = int(images.shape[0])
            loss_sum += float(loss.detach()) * batch_count
            sample_count += batch_count
        train_losses.append(loss_sum / sample_count)
        validation_loss = _mean_object_slot_loss(
            model,
            validation_loader,
            foreground_loss_weight=foreground_loss_weight,
            mask_loss_weight=mask_loss_weight,
            centre_loss_weight=centre_loss_weight,
            heading_loss_weight=heading_loss_weight,
        )
        validation_losses.append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch_index + 1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("object-slot training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return PhaseTrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
    )


def evaluate_object_slot_decoder(
    model: SpatialConvAutoencoder,
    dataset: dict[str, np.ndarray],
    *,
    selected_episode_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float | int]:
    """Measure direct state readout and alpha localization on held-out frames."""

    if not model.object_slot_decoder:
        raise ValueError("model must have an object slot decoder")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frame_data = VisualObjectSlotFrameDataset(
        dataset,
        selected_episode_ids,
    )
    if len(frame_data) == 0:
        raise ValueError("object-slot evaluation data must not be empty")
    loader = DataLoader(
        frame_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    centre_error_sum = 0.0
    heading_error_sum = 0.0
    true_positive = 0
    predicted_positive = 0
    target_positive = 0
    union = 0
    object_alpha_sum = 0.0
    background_alpha_sum = 0.0
    background_count = 0
    max_support_pixels = 0
    model.eval()
    with torch.no_grad():
        for images, masks, targets in loader:
            decoded = model.decode_object_slot_components(model.encode(images))
            centre_errors = torch.linalg.vector_norm(
                decoded.slot[:, :2] - targets[:, :2],
                dim=1,
            ) * ((IMAGE_SIZE - 1) / 2.0)
            heading_cosine = torch.clamp(
                torch.sum(decoded.slot[:, 2:] * targets[:, 2:], dim=1),
                min=-1.0,
                max=1.0,
            )
            heading_errors = torch.rad2deg(torch.acos(heading_cosine))
            centre_error_sum += float(torch.sum(centre_errors))
            heading_error_sum += float(torch.sum(heading_errors))
            predicted = decoded.alpha >= 0.5
            target = masks.to(dtype=torch.bool)
            true_positive += int(torch.sum(predicted & target))
            predicted_positive += int(torch.sum(predicted))
            target_positive += int(torch.sum(target))
            union += int(torch.sum(predicted | target))
            object_alpha_sum += float(torch.sum(decoded.alpha * masks))
            background = 1.0 - masks
            background_alpha_sum += float(
                torch.sum(decoded.alpha * background)
            )
            background_count += int(torch.sum(background))
            max_support_pixels = max(
                max_support_pixels,
                int(
                    torch.max(
                        torch.sum(decoded.support, dim=(1, 2, 3))
                    )
                ),
            )
    if target_positive == 0 or background_count == 0:
        raise ValueError(
            "object-slot evaluation requires object and background pixels"
        )
    return {
        "frames": len(frame_data),
        "object_pixels": target_positive,
        "background_pixels": background_count,
        "mean_centre_error_pixels": centre_error_sum / len(frame_data),
        "mean_heading_error_degrees": heading_error_sum / len(frame_data),
        "object_mask_iou": true_positive / union if union else 0.0,
        "object_mask_precision": (
            true_positive / predicted_positive
            if predicted_positive
            else 0.0
        ),
        "object_mask_recall": true_positive / target_positive,
        "mean_object_alpha": object_alpha_sum / target_positive,
        "mean_background_alpha": background_alpha_sum / background_count,
        "max_support_pixels": max_support_pixels,
    }
