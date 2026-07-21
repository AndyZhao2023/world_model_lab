"""Train an object-only alpha/foreground head over a frozen spatial decoder."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from .train_visual_latent_model import PhaseTrainingResult
from .visual_latent_data import VisualObjectFrameDataset
from .visual_latent_model import SpatialConvAutoencoder


def _validate_residual_parameters(
    *,
    head_channels: int,
    initial_alpha: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    foreground_loss_weight: float,
    mask_loss_weight: float,
    seed: int,
) -> None:
    if (
        isinstance(head_channels, bool)
        or not isinstance(head_channels, int)
        or head_channels <= 0
    ):
        raise ValueError("head_channels must be a positive integer")
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


def initialize_object_residual_autoencoder(
    source: SpatialConvAutoencoder,
    *,
    head_channels: int,
    initial_alpha: float,
    seed: int,
) -> SpatialConvAutoencoder:
    """Copy the source representation and freeze everything but a new head."""

    if not isinstance(source, SpatialConvAutoencoder):
        raise ValueError("source must be a spatial autoencoder")
    if source.object_residual_decoder:
        raise ValueError("source must not already have an object residual head")
    _validate_residual_parameters(
        head_channels=head_channels,
        initial_alpha=initial_alpha,
        epochs=1,
        batch_size=1,
        learning_rate=1.0,
        foreground_loss_weight=1.0,
        mask_loss_weight=1.0,
        seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate = SpatialConvAutoencoder(
            latent_channels=source.latent_channels,
            base_channels=source.base_channels,
            object_residual_decoder=True,
            object_head_channels=head_channels,
            object_initial_alpha=initial_alpha,
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
    candidate.eval()
    return candidate


def object_residual_objective(
    model: SpatialConvAutoencoder,
    images: torch.Tensor,
    object_masks: torch.Tensor,
    *,
    foreground_loss_weight: float,
    mask_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return full, object-foreground, and balanced-mask supervision."""

    if (
        not isinstance(model, SpatialConvAutoencoder)
        or not model.object_residual_decoder
    ):
        raise ValueError("model must have an object residual decoder")
    if (
        images.ndim != 4
        or images.shape[1] != 3
        or tuple(images.shape[2:]) != (64, 64)
    ):
        raise ValueError("images must have shape [B, 3, 64, 64]")
    expected_mask_shape = (images.shape[0], 1, 64, 64)
    if tuple(object_masks.shape) != expected_mask_shape:
        raise ValueError("object_masks must have shape [B, 1, 64, 64]")
    if not torch.all(torch.isfinite(images)):
        raise ValueError("images must be finite")
    if not bool(torch.all((object_masks == 0) | (object_masks == 1))):
        raise ValueError("object_masks must be binary")
    for name, value in (
        ("foreground_loss_weight", foreground_loss_weight),
        ("mask_loss_weight", mask_loss_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    object_count = torch.sum(object_masks)
    background_masks = 1.0 - object_masks
    background_count = torch.sum(background_masks)
    if float(object_count) == 0.0 or float(background_count) == 0.0:
        raise ValueError(
            "residual objective requires non-empty object and background "
            "regions"
        )

    latents = model.encode(images)
    _, foreground, mask_logits, composite = model.decode_components(latents)
    full_mse = torch.mean(torch.square(composite - images))
    expanded_object = object_masks.expand_as(images)
    foreground_object_mse = torch.sum(
        torch.square(foreground - images) * expanded_object
    ) / torch.sum(expanded_object)
    mask_bce_values = functional.binary_cross_entropy_with_logits(
        mask_logits,
        object_masks,
        reduction="none",
    )
    balanced_mask_bce = (
        torch.sum(mask_bce_values * object_masks) / object_count
        + torch.sum(mask_bce_values * background_masks) / background_count
    )
    total = (
        full_mse
        + foreground_loss_weight * foreground_object_mse
        + mask_loss_weight * balanced_mask_bce
    )
    return total, {
        "full_mse": full_mse,
        "foreground_object_mse": foreground_object_mse,
        "balanced_mask_bce": balanced_mask_bce,
    }


def _mean_residual_loss(
    model: SpatialConvAutoencoder,
    loader: DataLoader,
    *,
    foreground_loss_weight: float,
    mask_loss_weight: float,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, masks in loader:
            loss, _ = object_residual_objective(
                model,
                images,
                masks,
                foreground_loss_weight=foreground_loss_weight,
                mask_loss_weight=mask_loss_weight,
            )
            batch_count = int(images.shape[0])
            loss_sum += float(loss) * batch_count
            sample_count += batch_count
    if sample_count == 0:
        raise ValueError("residual validation data must not be empty")
    return loss_sum / sample_count


def train_object_residual_decoder(
    source: SpatialConvAutoencoder,
    dataset: dict[str, np.ndarray],
    *,
    split_episode_ids: dict[str, np.ndarray],
    head_channels: int = 16,
    initial_alpha: float = 0.01,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    foreground_loss_weight: float = 1.0,
    mask_loss_weight: float = 0.01,
    seed: int = 0,
) -> PhaseTrainingResult:
    """Train only a new object head and restore its best validation state."""

    _validate_residual_parameters(
        head_channels=head_channels,
        initial_alpha=initial_alpha,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        foreground_loss_weight=foreground_loss_weight,
        mask_loss_weight=mask_loss_weight,
        seed=seed,
    )
    train_data = VisualObjectFrameDataset(
        dataset,
        split_episode_ids["train"],
    )
    validation_data = VisualObjectFrameDataset(
        dataset,
        split_episode_ids["validation"],
    )
    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("residual train and validation data must not be empty")
    model = initialize_object_residual_autoencoder(
        source,
        head_channels=head_channels,
        initial_alpha=initial_alpha,
        seed=seed,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("object residual head has no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )
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
        for images, masks in train_loader:
            optimizer.zero_grad()
            loss, _ = object_residual_objective(
                model,
                images,
                masks,
                foreground_loss_weight=foreground_loss_weight,
                mask_loss_weight=mask_loss_weight,
            )
            if not torch.isfinite(loss):
                raise ValueError("object residual training loss is non-finite")
            loss.backward()
            optimizer.step()
            batch_count = int(images.shape[0])
            loss_sum += float(loss.detach()) * batch_count
            sample_count += batch_count
        train_loss = loss_sum / sample_count
        validation_loss = _mean_residual_loss(
            model,
            validation_loader,
            foreground_loss_weight=foreground_loss_weight,
            mask_loss_weight=mask_loss_weight,
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch_index + 1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("object residual training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return PhaseTrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
    )


def evaluate_object_residual_mask(
    model: SpatialConvAutoencoder,
    dataset: dict[str, np.ndarray],
    *,
    selected_episode_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float | int]:
    """Measure held-out alpha localization against exact renderer masks."""

    if not model.object_residual_decoder:
        raise ValueError("model must have an object residual decoder")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frame_data = VisualObjectFrameDataset(dataset, selected_episode_ids)
    if len(frame_data) == 0:
        raise ValueError("residual evaluation data must not be empty")
    loader = DataLoader(
        frame_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    true_positive = 0
    predicted_positive = 0
    target_positive = 0
    union = 0
    object_alpha_sum = 0.0
    background_alpha_sum = 0.0
    background_count = 0
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            latents = model.encode(images)
            _, _, mask_logits, _ = model.decode_components(latents)
            alpha = torch.sigmoid(mask_logits)
            predicted = alpha >= 0.5
            target = masks.to(dtype=torch.bool)
            true_positive += int(torch.sum(predicted & target))
            predicted_positive += int(torch.sum(predicted))
            batch_target_count = int(torch.sum(target))
            target_positive += batch_target_count
            union += int(torch.sum(predicted | target))
            object_alpha_sum += float(torch.sum(alpha * masks))
            background = 1.0 - masks
            background_alpha_sum += float(torch.sum(alpha * background))
            background_count += int(torch.sum(background))
    if target_positive == 0 or background_count == 0:
        raise ValueError(
            "residual mask evaluation requires object and background pixels"
        )
    return {
        "frames": len(frame_data),
        "object_pixels": target_positive,
        "background_pixels": background_count,
        "object_mask_iou": true_positive / union if union else 0.0,
        "object_mask_precision": (
            true_positive / predicted_positive
            if predicted_positive
            else 0.0
        ),
        "object_mask_recall": true_positive / target_positive,
        "mean_object_alpha": object_alpha_sum / target_positive,
        "mean_background_alpha": (
            background_alpha_sum / background_count
        ),
    }
