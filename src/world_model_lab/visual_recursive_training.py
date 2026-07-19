"""Differentiable recursive training for visual latent dynamics."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import Normalizer
from .train_visual_latent_model import (
    PhaseTrainingResult,
    VisualAutoencoder,
    VisualLatentDynamics,
    _changed_pixel_mae_loss,
    _dynamics_batch_loss,
    _dynamics_image_supervision_tensors,
    _make_dynamics,
    _mean_dynamics_loss,
    _normalized_window_tensors,
)
from .visual_dataset import CONTEXT_FRAMES, validate_visual_dataset
from .visual_latent_data import LatentRolloutArrays, LatentWindowArrays
from .visual_object_position import LinearObjectPositionProbe
from .visual_object_position import normalize_object_positions


def _validate_recursive_tensors(
    *,
    context_latents: torch.Tensor,
    history_actions: torch.Tensor,
    rollout_actions: torch.Tensor,
    target_latents: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    if (
        context_latents.ndim != 3
        or context_latents.shape[0] == 0
        or context_latents.shape[1] != CONTEXT_FRAMES
        or context_latents.shape[2] == 0
    ):
        raise ValueError(
            "context_latents must have shape [B, 4, latent_dim]"
        )
    batch_size, _, latent_dim = context_latents.shape
    if history_actions.shape != (
        batch_size,
        CONTEXT_FRAMES - 1,
        2,
    ):
        raise ValueError("history_actions must have shape [B, 3, 2]")
    if (
        rollout_actions.ndim != 3
        or rollout_actions.shape[0] != batch_size
        or rollout_actions.shape[1] == 0
        or rollout_actions.shape[2] != 2
    ):
        raise ValueError("rollout_actions must have shape [B, H, 2]")
    horizon = int(rollout_actions.shape[1])
    if target_latents is not None and target_latents.shape != (
        batch_size,
        horizon,
        latent_dim,
    ):
        raise ValueError("target_latents must have shape [B, H, latent_dim]")
    values = [context_latents, history_actions, rollout_actions]
    if target_latents is not None:
        values.append(target_latents)
    if not all(bool(torch.all(torch.isfinite(value))) for value in values):
        raise ValueError("recursive rollout tensors must be finite")
    return batch_size, horizon, latent_dim


def recursive_normalized_latents(
    model: nn.Module,
    *,
    context_latents: torch.Tensor,
    history_actions: torch.Tensor,
    rollout_actions: torch.Tensor,
) -> torch.Tensor:
    """Recursively predict normalized latents while preserving autograd."""

    _, horizon, latent_dim = _validate_recursive_tensors(
        context_latents=context_latents,
        history_actions=history_actions,
        rollout_actions=rollout_actions,
    )
    context = context_latents
    history = history_actions
    predictions: list[torch.Tensor] = []
    for step in range(horizon):
        prediction = model(context, history, rollout_actions[:, step])
        if prediction.shape != (context.shape[0], latent_dim):
            raise ValueError(
                "dynamics produced latents with an invalid shape"
            )
        if not bool(torch.all(torch.isfinite(prediction))):
            raise ValueError("dynamics produced non-finite latents")
        predictions.append(prediction)
        context = torch.cat(
            (context[:, 1:], prediction[:, None, :]),
            dim=1,
        )
        history = torch.cat(
            (
                history[:, 1:],
                rollout_actions[:, step : step + 1],
            ),
            dim=1,
        )
    return torch.stack(predictions, dim=1)


def recursive_rollout_objective(
    model: nn.Module,
    *,
    context_latents: torch.Tensor,
    history_actions: torch.Tensor,
    rollout_actions: torch.Tensor,
    target_latents: torch.Tensor,
    latent_normalizer: Normalizer,
    decoder: VisualAutoencoder | nn.Module | None = None,
    target_frames_uint8: torch.Tensor | None = None,
    changed_masks: torch.Tensor | None = None,
    changed_pixel_loss_weight: float = 0.0,
    position_probe: LinearObjectPositionProbe | None = None,
    target_positions: torch.Tensor | None = None,
    object_position_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Return the mean latent-plus-image loss across recursive steps."""

    batch_size, horizon, latent_dim = _validate_recursive_tensors(
        context_latents=context_latents,
        history_actions=history_actions,
        rollout_actions=rollout_actions,
        target_latents=target_latents,
    )
    if (
        not math.isfinite(changed_pixel_loss_weight)
        or changed_pixel_loss_weight < 0.0
    ):
        raise ValueError(
            "changed_pixel_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(object_position_loss_weight)
        or object_position_loss_weight < 0.0
    ):
        raise ValueError(
            "object_position_loss_weight must be finite and non-negative"
        )
    mean = np.asarray(latent_normalizer.mean)
    std = np.asarray(latent_normalizer.std)
    if (
        mean.shape != (latent_dim,)
        or std.shape != (latent_dim,)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0.0)
    ):
        raise ValueError("latent_normalizer is incompatible with latents")
    if changed_pixel_loss_weight > 0.0:
        if (
            decoder is None
            or target_frames_uint8 is None
            or changed_masks is None
        ):
            raise ValueError(
                "positive changed-pixel loss requires decoder, target "
                "frames, and changed masks"
            )
        if (
            target_frames_uint8.ndim != 5
            or target_frames_uint8.shape[:2] != (batch_size, horizon)
            or target_frames_uint8.shape[-1] != 3
            or target_frames_uint8.dtype != torch.uint8
            or changed_masks.shape
            != target_frames_uint8.shape[:2]
            + target_frames_uint8.shape[2:4]
            or changed_masks.dtype != torch.bool
        ):
            raise ValueError("recursive image supervision is invalid")
    if object_position_loss_weight > 0.0:
        if position_probe is None or target_positions is None:
            raise ValueError(
                "positive object-position loss requires probe and targets"
            )
        if (
            target_positions.shape != (batch_size, horizon, 2)
            or not bool(torch.all(torch.isfinite(target_positions)))
        ):
            raise ValueError(
                "target_positions must be finite with shape [B, H, 2]"
            )

    predictions = recursive_normalized_latents(
        model,
        context_latents=context_latents,
        history_actions=history_actions,
        rollout_actions=rollout_actions,
    )
    latent_mean = torch.as_tensor(
        mean,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    latent_std = torch.as_tensor(
        std,
        dtype=predictions.dtype,
        device=predictions.device,
    )
    step_losses: list[torch.Tensor] = []
    for step in range(horizon):
        prediction = predictions[:, step]
        latent_mse = torch.mean(
            torch.square(prediction - target_latents[:, step])
        )
        if changed_pixel_loss_weight == 0.0:
            step_loss = latent_mse
        else:
            assert decoder is not None
            assert target_frames_uint8 is not None
            assert changed_masks is not None
            target_frames = (
                target_frames_uint8[:, step]
                .permute(0, 3, 1, 2)
                .to(dtype=prediction.dtype)
                .div(255.0)
            )
            masks = changed_masks[:, step, None].to(
                dtype=prediction.dtype
            )
            decoded = decoder.decode(
                prediction * latent_std + latent_mean
            )
            changed_mae = _changed_pixel_mae_loss(
                decoded,
                target_frames,
                masks,
            )
            step_loss = (
                latent_mse
                + changed_pixel_loss_weight * changed_mae
            )
        if object_position_loss_weight > 0.0:
            assert position_probe is not None
            assert target_positions is not None
            predicted_positions = position_probe(prediction)
            position_mse = torch.mean(
                torch.square(
                    predicted_positions - target_positions[:, step]
                )
            )
            step_loss = (
                step_loss
                + object_position_loss_weight * position_mse
            )
        step_losses.append(step_loss)
    loss = torch.mean(torch.stack(step_losses))
    if not bool(torch.isfinite(loss)):
        raise ValueError("recursive rollout objective is non-finite")
    return loss


@dataclass
class RecursiveDynamicsTrainingResult(PhaseTrainingResult):
    """Best dynamics plus one-step and recursive objective histories."""

    train_one_step_losses: list[float] = field(default_factory=list)
    train_rollout_losses: list[float] = field(default_factory=list)
    validation_one_step_losses: list[float] = field(default_factory=list)
    validation_rollout_losses: list[float] = field(default_factory=list)


def _normalized_rollout_tensors(
    arrays: LatentRolloutArrays,
    *,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = (
        latent_normalizer.normalize(arrays.context_latents),
        action_normalizer.normalize(arrays.history_actions),
        action_normalizer.normalize(arrays.rollout_actions),
        latent_normalizer.normalize(arrays.target_latents),
    )
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("normalized recursive rollout arrays must be finite")
    return tuple(
        torch.as_tensor(value, dtype=torch.float32)
        for value in values
    )


def _rollout_image_supervision_tensors(
    visual_dataset: Mapping[str, np.ndarray],
    arrays: LatentRolloutArrays,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_visual_dataset(visual_dataset)
    frames = np.asarray(visual_dataset["frames"])
    frame_count = int(frames.shape[0])
    initial_indices = np.asarray(
        arrays.initial_frame_indices,
        dtype=np.int64,
    )
    target_indices = np.asarray(
        arrays.target_frame_indices,
        dtype=np.int64,
    )
    all_indices = np.concatenate(
        (initial_indices, target_indices.reshape(-1))
    )
    if np.any(all_indices < 0) or np.any(all_indices >= frame_count):
        raise ValueError("recursive rollout frame indices are out of range")
    target_frames = frames[target_indices]
    comparison_frames = np.concatenate(
        (
            frames[initial_indices, None],
            target_frames[:, :-1],
        ),
        axis=1,
    )
    changed_masks = np.any(
        comparison_frames != target_frames,
        axis=4,
    )
    return (
        torch.from_numpy(np.ascontiguousarray(target_frames)),
        torch.from_numpy(np.ascontiguousarray(changed_masks)),
    )


def _mean_recursive_objective(
    model: VisualLatentDynamics,
    tensors: tuple[torch.Tensor, ...],
    *,
    batch_size: int,
    latent_normalizer: Normalizer,
    decoder: VisualAutoencoder | None,
    changed_pixel_loss_weight: float,
    position_probe: LinearObjectPositionProbe | None = None,
    object_position_loss_weight: float = 0.0,
) -> float:
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    loss_sum = 0.0
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            objective_batch = (
                batch[:-1]
                if object_position_loss_weight > 0.0
                else batch
            )
            loss = recursive_rollout_objective(
                model,
                context_latents=objective_batch[0],
                history_actions=objective_batch[1],
                rollout_actions=objective_batch[2],
                target_latents=objective_batch[3],
                latent_normalizer=latent_normalizer,
                decoder=decoder,
                target_frames_uint8=(
                    objective_batch[4]
                    if len(objective_batch) == 6
                    else None
                ),
                changed_masks=(
                    objective_batch[5]
                    if len(objective_batch) == 6
                    else None
                ),
                changed_pixel_loss_weight=changed_pixel_loss_weight,
                position_probe=position_probe,
                target_positions=(
                    batch[-1]
                    if object_position_loss_weight > 0.0
                    else None
                ),
                object_position_loss_weight=(
                    object_position_loss_weight
                ),
            )
            batch_count = int(batch[0].shape[0])
            loss_sum += float(loss) * batch_count
            sample_count += batch_count
    if sample_count == 0:
        raise ValueError("recursive validation data must not be empty")
    return loss_sum / sample_count


def _object_position_supervision_tensor(
    visual_dataset: Mapping[str, np.ndarray],
    target_frame_indices: np.ndarray,
) -> torch.Tensor:
    """Resolve frame-aligned physical states to normalized XY targets."""

    validate_visual_dataset(visual_dataset)
    indices = np.asarray(target_frame_indices, dtype=np.int64)
    frame_count = int(np.asarray(visual_dataset["states"]).shape[0])
    if (
        indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= frame_count)
    ):
        raise ValueError("object-position frame indices are out of range")
    positions = normalize_object_positions(
        np.asarray(visual_dataset["states"])[indices],
        np.asarray(visual_dataset["scene_world_bounds"]),
    )
    return torch.from_numpy(np.ascontiguousarray(positions))


def train_recursive_latent_dynamics(
    train_one_step_arrays: LatentWindowArrays,
    validation_one_step_arrays: LatentWindowArrays,
    train_rollout_arrays: LatentRolloutArrays,
    validation_rollout_arrays: LatentRolloutArrays,
    *,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
    spatial_latent_channels: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    decoder: VisualAutoencoder | None,
    visual_dataset: Mapping[str, np.ndarray] | None,
    changed_pixel_loss_weight: float,
    rollout_loss_weight: float,
    position_probe: LinearObjectPositionProbe | None = None,
    object_position_loss_weight: float = 0.0,
) -> RecursiveDynamicsTrainingResult:
    """Train fresh CNN dynamics with one-step and recursive objectives."""

    if (
        epochs <= 0
        or batch_size <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
    ):
        raise ValueError(
            "epochs, batch_size, and learning_rate must be positive"
        )
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if spatial_latent_channels <= 0 or hidden_size <= 0:
        raise ValueError(
            "spatial_latent_channels and hidden_size must be positive"
        )
    if (
        train_one_step_arrays.count == 0
        or validation_one_step_arrays.count == 0
        or train_rollout_arrays.count == 0
        or validation_rollout_arrays.count == 0
    ):
        raise ValueError("one-step and recursive arrays must not be empty")
    if train_rollout_arrays.horizon != validation_rollout_arrays.horizon:
        raise ValueError("train and validation rollout horizons must match")
    if train_rollout_arrays.horizon <= 1:
        raise ValueError("rollout horizon must be greater than one")
    latent_dim = int(train_one_step_arrays.context_latents.shape[2])
    if (
        validation_one_step_arrays.context_latents.shape[2] != latent_dim
        or train_rollout_arrays.context_latents.shape[2] != latent_dim
        or validation_rollout_arrays.context_latents.shape[2] != latent_dim
    ):
        raise ValueError("one-step and recursive latent dimensions must match")
    if (
        not math.isfinite(changed_pixel_loss_weight)
        or changed_pixel_loss_weight < 0.0
    ):
        raise ValueError(
            "changed_pixel_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(rollout_loss_weight)
        or rollout_loss_weight < 0.0
    ):
        raise ValueError(
            "rollout_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(object_position_loss_weight)
        or object_position_loss_weight < 0.0
    ):
        raise ValueError(
            "object_position_loss_weight must be finite and non-negative"
        )
    if changed_pixel_loss_weight > 0.0 and (
        decoder is None or visual_dataset is None
    ):
        raise ValueError(
            "positive changed-pixel loss requires decoder and visual_dataset"
        )
    if object_position_loss_weight > 0.0 and (
        position_probe is None or visual_dataset is None
    ):
        raise ValueError(
            "positive object-position loss requires probe and visual_dataset"
        )
    if (
        position_probe is not None
        and position_probe.latent_dim != latent_dim
    ):
        raise ValueError("position probe latent dimension does not match")

    train_one_step_tensors = _normalized_window_tensors(
        train_one_step_arrays,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    validation_one_step_tensors = _normalized_window_tensors(
        validation_one_step_arrays,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    train_rollout_tensors = _normalized_rollout_tensors(
        train_rollout_arrays,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    validation_rollout_tensors = _normalized_rollout_tensors(
        validation_rollout_arrays,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    if changed_pixel_loss_weight > 0.0:
        assert visual_dataset is not None
        train_one_step_tensors += _dynamics_image_supervision_tensors(
            visual_dataset,
            train_one_step_arrays,
        )
        validation_one_step_tensors += _dynamics_image_supervision_tensors(
            visual_dataset,
            validation_one_step_arrays,
        )
        train_rollout_tensors += _rollout_image_supervision_tensors(
            visual_dataset,
            train_rollout_arrays,
        )
        validation_rollout_tensors += _rollout_image_supervision_tensors(
            visual_dataset,
            validation_rollout_arrays,
        )
        assert decoder is not None
    if object_position_loss_weight > 0.0:
        assert visual_dataset is not None
        train_one_step_tensors += (
            _object_position_supervision_tensor(
                visual_dataset,
                train_one_step_arrays.target_frame_indices,
            ),
        )
        validation_one_step_tensors += (
            _object_position_supervision_tensor(
                visual_dataset,
                validation_one_step_arrays.target_frame_indices,
            ),
        )
        train_rollout_tensors += (
            _object_position_supervision_tensor(
                visual_dataset,
                train_rollout_arrays.target_frame_indices,
            ),
        )
        validation_rollout_tensors += (
            _object_position_supervision_tensor(
                visual_dataset,
                validation_rollout_arrays.target_frame_indices,
            ),
        )
    if decoder is not None:
        decoder.eval()
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    if position_probe is not None:
        position_probe.eval()
        for parameter in position_probe.parameters():
            parameter.requires_grad_(False)

    torch.manual_seed(seed)
    model = _make_dynamics(
        latent_layout="spatial",
        latent_dim=latent_dim,
        spatial_latent_channels=spatial_latent_channels,
        hidden_size=hidden_size,
        context_frames=CONTEXT_FRAMES,
        spatial_dynamics_architecture="cnn",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    one_step_generator = torch.Generator().manual_seed(seed)
    rollout_generator = torch.Generator().manual_seed(seed + 1)
    validation_one_step_loader = DataLoader(
        TensorDataset(*validation_one_step_tensors),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    train_one_step_losses: list[float] = []
    train_rollout_losses: list[float] = []
    validation_one_step_losses: list[float] = []
    validation_rollout_losses: list[float] = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    train_count = train_one_step_arrays.count
    rollout_count = train_rollout_arrays.count
    for epoch_index in range(epochs):
        one_step_permutation = torch.randperm(
            train_count,
            generator=one_step_generator,
        )
        rollout_permutation = torch.randperm(
            rollout_count,
            generator=rollout_generator,
        )
        model.train()
        epoch_one_step_loss = 0.0
        epoch_rollout_loss = 0.0
        for start in range(0, train_count, batch_size):
            one_step_indices = one_step_permutation[
                start : start + batch_size
            ]
            batch_count = int(one_step_indices.numel())
            rollout_offsets = (
                torch.arange(batch_count, dtype=torch.int64) + start
            ) % rollout_count
            rollout_indices = rollout_permutation[rollout_offsets]
            one_step_batch = tuple(
                tensor[one_step_indices]
                for tensor in train_one_step_tensors
            )
            rollout_batch = tuple(
                tensor[rollout_indices]
                for tensor in train_rollout_tensors
            )
            one_step_objective_batch = (
                one_step_batch[:-1]
                if object_position_loss_weight > 0.0
                else one_step_batch
            )
            rollout_objective_batch = (
                rollout_batch[:-1]
                if object_position_loss_weight > 0.0
                else rollout_batch
            )
            optimizer.zero_grad()
            one_step_loss = _dynamics_batch_loss(
                model,
                one_step_objective_batch,
                latent_normalizer=latent_normalizer,
                decoder=decoder,
                changed_pixel_loss_weight=changed_pixel_loss_weight,
                position_probe=position_probe,
                target_positions=(
                    one_step_batch[-1]
                    if object_position_loss_weight > 0.0
                    else None
                ),
                object_position_loss_weight=(
                    object_position_loss_weight
                ),
            )
            rollout_loss = recursive_rollout_objective(
                model,
                context_latents=rollout_objective_batch[0],
                history_actions=rollout_objective_batch[1],
                rollout_actions=rollout_objective_batch[2],
                target_latents=rollout_objective_batch[3],
                latent_normalizer=latent_normalizer,
                decoder=decoder,
                target_frames_uint8=(
                    rollout_objective_batch[4]
                    if len(rollout_objective_batch) == 6
                    else None
                ),
                changed_masks=(
                    rollout_objective_batch[5]
                    if len(rollout_objective_batch) == 6
                    else None
                ),
                changed_pixel_loss_weight=changed_pixel_loss_weight,
                position_probe=position_probe,
                target_positions=(
                    rollout_batch[-1]
                    if object_position_loss_weight > 0.0
                    else None
                ),
                object_position_loss_weight=(
                    object_position_loss_weight
                ),
            )
            total_loss = (
                one_step_loss + rollout_loss_weight * rollout_loss
            )
            if not bool(torch.isfinite(total_loss)):
                raise ValueError("recursive training loss is non-finite")
            total_loss.backward()
            optimizer.step()
            epoch_one_step_loss += (
                float(one_step_loss.detach()) * batch_count
            )
            epoch_rollout_loss += (
                float(rollout_loss.detach()) * batch_count
            )
        train_one_step = epoch_one_step_loss / train_count
        train_rollout = epoch_rollout_loss / train_count
        train_total = (
            train_one_step + rollout_loss_weight * train_rollout
        )
        model.eval()
        validation_one_step = _mean_dynamics_loss(
            model,
            validation_one_step_loader,
            latent_normalizer=latent_normalizer,
            decoder=decoder,
            changed_pixel_loss_weight=changed_pixel_loss_weight,
            position_probe=position_probe,
            object_position_loss_weight=object_position_loss_weight,
        )
        validation_rollout = _mean_recursive_objective(
            model,
            validation_rollout_tensors,
            batch_size=batch_size,
            latent_normalizer=latent_normalizer,
            decoder=decoder,
            changed_pixel_loss_weight=changed_pixel_loss_weight,
            position_probe=position_probe,
            object_position_loss_weight=object_position_loss_weight,
        )
        validation_total = (
            validation_one_step
            + rollout_loss_weight * validation_rollout
        )
        train_losses.append(train_total)
        validation_losses.append(validation_total)
        train_one_step_losses.append(train_one_step)
        train_rollout_losses.append(train_rollout)
        validation_one_step_losses.append(validation_one_step)
        validation_rollout_losses.append(validation_rollout)
        if validation_total < best_validation_loss:
            best_validation_loss = validation_total
            best_epoch = epoch_index + 1
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is None:
        raise RuntimeError(
            "recursive dynamics training did not produce a checkpoint"
        )
    model.load_state_dict(best_state_dict)
    model.eval()
    return RecursiveDynamicsTrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        train_one_step_losses=train_one_step_losses,
        train_rollout_losses=train_rollout_losses,
        validation_one_step_losses=validation_one_step_losses,
        validation_rollout_losses=validation_rollout_losses,
    )
