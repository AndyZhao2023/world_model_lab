"""Diagnose recursive visual latent rollouts and action sensitivity."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .dataset import Normalizer
from .diagnose_model import sha256_file
from .train_visual_latent_model import (
    LoadedVisualLatentModel,
    load_visual_latent_checkpoint,
)
from .visual_dataset import (
    CONTEXT_FRAMES,
    load_visual_dataset,
    validate_visual_dataset,
)
from .visual_latent_data import encode_all_frames
from .visual_latent_model import (
    SpatialConvAutoencoder,
    SpatialLatentDynamicsCNN,
)


@dataclass(frozen=True)
class VisualRolloutWindow:
    """One episode-local visual rollout start and its future targets."""

    episode_id: int
    start_step: int
    context_latents: np.ndarray
    history_actions: np.ndarray
    future_actions: np.ndarray
    target_latents: np.ndarray
    initial_frame_index: int
    target_frame_indices: np.ndarray

    def __post_init__(self) -> None:
        context = np.asarray(self.context_latents)
        future = np.asarray(self.future_actions)
        if (
            context.ndim != 2
            or context.shape[0] != CONTEXT_FRAMES
            or context.shape[1] == 0
        ):
            raise ValueError(
                "context_latents must have shape [4, latent_dim]"
            )
        latent_dim = int(context.shape[1])
        horizon = int(future.shape[0]) if future.ndim == 2 else 0
        expected_shapes = {
            "history_actions": (CONTEXT_FRAMES - 1, 2),
            "future_actions": (horizon, 2),
            "target_latents": (horizon, latent_dim),
            "target_frame_indices": (horizon,),
        }
        if horizon <= 0:
            raise ValueError("future_actions must have a positive horizon")
        for name, shape in expected_shapes.items():
            values = np.asarray(getattr(self, name))
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {list(shape)}")
        for name in (
            "context_latents",
            "history_actions",
            "future_actions",
            "target_latents",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise ValueError(f"{name} must contain finite values")
        target_indices = np.asarray(self.target_frame_indices)
        if target_indices.dtype != np.dtype(np.int64):
            raise ValueError("target_frame_indices must have dtype int64")
        if self.episode_id < 0 or self.start_step < CONTEXT_FRAMES - 1:
            raise ValueError("episode_id and start_step are invalid")
        if self.initial_frame_index < 0:
            raise ValueError("initial_frame_index must be non-negative")
        for name in (
            "context_latents",
            "history_actions",
            "future_actions",
            "target_latents",
            "target_frame_indices",
        ):
            values = np.asarray(getattr(self, name)).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class VisualRolloutSelection:
    """Selected rollout windows and episode eligibility metadata."""

    windows: tuple[VisualRolloutWindow, ...]
    eligible_episode_ids: np.ndarray
    skipped_episode_ids: np.ndarray


def _evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= 0 or limit <= 0:
        raise ValueError("count and limit must be positive")
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    if limit == 1:
        return np.asarray([0], dtype=np.int64)
    return np.rint(np.linspace(0, count - 1, limit)).astype(np.int64)


def select_visual_rollout_windows(
    *,
    dataset: Mapping[str, np.ndarray],
    latent_frames: np.ndarray,
    selected_episode_ids: np.ndarray,
    max_horizon: int,
    windows_per_episode: int,
) -> VisualRolloutSelection:
    """Select deterministic visual windows without crossing episodes."""

    validate_visual_dataset(dataset)
    if max_horizon <= 0 or windows_per_episode <= 0:
        raise ValueError(
            "max_horizon and windows_per_episode must be positive"
        )
    selected = np.asarray(selected_episode_ids)
    if (
        selected.ndim != 1
        or selected.size == 0
        or selected.dtype.kind not in "iu"
        or np.unique(selected).size != selected.size
    ):
        raise ValueError(
            "selected_episode_ids must be a non-empty unique integer vector"
        )
    frames = np.asarray(dataset["frames"])
    latents = np.asarray(latent_frames)
    if (
        latents.ndim != 2
        or latents.shape[0] != frames.shape[0]
        or latents.shape[1] == 0
        or not np.all(np.isfinite(latents))
    ):
        raise ValueError(
            "latent_frames must be finite [F, latent_dim] matching frames"
        )

    available_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    positions = {
        int(episode_id): index
        for index, episode_id in enumerate(available_ids.tolist())
    }
    frame_offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
    transition_offsets = np.asarray(
        dataset["transition_offsets"],
        dtype=np.int64,
    )
    actions = np.asarray(dataset["actions"])
    windows: list[VisualRolloutWindow] = []
    eligible_ids: list[int] = []
    skipped_ids: list[int] = []
    for episode_id_value in selected.tolist():
        episode_id = int(episode_id_value)
        if episode_id not in positions:
            raise ValueError(
                f"episode {episode_id} is missing from the visual dataset"
            )
        position = positions[episode_id]
        frame_start = int(frame_offsets[position])
        action_start = int(transition_offsets[position])
        transition_count = int(
            transition_offsets[position + 1]
            - transition_offsets[position]
        )
        valid_start_count = (
            transition_count
            - (CONTEXT_FRAMES - 1)
            - max_horizon
            + 1
        )
        if valid_start_count <= 0:
            skipped_ids.append(episode_id)
            continue
        eligible_ids.append(episode_id)
        for relative_start in _evenly_spaced_indices(
            valid_start_count,
            windows_per_episode,
        ):
            start_step = CONTEXT_FRAMES - 1 + int(relative_start)
            context_start = frame_start + start_step - CONTEXT_FRAMES + 1
            action_history_start = (
                action_start + start_step - CONTEXT_FRAMES + 1
            )
            future_action_start = action_start + start_step
            target_frame_start = frame_start + start_step + 1
            target_frame_indices = np.arange(
                target_frame_start,
                target_frame_start + max_horizon,
                dtype=np.int64,
            )
            windows.append(
                VisualRolloutWindow(
                    episode_id=episode_id,
                    start_step=start_step,
                    context_latents=latents[
                        context_start : context_start + CONTEXT_FRAMES
                    ],
                    history_actions=actions[
                        action_history_start : action_history_start
                        + CONTEXT_FRAMES
                        - 1
                    ],
                    future_actions=actions[
                        future_action_start : future_action_start
                        + max_horizon
                    ],
                    target_latents=latents[target_frame_indices],
                    initial_frame_index=frame_start + start_step,
                    target_frame_indices=target_frame_indices,
                )
            )
    if not windows:
        raise ValueError(
            "selected episodes are not long enough for max_horizon"
        )
    eligible = np.asarray(eligible_ids, dtype=np.int64)
    skipped = np.asarray(skipped_ids, dtype=np.int64)
    eligible.setflags(write=False)
    skipped.setflags(write=False)
    return VisualRolloutSelection(
        windows=tuple(windows),
        eligible_episode_ids=eligible,
        skipped_episode_ids=skipped,
    )


def _validate_rollout_arrays(
    *,
    context_latents: np.ndarray,
    history_actions: np.ndarray,
    future_actions: np.ndarray,
    target_latents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    context = np.asarray(context_latents)
    history = np.asarray(history_actions)
    future = np.asarray(future_actions)
    if (
        context.ndim != 3
        or context.shape[1] != CONTEXT_FRAMES
        or context.shape[2] == 0
    ):
        raise ValueError(
            "context_latents must have shape [N, 4, latent_dim]"
        )
    count, _, latent_dim = context.shape
    if history.shape != (count, CONTEXT_FRAMES - 1, 2):
        raise ValueError("history_actions must have shape [N, 3, 2]")
    if future.ndim != 3 or future.shape[0] != count or future.shape[2] != 2:
        raise ValueError("future_actions must have shape [N, H, 2]")
    if future.shape[1] == 0:
        raise ValueError("future_actions horizon must be positive")
    target = None if target_latents is None else np.asarray(target_latents)
    if target is not None and target.shape != (
        count,
        future.shape[1],
        latent_dim,
    ):
        raise ValueError("target_latents must have shape [N, H, latent_dim]")
    values = (context, history, future)
    if target is not None:
        values += (target,)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("visual rollout arrays must be finite")
    return context, history, future, target


def _normalized_rollout_tensors(
    *,
    context_latents: np.ndarray,
    history_actions: np.ndarray,
    future_actions: np.ndarray,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = latent_normalizer.normalize(context_latents)
    history = action_normalizer.normalize(history_actions)
    future = action_normalizer.normalize(future_actions)
    return (
        torch.as_tensor(context, dtype=torch.float32),
        torch.as_tensor(history, dtype=torch.float32),
        torch.as_tensor(future, dtype=torch.float32),
    )


def rollout_normalized_latents(
    dynamics: nn.Module,
    *,
    context_latents: np.ndarray,
    history_actions: np.ndarray,
    future_actions: np.ndarray,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
) -> np.ndarray:
    """Recursively predict normalized latents under future actions."""

    context, history, future, _ = _validate_rollout_arrays(
        context_latents=context_latents,
        history_actions=history_actions,
        future_actions=future_actions,
    )
    context_tensor, history_tensor, future_tensor = (
        _normalized_rollout_tensors(
            context_latents=context,
            history_actions=history,
            future_actions=future,
            latent_normalizer=latent_normalizer,
            action_normalizer=action_normalizer,
        )
    )
    dynamics.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(future_tensor.shape[1]):
            prediction = dynamics(
                context_tensor,
                history_tensor,
                future_tensor[:, step],
            )
            if (
                prediction.shape
                != (context_tensor.shape[0], context_tensor.shape[2])
                or not torch.all(torch.isfinite(prediction))
            ):
                raise ValueError(
                    "dynamics produced invalid normalized latents"
                )
            predictions.append(prediction)
            context_tensor = torch.cat(
                (context_tensor[:, 1:], prediction[:, None, :]),
                dim=1,
            )
            history_tensor = torch.cat(
                (
                    history_tensor[:, 1:],
                    future_tensor[:, step : step + 1],
                ),
                dim=1,
            )
    return (
        torch.stack(predictions, dim=1)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def teacher_forced_normalized_latents(
    dynamics: nn.Module,
    *,
    context_latents: np.ndarray,
    target_latents: np.ndarray,
    history_actions: np.ndarray,
    future_actions: np.ndarray,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
) -> np.ndarray:
    """Predict each future latent from its true four-frame context."""

    context, history, future, targets = _validate_rollout_arrays(
        context_latents=context_latents,
        history_actions=history_actions,
        future_actions=future_actions,
        target_latents=target_latents,
    )
    assert targets is not None
    true_sequence = np.concatenate((context, targets), axis=1)
    action_sequence = np.concatenate((history, future), axis=1)
    predictions: list[np.ndarray] = []
    for step in range(future.shape[1]):
        prediction = rollout_normalized_latents(
            dynamics,
            context_latents=true_sequence[:, step : step + CONTEXT_FRAMES],
            history_actions=action_sequence[
                :,
                step : step + CONTEXT_FRAMES - 1,
            ],
            future_actions=future[:, step : step + 1],
            latent_normalizer=latent_normalizer,
            action_normalizer=action_normalizer,
        )
        predictions.append(prediction[:, 0])
    return np.stack(predictions, axis=1).astype(
        np.float32,
        copy=False,
    )


def _sattolo_permutation(count: int, *, seed: int) -> np.ndarray:
    """Return a deterministic single-cycle permutation with no fixed points."""

    if count < 2:
        raise ValueError("counterfactual permutation requires two windows")
    if seed < 0:
        raise ValueError("counterfactual seed must be non-negative")
    rng = np.random.default_rng(seed)
    permutation = np.arange(count, dtype=np.int64)
    for index in range(count - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        permutation[index], permutation[swap_index] = (
            permutation[swap_index],
            permutation[index],
        )
    return permutation


def _stack_rollout_windows(
    windows: Iterable[VisualRolloutWindow],
) -> dict[str, np.ndarray]:
    selected = tuple(windows)
    if not selected:
        raise ValueError("visual rollout windows must not be empty")
    horizon = selected[0].future_actions.shape[0]
    latent_dim = selected[0].context_latents.shape[1]
    if any(
        window.future_actions.shape != (horizon, 2)
        or window.target_latents.shape != (horizon, latent_dim)
        or window.context_latents.shape != (CONTEXT_FRAMES, latent_dim)
        for window in selected
    ):
        raise ValueError("visual rollout windows must share one shape")
    return {
        "context_latents": np.stack(
            [window.context_latents for window in selected]
        ),
        "history_actions": np.stack(
            [window.history_actions for window in selected]
        ),
        "future_actions": np.stack(
            [window.future_actions for window in selected]
        ),
        "target_latents": np.stack(
            [window.target_latents for window in selected]
        ),
        "episode_ids": np.asarray(
            [window.episode_id for window in selected],
            dtype=np.int64,
        ),
        "initial_frame_indices": np.asarray(
            [window.initial_frame_index for window in selected],
            dtype=np.int64,
        ),
        "target_frame_indices": np.stack(
            [window.target_frame_indices for window in selected]
        ),
    }


def _decode_normalized_rollout_latents(
    autoencoder: nn.Module,
    *,
    normalized_latents: np.ndarray,
    latent_normalizer: Normalizer,
    batch_size: int = 256,
) -> np.ndarray:
    values = np.asarray(normalized_latents)
    if (
        values.ndim != 3
        or values.shape[0] == 0
        or values.shape[1] == 0
        or values.shape[2] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "normalized_latents must be finite non-empty [N, H, D]"
        )
    if batch_size <= 0:
        raise ValueError("decode batch_size must be positive")
    count, horizon, latent_dim = values.shape
    raw = latent_normalizer.denormalize(
        values.reshape(count * horizon, latent_dim)
    )
    autoencoder.eval()
    decoded_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, raw.shape[0], batch_size):
            latent_tensor = torch.as_tensor(
                raw[start : start + batch_size],
                dtype=torch.float32,
            )
            decoded = autoencoder.decode(latent_tensor)
            if (
                decoded.ndim != 4
                or decoded.shape[0] != latent_tensor.shape[0]
                or decoded.shape[1] != 3
                or not torch.all(torch.isfinite(decoded))
            ):
                raise ValueError("autoencoder decoded invalid rollout frames")
            decoded_batches.append(
                decoded.cpu().numpy().astype(np.float32, copy=False)
            )
    decoded_values = np.concatenate(decoded_batches, axis=0)
    return decoded_values.reshape(
        count,
        horizon,
        *decoded_values.shape[1:],
    )


def evaluate_counterfactual_sensitivity(
    *,
    dynamics: nn.Module,
    autoencoder: nn.Module,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
    windows: Iterable[VisualRolloutWindow],
    seeds: Iterable[int],
    decode_batch_size: int = 256,
) -> dict[str, object]:
    """Measure divergence after replacing complete future action sequences."""

    arrays = _stack_rollout_windows(windows)
    seed_values = tuple(seeds)
    if (
        not seed_values
        or any(
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
            for seed in seed_values
        )
        or len(set(int(seed) for seed in seed_values)) != len(seed_values)
    ):
        raise ValueError(
            "counterfactual seeds must be unique non-negative integers"
        )
    count = int(arrays["episode_ids"].shape[0])
    if count < 2:
        raise ValueError(
            "counterfactual sensitivity requires at least two windows"
        )
    recorded_latents = rollout_normalized_latents(
        dynamics,
        context_latents=arrays["context_latents"],
        history_actions=arrays["history_actions"],
        future_actions=arrays["future_actions"],
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    recorded_frames = _decode_normalized_rollout_latents(
        autoencoder,
        normalized_latents=recorded_latents,
        latent_normalizer=latent_normalizer,
        batch_size=decode_batch_size,
    )
    metrics_by_seed: dict[str, list[np.ndarray]] = {
        "normalized_latent_rms": [],
        "decoded_pixel_mse": [],
        "decoded_pixel_mae": [],
    }
    for seed in seed_values:
        permutation = _sattolo_permutation(count, seed=int(seed))
        counterfactual_latents = rollout_normalized_latents(
            dynamics,
            context_latents=arrays["context_latents"],
            history_actions=arrays["history_actions"],
            future_actions=arrays["future_actions"][permutation],
            latent_normalizer=latent_normalizer,
            action_normalizer=action_normalizer,
        )
        counterfactual_frames = _decode_normalized_rollout_latents(
            autoencoder,
            normalized_latents=counterfactual_latents,
            latent_normalizer=latent_normalizer,
            batch_size=decode_batch_size,
        )
        latent_rms = np.sqrt(
            np.mean(
                np.square(
                    counterfactual_latents.astype(np.float64)
                    - recorded_latents
                ),
                axis=2,
            )
        )
        pixel_differences = (
            counterfactual_frames.astype(np.float64) - recorded_frames
        )
        window_metrics = {
            "normalized_latent_rms": latent_rms,
            "decoded_pixel_mse": np.mean(
                np.square(pixel_differences),
                axis=(2, 3, 4),
            ),
            "decoded_pixel_mae": np.mean(
                np.abs(pixel_differences),
                axis=(2, 3, 4),
            ),
        }
        for name, values in window_metrics.items():
            metrics_by_seed[name].append(
                _episode_macro_mean(
                    values,
                    episode_ids=arrays["episode_ids"],
                )
            )

    stacked = {
        name: np.stack(values)
        for name, values in metrics_by_seed.items()
    }
    horizon = recorded_latents.shape[1]
    return {
        "episodes": int(np.unique(arrays["episode_ids"]).size),
        "windows": count,
        "seeds": [int(seed) for seed in seed_values],
        "steps": {
            str(step + 1): {
                name: {
                    "mean": float(np.mean(values[:, step])),
                    "sample_std": (
                        float(np.std(values[:, step], ddof=1))
                        if len(seed_values) > 1
                        else 0.0
                    ),
                }
                for name, values in stacked.items()
            }
            for step in range(horizon)
        },
    }


def _episode_macro_mean(
    values: np.ndarray,
    *,
    episode_ids: np.ndarray,
) -> np.ndarray:
    """Average windows within episodes before averaging episodes."""

    array = np.asarray(values, dtype=np.float64)
    ids = np.asarray(episode_ids)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("episode metric values must have shape [N, H]")
    if (
        ids.shape != (array.shape[0],)
        or ids.dtype.kind not in "iu"
        or np.unique(ids).size == 0
    ):
        raise ValueError("episode_ids must be an integer vector matching N")
    per_episode = [
        np.mean(array[ids == episode_id], axis=0)
        for episode_id in np.unique(ids)
    ]
    result = np.mean(np.stack(per_episode), axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError("episode macro metrics must be finite")
    return result


def _changed_pixel_episode_macro_mae(
    *,
    predicted_frames: np.ndarray,
    true_target_frames: np.ndarray,
    comparison_frames: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(predicted_frames, dtype=np.float64)
    targets = np.asarray(true_target_frames)
    comparisons = np.asarray(comparison_frames)
    _, horizon, channels, _, _ = predictions.shape
    if (
        targets.shape != predictions.shape
        or comparisons.shape != predictions.shape
        or targets.dtype != np.dtype(np.uint8)
        or comparisons.dtype != np.dtype(np.uint8)
    ):
        raise ValueError(
            "true and comparison frames must be uint8 and match predictions"
        )
    target_values = targets.astype(np.float64) / 255.0
    errors = np.abs(predictions - target_values)
    changed = np.any(comparisons != targets, axis=2)
    ids = np.asarray(episode_ids)
    per_episode: list[np.ndarray] = []
    for episode_id in np.unique(ids):
        selected = ids == episode_id
        masks = changed[selected]
        selected_errors = errors[selected]
        numerators = np.sum(
            selected_errors * masks[:, :, None, :, :],
            axis=(0, 2, 3, 4),
        )
        denominators = (
            np.sum(masks, axis=(0, 2, 3), dtype=np.float64) * channels
        )
        per_episode.append(
            np.divide(
                numerators,
                denominators,
                out=np.zeros(horizon, dtype=np.float64),
                where=denominators > 0.0,
            )
        )
    result = np.mean(np.stack(per_episode), axis=0)
    if result.shape != (horizon,) or not np.all(np.isfinite(result)):
        raise ValueError("changed-pixel metrics must be finite")
    return result


def summarize_visual_predictions(
    *,
    predicted_normalized_latents: np.ndarray,
    target_normalized_latents: np.ndarray,
    predicted_frames: np.ndarray,
    true_initial_frames: np.ndarray,
    true_target_frames: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize dense visual rollout errors with episode-equal weighting."""

    predictions = np.asarray(predicted_normalized_latents)
    targets = np.asarray(target_normalized_latents)
    decoded = np.asarray(predicted_frames)
    initial = np.asarray(true_initial_frames)
    true_frames = np.asarray(true_target_frames)
    ids = np.asarray(episode_ids)
    if (
        predictions.ndim != 3
        or predictions.shape != targets.shape
        or predictions.shape[0] == 0
        or predictions.shape[1] == 0
        or predictions.shape[2] == 0
    ):
        raise ValueError(
            "predicted and target latents must share non-empty shape [N, H, D]"
        )
    count, horizon, _ = predictions.shape
    if (
        decoded.ndim != 5
        or decoded.shape[:2] != (count, horizon)
        or decoded.shape[2] != 3
        or initial.shape != (
            count,
            decoded.shape[2],
            decoded.shape[3],
            decoded.shape[4],
        )
        or true_frames.shape != decoded.shape
    ):
        raise ValueError("visual frame arrays are not aligned")
    if (
        initial.dtype != np.dtype(np.uint8)
        or true_frames.dtype != np.dtype(np.uint8)
    ):
        raise ValueError("true frames must use uint8 pixels")
    if ids.shape != (count,) or ids.dtype.kind not in "iu":
        raise ValueError("episode_ids must be an integer vector matching N")
    if not (
        np.all(np.isfinite(predictions))
        and np.all(np.isfinite(targets))
        and np.all(np.isfinite(decoded))
    ):
        raise ValueError("visual predictions must be finite")

    latent_mse = np.mean(
        np.square(predictions.astype(np.float64) - targets),
        axis=2,
    )
    target_pixels = true_frames.astype(np.float64) / 255.0
    pixel_mse = np.mean(
        np.square(decoded.astype(np.float64) - target_pixels),
        axis=(2, 3, 4),
    )
    transition_comparisons = np.concatenate(
        (initial[:, None], true_frames[:, :-1]),
        axis=1,
    )
    cumulative_comparisons = np.repeat(initial[:, None], horizon, axis=1)
    curves = {
        "normalized_latent_mse": _episode_macro_mean(
            latent_mse,
            episode_ids=ids,
        ),
        "pixel_mse": _episode_macro_mean(
            pixel_mse,
            episode_ids=ids,
        ),
        "transition_changed_pixel_mae": (
            _changed_pixel_episode_macro_mae(
                predicted_frames=decoded,
                true_target_frames=true_frames,
                comparison_frames=transition_comparisons,
                episode_ids=ids,
            )
        ),
        "cumulative_changed_pixel_mae": (
            _changed_pixel_episode_macro_mae(
                predicted_frames=decoded,
                true_target_frames=true_frames,
                comparison_frames=cumulative_comparisons,
                episode_ids=ids,
            )
        ),
    }
    return {
        str(step + 1): {
            "episodes": int(np.unique(ids).size),
            "windows": count,
            **{
                name: float(values[step])
                for name, values in curves.items()
            },
        }
        for step in range(horizon)
    }


def evaluate_visual_rollout_model(
    *,
    model: LoadedVisualLatentModel,
    windows: Iterable[VisualRolloutWindow],
    frames: np.ndarray,
    decode_batch_size: int = 256,
) -> dict[str, object]:
    """Evaluate teacher-forced and recursive predictions at every horizon."""

    arrays = _stack_rollout_windows(windows)
    frame_values = np.asarray(frames)
    if (
        frame_values.ndim != 4
        or frame_values.shape[1:] != (64, 64, 3)
        or frame_values.dtype != np.dtype(np.uint8)
    ):
        raise ValueError("frames must be uint8 [F, 64, 64, 3]")
    all_indices = np.concatenate(
        (
            arrays["initial_frame_indices"],
            arrays["target_frame_indices"].reshape(-1),
        )
    )
    if np.any(all_indices < 0) or np.any(all_indices >= frame_values.shape[0]):
        raise ValueError("rollout frame indices are outside the dataset")

    target_normalized = model.latent_normalizer.normalize(
        arrays["target_latents"]
    )
    teacher_latents = teacher_forced_normalized_latents(
        model.dynamics,
        context_latents=arrays["context_latents"],
        target_latents=arrays["target_latents"],
        history_actions=arrays["history_actions"],
        future_actions=arrays["future_actions"],
        latent_normalizer=model.latent_normalizer,
        action_normalizer=model.action_normalizer,
    )
    free_latents = rollout_normalized_latents(
        model.dynamics,
        context_latents=arrays["context_latents"],
        history_actions=arrays["history_actions"],
        future_actions=arrays["future_actions"],
        latent_normalizer=model.latent_normalizer,
        action_normalizer=model.action_normalizer,
    )
    teacher_frames = _decode_normalized_rollout_latents(
        model.autoencoder,
        normalized_latents=teacher_latents,
        latent_normalizer=model.latent_normalizer,
        batch_size=decode_batch_size,
    )
    free_frames = _decode_normalized_rollout_latents(
        model.autoencoder,
        normalized_latents=free_latents,
        latent_normalizer=model.latent_normalizer,
        batch_size=decode_batch_size,
    )
    initial_frames = np.transpose(
        frame_values[arrays["initial_frame_indices"]],
        (0, 3, 1, 2),
    )
    count, horizon = arrays["target_frame_indices"].shape
    true_target_frames = np.transpose(
        frame_values[arrays["target_frame_indices"].reshape(-1)].reshape(
            count,
            horizon,
            64,
            64,
            3,
        ),
        (0, 1, 4, 2, 3),
    )
    teacher = summarize_visual_predictions(
        predicted_normalized_latents=teacher_latents,
        target_normalized_latents=target_normalized,
        predicted_frames=teacher_frames,
        true_initial_frames=initial_frames,
        true_target_frames=true_target_frames,
        episode_ids=arrays["episode_ids"],
    )
    free = summarize_visual_predictions(
        predicted_normalized_latents=free_latents,
        target_normalized_latents=target_normalized,
        predicted_frames=free_frames,
        true_initial_frames=initial_frames,
        true_target_frames=true_target_frames,
        episode_ids=arrays["episode_ids"],
    )
    return {
        "episodes": int(np.unique(arrays["episode_ids"]).size),
        "windows": count,
        "steps": {
            step: {
                "episodes": teacher[step]["episodes"],
                "windows": teacher[step]["windows"],
                "teacher_forcing": {
                    name: value
                    for name, value in teacher[step].items()
                    if name not in {"episodes", "windows"}
                },
                "free_rollout": {
                    name: value
                    for name, value in free[step].items()
                    if name not in {"episodes", "windows"}
                },
            }
            for step in teacher
        },
    }


def _same_normalizer(left: Normalizer, right: Normalizer) -> bool:
    return np.array_equal(left.mean, right.mean) and np.array_equal(
        left.std,
        right.std,
    )


def _same_module_state(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name])
        for name in left_state
    )


def _validate_visual_checkpoint_pair(
    *,
    baseline: LoadedVisualLatentModel,
    aligned: LoadedVisualLatentModel,
    dataset_sha256: str,
) -> None:
    for name, model in (("baseline", baseline), ("aligned", aligned)):
        if not isinstance(
            model.autoencoder,
            SpatialConvAutoencoder,
        ) or not isinstance(model.dynamics, SpatialLatentDynamicsCNN):
            raise ValueError(
                f"{name} checkpoint must use a spatial CNN world model"
            )
        recorded_sha = model.dataset_metadata.get("sha256")
        if recorded_sha != dataset_sha256:
            raise ValueError(
                f"{name} dataset SHA-256 does not match the input dataset"
            )
    if not _same_module_state(
        baseline.autoencoder,
        aligned.autoencoder,
    ):
        raise ValueError("checkpoint autoencoder weights do not match")
    if not _same_normalizer(
        baseline.latent_normalizer,
        aligned.latent_normalizer,
    ):
        raise ValueError("checkpoint latent normalizer values do not match")
    if not _same_normalizer(
        baseline.action_normalizer,
        aligned.action_normalizer,
    ):
        raise ValueError("checkpoint action normalizer values do not match")
    if not np.array_equal(
        baseline.split_episode_ids["test"],
        aligned.split_episode_ids["test"],
    ):
        raise ValueError("checkpoint test split episode IDs do not match")
    if (
        baseline.autoencoder.latent_dim
        != aligned.autoencoder.latent_dim
        or baseline.autoencoder.latent_channels
        != aligned.autoencoder.latent_channels
        or baseline.autoencoder.base_channels
        != aligned.autoencoder.base_channels
        or baseline.dynamics.latent_dim != aligned.dynamics.latent_dim
        or baseline.dynamics.hidden_channels
        != aligned.dynamics.hidden_channels
        or baseline.dynamics.context_frames != aligned.dynamics.context_frames
    ):
        raise ValueError("checkpoint spatial latent dimensions do not match")


def _validate_diagnostic_protocol(
    *,
    horizons: Iterable[int],
    windows_per_episode: int,
    counterfactual_seeds: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    horizon_values = tuple(horizons)
    if (
        not horizon_values
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in horizon_values
        )
    ):
        raise ValueError("horizons must be positive integers")
    normalized_horizons = tuple(int(value) for value in horizon_values)
    if (
        len(set(normalized_horizons)) != len(normalized_horizons)
        or any(
            left >= right
            for left, right in zip(
                normalized_horizons,
                normalized_horizons[1:],
            )
        )
    ):
        raise ValueError("horizons must be unique and strictly increasing")
    if (
        isinstance(windows_per_episode, (bool, np.bool_))
        or not isinstance(windows_per_episode, (int, np.integer))
        or int(windows_per_episode) <= 0
    ):
        raise ValueError("windows_per_episode must be a positive integer")
    seed_values = tuple(counterfactual_seeds)
    if (
        not seed_values
        or any(
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
            for seed in seed_values
        )
        or len(set(int(seed) for seed in seed_values)) != len(seed_values)
    ):
        raise ValueError(
            "counterfactual_seeds must be unique non-negative integers"
        )
    return normalized_horizons, tuple(int(seed) for seed in seed_values)


def _safe_relative_percent(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else float(
            (candidate - reference)
            / np.finfo(np.float64).eps
            * 100.0
        )
    return float((candidate - reference) / reference * 100.0)


def _safe_ratio(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 1.0 if candidate == 0.0 else float(
            candidate / np.finfo(np.float64).eps
        )
    return float(candidate / reference)


def _build_model_comparison(
    *,
    baseline: Mapping[str, object],
    aligned: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    baseline_steps = baseline["steps"]
    aligned_steps = aligned["steps"]
    baseline_counterfactual = baseline["counterfactual"]
    aligned_counterfactual = aligned["counterfactual"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            baseline_steps,
            aligned_steps,
            baseline_counterfactual,
            aligned_counterfactual,
        )
    ):
        raise ValueError("model diagnostic records are malformed")
    assert isinstance(baseline_steps, Mapping)
    assert isinstance(aligned_steps, Mapping)
    assert isinstance(baseline_counterfactual, Mapping)
    assert isinstance(aligned_counterfactual, Mapping)
    baseline_cf_steps = baseline_counterfactual["steps"]
    aligned_cf_steps = aligned_counterfactual["steps"]
    if not isinstance(baseline_cf_steps, Mapping) or not isinstance(
        aligned_cf_steps,
        Mapping,
    ):
        raise ValueError("counterfactual diagnostic records are malformed")

    result: dict[str, dict[str, object]] = {}
    for step in baseline_steps:
        baseline_step = baseline_steps[step]
        aligned_step = aligned_steps[step]
        baseline_cf = baseline_cf_steps[step]
        aligned_cf = aligned_cf_steps[step]
        if not all(
            isinstance(value, Mapping)
            for value in (
                baseline_step,
                aligned_step,
                baseline_cf,
                aligned_cf,
            )
        ):
            raise ValueError("model step records are malformed")
        baseline_free = baseline_step["free_rollout"]
        aligned_free = aligned_step["free_rollout"]
        if not isinstance(baseline_free, Mapping) or not isinstance(
            aligned_free,
            Mapping,
        ):
            raise ValueError("free rollout records are malformed")
        result[str(step)] = {
            "free_rollout": {
                name: {
                    "absolute": float(aligned_free[name])
                    - float(reference),
                    "relative_percent": _safe_relative_percent(
                        float(aligned_free[name]),
                        float(reference),
                    ),
                }
                for name, reference in baseline_free.items()
            },
            "counterfactual_sensitivity_ratio": {
                name: _safe_ratio(
                    float(aligned_cf[name]["mean"]),
                    float(reference["mean"]),
                )
                for name, reference in baseline_cf.items()
            },
        }
    return result


def _write_json(path: Path, value: Mapping[str, object]) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def plot_visual_rollout_comparison(
    *,
    metrics: Mapping[str, object],
    output_path: Path | str,
) -> Path:
    models = metrics["models"]
    if not isinstance(models, Mapping):
        raise ValueError("metrics models record is malformed")
    colors = {"baseline": "#4C78A8", "aligned": "#F58518"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for model_name in ("baseline", "aligned"):
        record = models[model_name]
        if not isinstance(record, Mapping):
            raise ValueError("model metrics record is malformed")
        steps = record["steps"]
        counterfactual = record["counterfactual"]
        if not isinstance(steps, Mapping) or not isinstance(
            counterfactual,
            Mapping,
        ):
            raise ValueError("model curve records are malformed")
        step_values = np.asarray(
            [int(step) for step in steps],
            dtype=np.int64,
        )
        for axis, metric_name, title, ylabel in (
            (
                axes[0, 0],
                "normalized_latent_mse",
                "Normalized latent rollout error",
                "MSE",
            ),
            (
                axes[0, 1],
                "cumulative_changed_pixel_mae",
                "Cumulative changed-pixel error",
                "MAE",
            ),
        ):
            for mode, style, label in (
                ("teacher_forcing", "--", "teacher"),
                ("free_rollout", "-", "free"),
            ):
                axis.plot(
                    step_values,
                    [
                        float(steps[str(step)][mode][metric_name])
                        for step in step_values
                    ],
                    linestyle=style,
                    color=colors[model_name],
                    label=f"{model_name} {label}",
                )
            axis.set_title(title)
            axis.set_xlabel("rollout step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)

        cf_steps = counterfactual["steps"]
        if not isinstance(cf_steps, Mapping):
            raise ValueError("counterfactual steps record is malformed")
        for axis, metric_name, title, ylabel in (
            (
                axes[1, 0],
                "normalized_latent_rms",
                "Counterfactual latent divergence",
                "normalized RMS",
            ),
            (
                axes[1, 1],
                "decoded_pixel_mse",
                "Counterfactual decoded-frame divergence",
                "pixel MSE",
            ),
        ):
            means = np.asarray(
                [
                    float(cf_steps[str(step)][metric_name]["mean"])
                    for step in step_values
                ]
            )
            stds = np.asarray(
                [
                    float(cf_steps[str(step)][metric_name]["sample_std"])
                    for step in step_values
                ]
            )
            axis.plot(
                step_values,
                means,
                color=colors[model_name],
                label=model_name,
            )
            axis.fill_between(
                step_values,
                np.maximum(0.0, means - stds),
                means + stds,
                color=colors[model_name],
                alpha=0.2,
            )
            axis.set_title(title)
            axis.set_xlabel("rollout step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    for axis in axes.flat:
        axis.legend()
    figure.suptitle("Visual world-model rollout diagnostics")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def run_visual_rollout_diagnostics(
    *,
    data_path: Path | str,
    baseline_checkpoint_path: Path | str,
    aligned_checkpoint_path: Path | str,
    output_dir: Path | str,
    horizons: Iterable[int] = (1, 5, 10),
    windows_per_episode: int = 8,
    counterfactual_seeds: Iterable[int] = tuple(range(10)),
    encode_batch_size: int = 256,
    decode_batch_size: int = 256,
) -> dict[str, object]:
    """Compare two compatible visual checkpoints in one atomic bundle."""

    horizon_values, seed_values = _validate_diagnostic_protocol(
        horizons=horizons,
        windows_per_episode=windows_per_episode,
        counterfactual_seeds=counterfactual_seeds,
    )
    if encode_batch_size <= 0 or decode_batch_size <= 0:
        raise ValueError("encode and decode batch sizes must be positive")
    data = Path(data_path)
    baseline_path = Path(baseline_checkpoint_path)
    aligned_path = Path(aligned_checkpoint_path)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")
    dataset = load_visual_dataset(data)
    dataset_sha = sha256_file(data)
    baseline = load_visual_latent_checkpoint(baseline_path)
    aligned = load_visual_latent_checkpoint(aligned_path)
    _validate_visual_checkpoint_pair(
        baseline=baseline,
        aligned=aligned,
        dataset_sha256=dataset_sha,
    )
    test_ids = baseline.split_episode_ids["test"]
    missing_ids = sorted(
        set(int(value) for value in test_ids.tolist())
        - set(int(value) for value in dataset["episode_ids"].tolist())
    )
    if missing_ids:
        raise ValueError(
            "checkpoint test episode IDs are missing from the dataset: "
            + ", ".join(map(str, missing_ids))
        )
    latent_frames = encode_all_frames(
        baseline.autoencoder,
        dataset["frames"],
        batch_size=encode_batch_size,
    )
    selection = select_visual_rollout_windows(
        dataset=dataset,
        latent_frames=latent_frames,
        selected_episode_ids=test_ids,
        max_horizon=horizon_values[-1],
        windows_per_episode=int(windows_per_episode),
    )
    model_records: dict[str, dict[str, object]] = {}
    for name, model in (("baseline", baseline), ("aligned", aligned)):
        accuracy = evaluate_visual_rollout_model(
            model=model,
            windows=selection.windows,
            frames=dataset["frames"],
            decode_batch_size=decode_batch_size,
        )
        accuracy["counterfactual"] = evaluate_counterfactual_sensitivity(
            dynamics=model.dynamics,
            autoencoder=model.autoencoder,
            latent_normalizer=model.latent_normalizer,
            action_normalizer=model.action_normalizer,
            windows=selection.windows,
            seeds=seed_values,
            decode_batch_size=decode_batch_size,
        )
        model_records[name] = accuracy
    comparison = _build_model_comparison(
        baseline=model_records["baseline"],
        aligned=model_records["aligned"],
    )
    metrics: dict[str, object] = {
        "schema_version": 1,
        "models": model_records,
        "comparison": {"aligned_minus_baseline": comparison},
        "snapshots": {
            str(horizon): {
                "baseline": model_records["baseline"]["steps"][
                    str(horizon)
                ],
                "aligned": model_records["aligned"]["steps"][str(horizon)],
                "comparison": comparison[str(horizon)],
            }
            for horizon in horizon_values
        },
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "path": str(data.resolve()),
            "sha256": dataset_sha,
        },
        "checkpoints": {
            "baseline": {
                "path": str(baseline_path.resolve()),
                "sha256": sha256_file(baseline_path),
            },
            "aligned": {
                "path": str(aligned_path.resolve()),
                "sha256": sha256_file(aligned_path),
            },
        },
        "protocol": {
            "snapshot_horizons": list(horizon_values),
            "max_horizon": horizon_values[-1],
            "windows_per_episode": int(windows_per_episode),
            "counterfactual_seeds": list(seed_values),
            "aggregation": "windows-within-episode-then-episodes-equally",
            "counterfactual_action_replacement": (
                "complete-current-and-future-action-row"
            ),
        },
        "test_episode_ids": [
            int(value) for value in test_ids.tolist()
        ],
        "eligible_episode_ids": [
            int(value)
            for value in selection.eligible_episode_ids.tolist()
        ],
        "skipped_episode_ids": [
            int(value)
            for value in selection.skipped_episode_ids.tolist()
        ],
        "windows": [
            {
                "episode_id": window.episode_id,
                "start_step": window.start_step,
            }
            for window in selection.windows
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    try:
        _write_json(staging / "metrics.json", metrics)
        _write_json(staging / "manifest.json", manifest)
        plot_visual_rollout_comparison(
            metrics=metrics,
            output_path=staging / "visual_rollout_comparison.png",
        )
        if output.exists():
            output.rmdir()
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output_dir": str(output),
        "manifest": str(output / "manifest.json"),
        "metrics": str(output / "metrics.json"),
        "plot": str(output / "visual_rollout_comparison.png"),
        "horizons": list(horizon_values),
        "windows": len(selection.windows),
        "eligible_episodes": int(selection.eligible_episode_ids.size),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare compatible visual world models under recursive and "
            "counterfactual action rollouts."
        )
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--baseline-checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--aligned-checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 5, 10],
    )
    parser.add_argument(
        "--windows-per-episode",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--counterfactual-seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
    )
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    summary = run_visual_rollout_diagnostics(
        data_path=arguments.data,
        baseline_checkpoint_path=arguments.baseline_checkpoint,
        aligned_checkpoint_path=arguments.aligned_checkpoint,
        output_dir=arguments.output_dir,
        horizons=arguments.horizons,
        windows_per_episode=arguments.windows_per_episode,
        counterfactual_seeds=arguments.counterfactual_seeds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
