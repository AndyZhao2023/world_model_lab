"""Train a convolutional autoencoder and one-step latent dynamics baseline."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._artifact_io import write_new_file_atomically
from .dataset import Normalizer, split_episode_ids
from .diagnose_model import sha256_file
from .visual_dataset import (
    CONTEXT_FRAMES,
    load_visual_dataset,
)
from .visual_latent_data import (
    LatentWindowArrays,
    VisualFrameDataset,
    build_latent_window_arrays,
    encode_all_frames,
    fit_safe_normalizer,
    frame_indices_for_episode_ids,
    frames_to_tensor,
    transition_indices_for_episode_ids,
)
from .visual_latent_model import (
    ConvAutoencoder,
    LatentDynamicsMLP,
)
from .visual_windows import build_visual_window_index


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_KIND = "visual_latent_world_model"


@dataclass
class PhaseTrainingResult:
    """Best model and complete train/validation histories for one phase."""

    model: nn.Module
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int


@dataclass
class LoadedVisualLatentModel:
    """Loaded visual latent models plus their reproducibility metadata."""

    autoencoder: ConvAutoencoder
    dynamics: LatentDynamicsMLP
    latent_normalizer: Normalizer
    action_normalizer: Normalizer
    split_episode_ids: dict[str, np.ndarray]
    training_config: dict[str, Any]
    dataset_metadata: dict[str, Any]
    autoencoder_history: dict[str, Any]
    dynamics_history: dict[str, Any]
    autoencoder_test_metrics: dict[str, float | int]
    dynamics_test_metrics: dict[str, float | int]


def _validate_training_parameters(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _mean_autoencoder_loss(
    model: ConvAutoencoder,
    loader: DataLoader,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for images in loader:
            reconstructions = model(images)
            if not torch.all(torch.isfinite(reconstructions)):
                raise ValueError("autoencoder produced non-finite pixels")
            batch_loss = torch.mean(torch.square(reconstructions - images))
            batch_count = int(images.shape[0])
            loss_sum += float(batch_loss) * batch_count
            sample_count += batch_count
    if sample_count == 0:
        raise ValueError("autoencoder evaluation data must not be empty")
    return loss_sum / sample_count


def train_autoencoder(
    dataset: dict[str, np.ndarray],
    *,
    split_episode_ids: dict[str, np.ndarray],
    latent_dim: int = 32,
    base_channels: int = 16,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> PhaseTrainingResult:
    """Train the frame autoencoder and restore its best validation weights."""

    _validate_training_parameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    train_data = VisualFrameDataset(dataset, split_episode_ids["train"])
    validation_data = VisualFrameDataset(
        dataset,
        split_episode_ids["validation"],
    )
    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("autoencoder train and validation data must not be empty")

    torch.manual_seed(seed)
    model = ConvAutoencoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
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
    best_validation_loss = math.inf
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch_index in range(epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for images in train_loader:
            optimizer.zero_grad()
            reconstructions = model(images)
            loss = torch.mean(torch.square(reconstructions - images))
            if not torch.isfinite(loss):
                raise ValueError("autoencoder training loss is non-finite")
            loss.backward()
            optimizer.step()
            batch_count = int(images.shape[0])
            epoch_loss += float(loss.detach()) * batch_count
            sample_count += batch_count
        train_loss = epoch_loss / sample_count
        validation_loss = _mean_autoencoder_loss(model, validation_loader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch_index + 1
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is None:
        raise RuntimeError("autoencoder training did not produce a checkpoint")
    model.load_state_dict(best_state_dict)
    model.eval()
    return PhaseTrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
    )


def evaluate_autoencoder(
    model: ConvAutoencoder,
    dataset: dict[str, np.ndarray],
    *,
    selected_episode_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float | int]:
    """Measure held-out frame reconstruction in normalized pixel space."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frame_data = VisualFrameDataset(dataset, selected_episode_ids)
    if len(frame_data) == 0:
        raise ValueError("autoencoder evaluation data must not be empty")
    loader = DataLoader(
        frame_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    with torch.no_grad():
        for images in loader:
            reconstructions = model(images)
            if not torch.all(torch.isfinite(reconstructions)):
                raise ValueError("autoencoder produced non-finite pixels")
            errors = reconstructions - images
            squared_error_sum += float(torch.sum(torch.square(errors)))
            absolute_error_sum += float(torch.sum(torch.abs(errors)))
            value_count += images.numel()
    pixel_mse = squared_error_sum / value_count
    pixel_mae = absolute_error_sum / value_count
    return {
        "frames": len(frame_data),
        "pixel_mse": pixel_mse,
        "pixel_mae": pixel_mae,
        "psnr_db": 10.0 * math.log10(1.0 / max(pixel_mse, 1e-12)),
    }


def _normalized_window_tensors(
    arrays: LatentWindowArrays,
    *,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context = latent_normalizer.normalize(arrays.context_latents)
    target = latent_normalizer.normalize(arrays.target_latents)
    history = action_normalizer.normalize(arrays.history_actions)
    current = action_normalizer.normalize(arrays.current_actions)
    values = (context, history, current, target)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("normalized latent window arrays must be finite")
    return tuple(
        torch.as_tensor(value, dtype=torch.float32)
        for value in values
    )


def _mean_dynamics_loss(
    model: LatentDynamicsMLP,
    loader: DataLoader,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for context, history, current, target in loader:
            prediction = model(context, history, current)
            loss = torch.mean(torch.square(prediction - target))
            if not torch.isfinite(loss):
                raise ValueError("latent dynamics validation loss is non-finite")
            batch_count = int(context.shape[0])
            loss_sum += float(loss) * batch_count
            sample_count += batch_count
    if sample_count == 0:
        raise ValueError("latent dynamics evaluation data must not be empty")
    return loss_sum / sample_count


def train_latent_dynamics(
    train_arrays: LatentWindowArrays,
    validation_arrays: LatentWindowArrays,
    *,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
    hidden_size: int = 256,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> PhaseTrainingResult:
    """Train residual one-step dynamics in normalized latent space."""

    _validate_training_parameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    if train_arrays.count == 0 or validation_arrays.count == 0:
        raise ValueError("latent dynamics train and validation data must not be empty")
    train_tensors = _normalized_window_tensors(
        train_arrays,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    validation_tensors = _normalized_window_tensors(
        validation_arrays,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    torch.manual_seed(seed)
    model = LatentDynamicsMLP(
        latent_dim=int(train_arrays.context_latents.shape[2]),
        hidden_size=hidden_size,
        context_frames=CONTEXT_FRAMES,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(*validation_tensors),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch_index in range(epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for context, history, current, target in train_loader:
            optimizer.zero_grad()
            prediction = model(context, history, current)
            loss = torch.mean(torch.square(prediction - target))
            if not torch.isfinite(loss):
                raise ValueError("latent dynamics training loss is non-finite")
            loss.backward()
            optimizer.step()
            batch_count = int(context.shape[0])
            epoch_loss += float(loss.detach()) * batch_count
            sample_count += batch_count
        train_loss = epoch_loss / sample_count
        validation_loss = _mean_dynamics_loss(model, validation_loader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch_index + 1
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is None:
        raise RuntimeError("latent dynamics training did not produce a checkpoint")
    model.load_state_dict(best_state_dict)
    model.eval()
    return PhaseTrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
    )


def _predict_normalized_latents(
    model: LatentDynamicsMLP,
    arrays: LatentWindowArrays,
    *,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = torch.as_tensor(
        latent_normalizer.normalize(arrays.context_latents[start:stop]),
        dtype=torch.float32,
    )
    history = torch.as_tensor(
        action_normalizer.normalize(arrays.history_actions[start:stop]),
        dtype=torch.float32,
    )
    current = torch.as_tensor(
        action_normalizer.normalize(arrays.current_actions[start:stop]),
        dtype=torch.float32,
    )
    target = torch.as_tensor(
        latent_normalizer.normalize(arrays.target_latents[start:stop]),
        dtype=torch.float32,
    )
    return model(context, history, current), target


def _decode_normalized_latents(
    autoencoder: ConvAutoencoder,
    normalized_latents: torch.Tensor,
    latent_normalizer: Normalizer,
) -> torch.Tensor:
    raw_latents = latent_normalizer.denormalize(
        normalized_latents.detach().cpu().numpy()
    )
    latent_tensor = torch.as_tensor(raw_latents, dtype=torch.float32)
    return autoencoder.decode(latent_tensor).clamp(0.0, 1.0)


def evaluate_latent_dynamics(
    model: LatentDynamicsMLP,
    autoencoder: ConvAutoencoder,
    dataset: dict[str, np.ndarray],
    arrays: LatentWindowArrays,
    *,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
    batch_size: int,
) -> dict[str, float | int]:
    """Measure held-out latent and decoded one-step prediction errors."""

    if arrays.count == 0:
        raise ValueError("latent dynamics evaluation data must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frames = np.asarray(dataset["frames"])
    model.eval()
    autoencoder.eval()
    latent_squared_error_sum = 0.0
    latent_value_count = 0
    pixel_squared_error_sum = 0.0
    pixel_absolute_error_sum = 0.0
    oracle_squared_error_sum = 0.0
    oracle_absolute_error_sum = 0.0
    copy_squared_error_sum = 0.0
    copy_absolute_error_sum = 0.0
    pixel_value_count = 0
    changed_absolute_error_sum = 0.0
    oracle_changed_absolute_error_sum = 0.0
    copy_changed_absolute_error_sum = 0.0
    changed_pixel_count = 0

    with torch.no_grad():
        for start in range(0, arrays.count, batch_size):
            stop = min(start + batch_size, arrays.count)
            predicted_normalized, target_normalized = (
                _predict_normalized_latents(
                    model,
                    arrays,
                    latent_normalizer=latent_normalizer,
                    action_normalizer=action_normalizer,
                    start=start,
                    stop=stop,
                )
            )
            if not torch.all(torch.isfinite(predicted_normalized)):
                raise ValueError("latent dynamics produced non-finite latents")
            latent_error = predicted_normalized - target_normalized
            latent_squared_error_sum += float(
                torch.sum(torch.square(latent_error))
            )
            latent_value_count += latent_error.numel()

            predictions = _decode_normalized_latents(
                autoencoder,
                predicted_normalized,
                latent_normalizer,
            )
            oracle_reconstructions = _decode_normalized_latents(
                autoencoder,
                target_normalized,
                latent_normalizer,
            )
            target_frames = frames_to_tensor(
                frames[arrays.target_frame_indices[start:stop]]
            )
            last_frames = frames_to_tensor(
                frames[arrays.last_frame_indices[start:stop]]
            )
            errors = predictions - target_frames
            oracle_errors = oracle_reconstructions - target_frames
            copy_errors = last_frames - target_frames
            pixel_squared_error_sum += float(torch.sum(torch.square(errors)))
            pixel_absolute_error_sum += float(torch.sum(torch.abs(errors)))
            oracle_squared_error_sum += float(
                torch.sum(torch.square(oracle_errors))
            )
            oracle_absolute_error_sum += float(
                torch.sum(torch.abs(oracle_errors))
            )
            copy_squared_error_sum += float(
                torch.sum(torch.square(copy_errors))
            )
            copy_absolute_error_sum += float(torch.sum(torch.abs(copy_errors)))
            pixel_value_count += target_frames.numel()

            changed = torch.any(
                target_frames != last_frames,
                dim=1,
                keepdim=True,
            )
            changed_values = changed.expand_as(target_frames)
            batch_changed_pixels = int(torch.sum(changed))
            if batch_changed_pixels:
                changed_absolute_error_sum += float(
                    torch.sum(torch.abs(errors)[changed_values])
                )
                oracle_changed_absolute_error_sum += float(
                    torch.sum(torch.abs(oracle_errors)[changed_values])
                )
                copy_changed_absolute_error_sum += float(
                    torch.sum(torch.abs(copy_errors)[changed_values])
                )
                changed_pixel_count += batch_changed_pixels

    pixel_mse = pixel_squared_error_sum / pixel_value_count
    pixel_mae = pixel_absolute_error_sum / pixel_value_count
    changed_value_count = changed_pixel_count * 3
    changed_pixel_mae = (
        changed_absolute_error_sum / changed_value_count
        if changed_value_count
        else 0.0
    )
    oracle_changed_pixel_mae = (
        oracle_changed_absolute_error_sum / changed_value_count
        if changed_value_count
        else 0.0
    )
    copy_changed_pixel_mae = (
        copy_changed_absolute_error_sum / changed_value_count
        if changed_value_count
        else 0.0
    )
    metrics: dict[str, float | int] = {
        "windows": arrays.count,
        "normalized_latent_mse": (
            latent_squared_error_sum / latent_value_count
        ),
        "pixel_mse": pixel_mse,
        "pixel_mae": pixel_mae,
        "psnr_db": 10.0 * math.log10(1.0 / max(pixel_mse, 1e-12)),
        "changed_pixel_mae": changed_pixel_mae,
        "changed_pixel_count": changed_pixel_count,
        "oracle_reconstruction_pixel_mse": (
            oracle_squared_error_sum / pixel_value_count
        ),
        "oracle_reconstruction_pixel_mae": (
            oracle_absolute_error_sum / pixel_value_count
        ),
        "oracle_reconstruction_changed_pixel_mae": (
            oracle_changed_pixel_mae
        ),
        "copy_last_pixel_mse": copy_squared_error_sum / pixel_value_count,
        "copy_last_pixel_mae": copy_absolute_error_sum / pixel_value_count,
        "copy_last_changed_pixel_mae": copy_changed_pixel_mae,
    }
    if not all(
        math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("latent dynamics metrics must be finite")
    return metrics


def _history_payload(result: PhaseTrainingResult) -> dict[str, Any]:
    return {
        "train_losses": [float(value) for value in result.train_losses],
        "validation_losses": [
            float(value) for value in result.validation_losses
        ],
        "best_epoch": int(result.best_epoch),
    }


def _metric_payload(
    metrics: dict[str, float | int],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            result[name] = int(value)
        else:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"metric {name} must be finite")
            result[name] = numeric
    return result


def save_visual_latent_checkpoint(
    path: Path | str,
    *,
    autoencoder_result: PhaseTrainingResult,
    dynamics_result: PhaseTrainingResult,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
    split_episode_ids: dict[str, np.ndarray],
    training_config: dict[str, Any],
    dataset_metadata: dict[str, Any],
    autoencoder_test_metrics: dict[str, float | int],
    dynamics_test_metrics: dict[str, float | int],
) -> Path:
    """Atomically publish both visual models and all inference metadata."""

    autoencoder = autoencoder_result.model
    dynamics = dynamics_result.model
    if not isinstance(autoencoder, ConvAutoencoder):
        raise ValueError("autoencoder_result must contain ConvAutoencoder")
    if not isinstance(dynamics, LatentDynamicsMLP):
        raise ValueError("dynamics_result must contain LatentDynamicsMLP")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": CHECKPOINT_KIND,
        "dataset": dict(dataset_metadata),
        "model_config": {
            "latent_dim": autoencoder.latent_dim,
            "base_channels": autoencoder.base_channels,
            "dynamics_hidden_size": dynamics.hidden_size,
            "context_frames": dynamics.context_frames,
        },
        "autoencoder_state_dict": autoencoder.state_dict(),
        "dynamics_state_dict": dynamics.state_dict(),
        "latent_mean": torch.as_tensor(
            latent_normalizer.mean,
            dtype=torch.float32,
        ),
        "latent_std": torch.as_tensor(
            latent_normalizer.std,
            dtype=torch.float32,
        ),
        "action_mean": torch.as_tensor(
            action_normalizer.mean,
            dtype=torch.float32,
        ),
        "action_std": torch.as_tensor(
            action_normalizer.std,
            dtype=torch.float32,
        ),
        "split_episode_ids": {
            name: torch.as_tensor(ids, dtype=torch.int64)
            for name, ids in split_episode_ids.items()
        },
        "training_config": dict(training_config),
        "autoencoder_history": _history_payload(autoencoder_result),
        "dynamics_history": _history_payload(dynamics_result),
        "autoencoder_test_metrics": _metric_payload(
            autoencoder_test_metrics
        ),
        "dynamics_test_metrics": _metric_payload(dynamics_test_metrics),
    }
    return write_new_file_atomically(
        output,
        writer=lambda handle: torch.save(payload, handle),
        exists_message=f"checkpoint already exists: {output}",
    )


def _loaded_normalizer(
    payload: dict[str, Any],
    *,
    prefix: str,
    size: int,
) -> Normalizer:
    mean_tensor = payload.get(f"{prefix}_mean")
    std_tensor = payload.get(f"{prefix}_std")
    if not isinstance(mean_tensor, torch.Tensor) or not isinstance(
        std_tensor,
        torch.Tensor,
    ):
        raise ValueError(f"{prefix}_normalizer tensors are missing")
    mean = mean_tensor.cpu().numpy().astype(np.float64)
    std = std_tensor.cpu().numpy().astype(np.float64)
    if (
        mean.shape != (size,)
        or std.shape != (size,)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0.0)
    ):
        raise ValueError(
            f"{prefix}_normalizer must have finite shape [{size}] "
            "and positive std"
        )
    return Normalizer(mean=mean, std=std)


def _loaded_history(
    payload: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    history = payload.get(name)
    if not isinstance(history, dict):
        raise ValueError(f"{name} is missing")
    train = [float(value) for value in history.get("train_losses", [])]
    validation = [
        float(value) for value in history.get("validation_losses", [])
    ]
    best_epoch = int(history.get("best_epoch", 0))
    if (
        not train
        or len(train) != len(validation)
        or not all(math.isfinite(value) for value in train + validation)
        or best_epoch < 1
        or best_epoch > len(train)
    ):
        raise ValueError(f"{name} is invalid")
    return {
        "train_losses": train,
        "validation_losses": validation,
        "best_epoch": best_epoch,
    }


def _loaded_metrics(
    payload: dict[str, Any],
    name: str,
) -> dict[str, float | int]:
    raw = payload.get(name)
    if not isinstance(raw, dict):
        raise ValueError(f"{name} is missing")
    result: dict[str, float | int] = {}
    for key, value in raw.items():
        if isinstance(value, int):
            result[str(key)] = int(value)
        else:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} contains non-finite values")
            result[str(key)] = numeric
    return result


def load_visual_latent_checkpoint(
    path: Path | str,
) -> LoadedVisualLatentModel:
    """Load a visual latent checkpoint without enabling arbitrary pickle."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported visual latent checkpoint format")
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError("unsupported visual latent checkpoint kind")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("model_config is missing")
    latent_dim = int(model_config.get("latent_dim", 0))
    base_channels = int(model_config.get("base_channels", 0))
    hidden_size = int(model_config.get("dynamics_hidden_size", 0))
    context_frames = int(model_config.get("context_frames", 0))
    autoencoder = ConvAutoencoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
    )
    dynamics = LatentDynamicsMLP(
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        context_frames=context_frames,
    )
    try:
        autoencoder.load_state_dict(payload["autoencoder_state_dict"])
        dynamics.load_state_dict(payload["dynamics_state_dict"])
    except (KeyError, RuntimeError) as error:
        raise ValueError("checkpoint model weights are invalid") from error
    autoencoder.eval()
    dynamics.eval()
    latent_normalizer = _loaded_normalizer(
        payload,
        prefix="latent",
        size=latent_dim,
    )
    action_normalizer = _loaded_normalizer(
        payload,
        prefix="action",
        size=2,
    )
    raw_splits = payload.get("split_episode_ids")
    if not isinstance(raw_splits, dict) or set(raw_splits) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("split_episode_ids must contain train/validation/test")
    splits: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for name in ("train", "validation", "test"):
        values = raw_splits[name]
        if not isinstance(values, torch.Tensor):
            raise ValueError("split_episode_ids must contain tensors")
        ids = values.cpu().numpy().astype(np.int64)
        if ids.ndim != 1 or ids.size == 0 or np.unique(ids).size != ids.size:
            raise ValueError(f"{name} split episode IDs are invalid")
        overlap = seen & set(int(value) for value in ids.tolist())
        if overlap:
            raise ValueError("split episode IDs must be disjoint")
        seen.update(int(value) for value in ids.tolist())
        splits[name] = ids

    dataset_metadata = payload.get("dataset")
    training_config = payload.get("training_config")
    if not isinstance(dataset_metadata, dict):
        raise ValueError("dataset metadata is missing")
    if not isinstance(training_config, dict):
        raise ValueError("training_config is missing")
    return LoadedVisualLatentModel(
        autoencoder=autoencoder,
        dynamics=dynamics,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
        split_episode_ids=splits,
        training_config=dict(training_config),
        dataset_metadata=dict(dataset_metadata),
        autoencoder_history=_loaded_history(
            payload,
            "autoencoder_history",
        ),
        dynamics_history=_loaded_history(payload, "dynamics_history"),
        autoencoder_test_metrics=_loaded_metrics(
            payload,
            "autoencoder_test_metrics",
        ),
        dynamics_test_metrics=_loaded_metrics(
            payload,
            "dynamics_test_metrics",
        ),
    )


def plot_visual_latent_predictions(
    output_path: Path | str,
    *,
    autoencoder: ConvAutoencoder,
    dynamics: LatentDynamicsMLP,
    dataset: dict[str, np.ndarray],
    arrays: LatentWindowArrays,
    latent_normalizer: Normalizer,
    action_normalizer: Normalizer,
    count: int = 6,
) -> Path:
    """Write a deterministic next-frame comparison grid."""

    if count <= 0:
        raise ValueError("preview count must be positive")
    row_count = min(count, arrays.count)
    if row_count == 0:
        raise ValueError("preview requires at least one latent window")
    frames = np.asarray(dataset["frames"])
    dynamics.eval()
    autoencoder.eval()
    with torch.no_grad():
        predicted_normalized, target_normalized = _predict_normalized_latents(
            dynamics,
            arrays,
            latent_normalizer=latent_normalizer,
            action_normalizer=action_normalizer,
            start=0,
            stop=row_count,
        )
        predicted = (
            _decode_normalized_latents(
                autoencoder,
                predicted_normalized,
                latent_normalizer,
            )
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        oracle = (
            _decode_normalized_latents(
                autoencoder,
                target_normalized,
                latent_normalizer,
            )
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
    last = frames[arrays.last_frame_indices[:row_count]].astype(np.float32) / 255.0
    target = (
        frames[arrays.target_frame_indices[:row_count]].astype(np.float32)
        / 255.0
    )
    oracle_errors = np.abs(oracle - target)
    errors = np.abs(predicted - target)

    figure, axes = plt.subplots(
        row_count,
        6,
        figsize=(15, 2.4 * row_count),
        squeeze=False,
    )
    titles = (
        "Last context",
        "Target",
        "Oracle reconstruction",
        "Oracle error",
        "Predicted",
        "Prediction error",
    )
    columns = (last, target, oracle, oracle_errors, predicted, errors)
    for row in range(row_count):
        for column, values in enumerate(columns):
            axis = axes[row, column]
            axis.imshow(np.clip(values[row], 0.0, 1.0))
            axis.axis("off")
            if row == 0:
                axis.set_title(titles[column])
    figure.tight_layout()
    payload = io.BytesIO()
    figure.savefig(payload, format="png", dpi=160)
    plt.close(figure)
    png_bytes = payload.getvalue()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return write_new_file_atomically(
        output,
        writer=lambda handle: handle.write(png_bytes),
        exists_message=f"preview already exists: {output}",
    )


def _preflight_output_paths(
    output_path: Path,
    preview_path: Path,
) -> None:
    if output_path.resolve(strict=False) == preview_path.resolve(strict=False):
        raise ValueError("checkpoint and preview paths must be different")
    if output_path.exists():
        raise FileExistsError(f"checkpoint already exists: {output_path}")
    if preview_path.exists():
        raise FileExistsError(f"preview already exists: {preview_path}")


def run_visual_latent_training(
    *,
    data_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
    latent_dim: int = 32,
    base_channels: int = 16,
    dynamics_hidden_size: int = 256,
    autoencoder_epochs: int = 20,
    dynamics_epochs: int = 50,
    autoencoder_batch_size: int = 128,
    dynamics_batch_size: int = 256,
    autoencoder_learning_rate: float = 1e-3,
    dynamics_learning_rate: float = 1e-3,
    seed: int = 0,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Run both visual training phases and publish checkpoint plus preview."""

    output = Path(output_path)
    preview = Path(preview_path)
    _preflight_output_paths(output, preview)
    if latent_dim <= 0 or base_channels <= 0 or dynamics_hidden_size <= 0:
        raise ValueError("model dimensions must be positive")
    if split_seed < 0:
        raise ValueError("split_seed must be non-negative")
    data = Path(data_path)
    dataset = load_visual_dataset(data)
    splits = split_episode_ids(dataset["episode_ids"], seed=split_seed)

    autoencoder_result = train_autoencoder(
        dataset,
        split_episode_ids=splits,
        latent_dim=latent_dim,
        base_channels=base_channels,
        epochs=autoencoder_epochs,
        batch_size=autoencoder_batch_size,
        learning_rate=autoencoder_learning_rate,
        seed=seed,
    )
    autoencoder = autoencoder_result.model
    assert isinstance(autoencoder, ConvAutoencoder)
    autoencoder_test_metrics = evaluate_autoencoder(
        autoencoder,
        dataset,
        selected_episode_ids=splits["test"],
        batch_size=autoencoder_batch_size,
    )
    latent_frames = encode_all_frames(
        autoencoder,
        dataset["frames"],
        batch_size=autoencoder_batch_size,
    )
    train_frame_indices = frame_indices_for_episode_ids(
        dataset,
        splits["train"],
    )
    train_transition_indices = transition_indices_for_episode_ids(
        dataset,
        splits["train"],
    )
    latent_normalizer = fit_safe_normalizer(
        latent_frames[train_frame_indices]
    )
    action_normalizer = fit_safe_normalizer(
        dataset["actions"][train_transition_indices]
    )
    window_indexes = {
        name: build_visual_window_index(dataset, ids)
        for name, ids in splits.items()
    }
    window_arrays = {
        name: build_latent_window_arrays(
            dataset,
            window_indexes[name],
            latent_frames,
        )
        for name in ("train", "validation", "test")
    }
    for name, arrays in window_arrays.items():
        if arrays.count == 0:
            raise ValueError(f"{name} split has no eligible visual windows")

    dynamics_result = train_latent_dynamics(
        window_arrays["train"],
        window_arrays["validation"],
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
        hidden_size=dynamics_hidden_size,
        epochs=dynamics_epochs,
        batch_size=dynamics_batch_size,
        learning_rate=dynamics_learning_rate,
        seed=seed,
    )
    dynamics = dynamics_result.model
    assert isinstance(dynamics, LatentDynamicsMLP)
    dynamics_test_metrics = evaluate_latent_dynamics(
        dynamics,
        autoencoder,
        dataset,
        window_arrays["test"],
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
        batch_size=dynamics_batch_size,
    )
    training_config = {
        "data_path": str(data),
        "device": "cpu",
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "dynamics_hidden_size": dynamics_hidden_size,
        "autoencoder_epochs": autoencoder_epochs,
        "dynamics_epochs": dynamics_epochs,
        "autoencoder_batch_size": autoencoder_batch_size,
        "dynamics_batch_size": dynamics_batch_size,
        "autoencoder_learning_rate": autoencoder_learning_rate,
        "dynamics_learning_rate": dynamics_learning_rate,
        "seed": seed,
        "split_seed": split_seed,
    }
    dataset_metadata = {
        "path": str(data.resolve()),
        "sha256": sha256_file(data),
        "schema_version": int(dataset["schema_version"].item()),
        "renderer_version": str(dataset["renderer_version"].item()),
    }
    save_visual_latent_checkpoint(
        output,
        autoencoder_result=autoencoder_result,
        dynamics_result=dynamics_result,
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
        split_episode_ids=splits,
        training_config=training_config,
        dataset_metadata=dataset_metadata,
        autoencoder_test_metrics=autoencoder_test_metrics,
        dynamics_test_metrics=dynamics_test_metrics,
    )
    plot_visual_latent_predictions(
        preview,
        autoencoder=autoencoder,
        dynamics=dynamics,
        dataset=dataset,
        arrays=window_arrays["test"],
        latent_normalizer=latent_normalizer,
        action_normalizer=action_normalizer,
    )
    return {
        "dataset": dataset_metadata,
        "split_episodes": {
            name: int(ids.size) for name, ids in splits.items()
        },
        "split_frames": {
            name: int(
                frame_indices_for_episode_ids(dataset, ids).size
            )
            for name, ids in splits.items()
        },
        "split_windows": {
            name: arrays.count for name, arrays in window_arrays.items()
        },
        "autoencoder": {
            "initial_train_loss": autoencoder_result.train_losses[0],
            "final_train_loss": autoencoder_result.train_losses[-1],
            "best_epoch": autoencoder_result.best_epoch,
            "best_validation_loss": min(
                autoencoder_result.validation_losses
            ),
            "test": autoencoder_test_metrics,
        },
        "dynamics": {
            "initial_train_loss": dynamics_result.train_losses[0],
            "final_train_loss": dynamics_result.train_losses[-1],
            "best_epoch": dynamics_result.best_epoch,
            "best_validation_loss": min(dynamics_result.validation_losses),
            "test": dynamics_test_metrics,
        },
        "checkpoint": str(output),
        "preview": str(preview),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/visual_episodes.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/visual_latent_world_model.pt"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("artifacts/visual_latent_predictions.png"),
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dynamics-hidden-size", type=int, default=256)
    parser.add_argument("--autoencoder-epochs", type=int, default=20)
    parser.add_argument("--dynamics-epochs", type=int, default=50)
    parser.add_argument("--autoencoder-batch-size", type=int, default=128)
    parser.add_argument("--dynamics-batch-size", type=int, default=256)
    parser.add_argument(
        "--autoencoder-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--dynamics-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary = run_visual_latent_training(
            data_path=args.data,
            output_path=args.output,
            preview_path=args.preview,
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            dynamics_hidden_size=args.dynamics_hidden_size,
            autoencoder_epochs=args.autoencoder_epochs,
            dynamics_epochs=args.dynamics_epochs,
            autoencoder_batch_size=args.autoencoder_batch_size,
            dynamics_batch_size=args.dynamics_batch_size,
            autoencoder_learning_rate=args.autoencoder_learning_rate,
            dynamics_learning_rate=args.dynamics_learning_rate,
            seed=args.seed,
            split_seed=args.split_seed,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
