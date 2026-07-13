"""Dataset preparation for the learned one-step world model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Normalizer:
    """Per-feature affine normalization statistics."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean


def wrap_angle(values: np.ndarray | float) -> np.ndarray:
    """Map angles in radians to the half-open interval ``[-pi, pi)``."""

    angles = np.asarray(values, dtype=np.float64)
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


def split_episode_ids(
    episode_ids: np.ndarray,
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, np.ndarray]:
    """Split unique episode IDs without separating their transitions."""

    ids = np.asarray(episode_ids)
    if ids.ndim != 1:
        raise ValueError("episode_ids must be a one-dimensional array")
    unique_ids = np.unique(ids)
    if unique_ids.size < 3:
        raise ValueError("at least three episodes are required")
    if len(ratios) != 3 or any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("split ratios must contain three positive values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to one")

    shuffled = unique_ids.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    count = shuffled.size
    train_count = max(1, int(count * ratios[0]))
    validation_count = max(1, int(count * ratios[1]))
    test_count = count - train_count - validation_count
    if test_count < 1:
        train_count -= 1 - test_count
        test_count = 1

    validation_end = train_count + validation_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count:validation_end],
        "test": shuffled[validation_end : validation_end + test_count],
    }


def build_model_inputs(
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Encode raw states and actions as continuous model inputs."""

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("states must have shape [N, 4]")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [N, 2]")
    if actions.shape[0] != states.shape[0]:
        raise ValueError("states and actions must have equal lengths")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("state and action arrays must contain only finite values")

    return np.column_stack(
        (
            states[:, 0],
            states[:, 1],
            np.sin(states[:, 2]),
            np.cos(states[:, 2]),
            states[:, 3],
            actions[:, 0],
            actions[:, 1],
        )
    )


def build_model_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw transitions into continuous inputs and delta targets."""

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    inputs = build_model_inputs(states, actions)
    if next_states.shape != states.shape:
        raise ValueError("next_states must have the same [N, 4] shape as states")
    if not np.all(np.isfinite(next_states)):
        raise ValueError("transition arrays must contain only finite values")

    targets = next_states - states
    targets[:, 2] = wrap_angle(targets[:, 2])
    return inputs, targets


def fit_normalizer(values: np.ndarray) -> Normalizer:
    """Fit per-column population mean and standard deviation."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("normalizer values must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("normalizer values must contain only finite values")
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    if np.any(std == 0.0):
        raise ValueError("cannot normalize a zero variance dimension")
    return Normalizer(mean=mean, std=std)
