"""Deterministic episode-level bootstrap sampling for world-model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeBootstrap:
    drawn_episode_ids: np.ndarray
    episode_counts: dict[int, int]

    @property
    def draw_count(self) -> int:
        return int(self.drawn_episode_ids.size)

    @property
    def unique_count(self) -> int:
        return sum(count > 0 for count in self.episode_counts.values())


def _validate_episode_ids(
    values: np.ndarray,
    *,
    name: str,
    unique: bool,
) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if np.issubdtype(ids.dtype, np.bool_) or not np.issubdtype(
        ids.dtype,
        np.integer,
    ):
        raise ValueError(f"{name} must use a non-boolean integer dtype")
    if np.any(ids < 0):
        raise ValueError(f"{name} must contain only non-negative values")
    normalized = ids.astype(np.int64, copy=True)
    if unique and np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def sample_episode_bootstrap(
    episode_ids: np.ndarray,
    *,
    seed: int,
) -> EpisodeBootstrap:
    source = _validate_episode_ids(
        episode_ids,
        name="episode_ids",
        unique=True,
    )
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("bootstrap seed must be a non-negative integer")
    drawn = np.random.default_rng(int(seed)).choice(
        source,
        size=source.size,
        replace=True,
    ).astype(np.int64, copy=False)
    counts = {
        int(episode_id): int(np.count_nonzero(drawn == episode_id))
        for episode_id in np.sort(source)
    }
    return EpisodeBootstrap(drawn.copy(), counts)


def expand_episode_transition_indices(
    dataset_episode_ids: np.ndarray,
    drawn_episode_ids: np.ndarray,
) -> np.ndarray:
    dataset_ids = _validate_episode_ids(
        dataset_episode_ids,
        name="dataset_episode_ids",
        unique=False,
    )
    drawn_ids = _validate_episode_ids(
        drawn_episode_ids,
        name="drawn_episode_ids",
        unique=False,
    )
    groups = []
    for episode_id in drawn_ids.tolist():
        indices = np.flatnonzero(dataset_ids == episode_id)
        if indices.size == 0:
            raise ValueError(
                f"drawn episode {episode_id} is missing from the dataset"
            )
        groups.append(indices)
    return np.concatenate(groups).astype(np.int64, copy=False)
