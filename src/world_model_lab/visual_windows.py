"""Lazy, episode-isolated training windows for visual world models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator

import numpy as np

from .dataset import split_episode_ids
from .visual_dataset import CONTEXT_FRAMES, validate_visual_dataset


@dataclass(frozen=True)
class VisualWindowIndex:
    """Lightweight locations and episode eligibility for visual samples."""

    episode_indices: np.ndarray
    step_ids: np.ndarray
    selected_episode_ids: np.ndarray
    eligible_episode_ids: np.ndarray
    skipped_episode_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "episode_indices",
            "step_ids",
            "selected_episode_ids",
            "eligible_episode_ids",
            "skipped_episode_ids",
        ):
            values = np.asarray(getattr(self, name))
            if values.ndim != 1 or values.dtype != np.dtype(np.int64):
                raise ValueError(f"{name} must be a one-dimensional int64 array")
            owned = values.copy()
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
            arrays[name] = owned
        if arrays["episode_indices"].size != arrays["step_ids"].size:
            raise ValueError("episode_indices and step_ids must have equal lengths")
        if arrays["selected_episode_ids"].size == 0:
            raise ValueError("selected_episode_ids must be non-empty")
        for name in (
            "selected_episode_ids",
            "eligible_episode_ids",
            "skipped_episode_ids",
        ):
            if np.unique(arrays[name]).size != arrays[name].size:
                raise ValueError(f"{name} must not contain duplicates")
        eligible = set(int(value) for value in arrays["eligible_episode_ids"])
        skipped = set(int(value) for value in arrays["skipped_episode_ids"])
        selected = set(int(value) for value in arrays["selected_episode_ids"])
        if eligible & skipped or eligible | skipped != selected:
            raise ValueError("eligible and skipped IDs must partition selected IDs")
        if np.any(arrays["episode_indices"] < 0):
            raise ValueError("episode_indices must be non-negative")
        if np.any(arrays["step_ids"] < CONTEXT_FRAMES - 1):
            raise ValueError("step_ids do not have enough frame history")

    @property
    def count(self) -> int:
        return int(self.step_ids.size)


def _selected_ids(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(selected_episode_ids)
    if raw.ndim != 1:
        raise ValueError("selected_episode_ids must be one-dimensional")
    if raw.size == 0:
        raise ValueError("selected_episode_ids must be non-empty")
    if raw.dtype.kind not in "iu":
        raise ValueError("selected_episode_ids must be an integer array")
    limits = np.iinfo(np.int64)
    integer_values = [int(value) for value in raw.tolist()]
    if min(integer_values) < limits.min or max(integer_values) > limits.max:
        raise ValueError("selected_episode_ids values must fit in int64")
    selected = np.asarray(integer_values, dtype=np.int64)
    if np.unique(selected).size != selected.size:
        raise ValueError("selected_episode_ids must not contain duplicates")
    available = np.asarray(dataset["episode_ids"], dtype=np.int64)
    missing = selected[~np.isin(selected, available)]
    if missing.size:
        raise ValueError(
            f"episode {int(missing[0])} is missing from the visual dataset"
        )
    return selected


def _build_visual_window_index_validated(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowIndex:
    selected = _selected_ids(dataset, selected_episode_ids)
    all_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    transition_offsets = np.asarray(dataset["transition_offsets"], dtype=np.int64)
    positions = {int(episode_id): index for index, episode_id in enumerate(all_ids)}
    episode_indices: list[int] = []
    step_ids: list[int] = []
    eligible_ids: list[int] = []
    skipped_ids: list[int] = []
    first_step = CONTEXT_FRAMES - 1
    for selected_id in selected.tolist():
        episode_index = positions[int(selected_id)]
        transition_count = int(
            transition_offsets[episode_index + 1]
            - transition_offsets[episode_index]
        )
        if transition_count < CONTEXT_FRAMES:
            skipped_ids.append(int(selected_id))
            continue
        eligible_ids.append(int(selected_id))
        for step_id in range(first_step, transition_count):
            episode_indices.append(episode_index)
            step_ids.append(step_id)
    return VisualWindowIndex(
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        selected_episode_ids=selected,
        eligible_episode_ids=np.asarray(eligible_ids, dtype=np.int64),
        skipped_episode_ids=np.asarray(skipped_ids, dtype=np.int64),
    )


def build_visual_window_index(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowIndex:
    """Validate one artifact and index every legal sample in selected episodes."""

    validate_visual_dataset(dataset)
    return _build_visual_window_index_validated(dataset, selected_episode_ids)


VisualWindowSample = dict[str, np.ndarray | int]


class VisualWindowDataset:
    """Map-style lazy access to one-step visual dynamics samples."""

    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        index: VisualWindowIndex,
    ) -> None:
        episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
        transition_offsets = np.asarray(dataset["transition_offsets"], dtype=np.int64)
        if np.any(index.episode_indices >= episode_ids.size):
            raise ValueError("window index refers to an unavailable episode")
        for episode_index, step_id in zip(
            index.episode_indices.tolist(),
            index.step_ids.tolist(),
            strict=True,
        ):
            transition_count = int(
                transition_offsets[episode_index + 1]
                - transition_offsets[episode_index]
            )
            if step_id >= transition_count:
                raise ValueError("window index refers to an unavailable step")
        self.index = index
        self._frames = np.asarray(dataset["frames"])
        self._actions = np.asarray(dataset["actions"])
        self._episode_ids = episode_ids
        self._frame_offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
        self._transition_offsets = transition_offsets

    def __len__(self) -> int:
        return self.index.count

    def __getitem__(self, item: int) -> VisualWindowSample:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("visual window index must be an integer")
        try:
            position = operator.index(item)
        except TypeError:
            raise TypeError("visual window index must be an integer") from None
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("visual window index out of range")

        episode_index = int(self.index.episode_indices[position])
        step_id = int(self.index.step_ids[position])
        history_steps = CONTEXT_FRAMES - 1
        frame_start = (
            int(self._frame_offsets[episode_index])
            + step_id
            - history_steps
        )
        action_start = (
            int(self._transition_offsets[episode_index])
            + step_id
            - history_steps
        )
        current_action_index = action_start + history_steps
        return {
            "context_frames": self._frames[
                frame_start : frame_start + CONTEXT_FRAMES
            ].copy(),
            "history_actions": self._actions[
                action_start:current_action_index
            ].copy(),
            "current_action": self._actions[current_action_index].copy(),
            "target_frame": self._frames[
                frame_start + CONTEXT_FRAMES
            ].copy(),
            "episode_id": int(self._episode_ids[episode_index]),
            "step_id": step_id,
        }


def build_visual_window_dataset(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowDataset:
    """Validate and lazily expose samples from selected complete episodes."""

    validate_visual_dataset(dataset)
    index = _build_visual_window_index_validated(dataset, selected_episode_ids)
    return VisualWindowDataset(dataset, index)


def build_visual_window_splits(
    dataset: Mapping[str, np.ndarray],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, VisualWindowDataset]:
    """Split complete episodes, then lazily expose each split's windows."""

    validate_visual_dataset(dataset)
    split_ids = split_episode_ids(
        np.asarray(dataset["episode_ids"], dtype=np.int64),
        seed=seed,
        ratios=ratios,
    )
    return {
        name: VisualWindowDataset(
            dataset,
            _build_visual_window_index_validated(dataset, split_ids[name]),
        )
        for name in ("train", "validation", "test")
    }
