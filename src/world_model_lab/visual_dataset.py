"""Episode-oriented visual artifacts built from transition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


TransitionDataset = dict[str, np.ndarray]
VisualDataset = dict[str, np.ndarray]

REQUIRED_SOURCE_ARRAYS = (
    "states",
    "actions",
    "next_states",
    "rewards",
    "dones",
    "episode_ids",
    "step_ids",
    "terminal_reasons",
)
VALID_TERMINAL_REASONS = frozenset(
    {"goal", "collision", "out_of_bounds", "time_limit"}
)


@dataclass(frozen=True)
class OrderedEpisode:
    episode_id: int
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    step_ids: np.ndarray
    terminal_reasons: np.ndarray

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])


def load_transition_dataset(path: Path | str) -> TransitionDataset:
    """Load required source arrays without enabling pickle."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"transition dataset is not a regular file: {source_path}"
        )
    with np.load(source_path, allow_pickle=False) as loaded:
        missing = set(REQUIRED_SOURCE_ARRAYS) - set(loaded.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"transition dataset is missing arrays: {names}")
        source = {
            name: np.asarray(loaded[name])
            for name in REQUIRED_SOURCE_ARRAYS
        }
    reconstruct_episodes(source)
    return source


def _require_numeric_shape(
    source: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(source[name])
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if values.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric")
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def reconstruct_episodes(
    source: Mapping[str, np.ndarray],
) -> tuple[OrderedEpisode, ...]:
    """Validate and return episodes sorted by ID and step."""

    missing = set(REQUIRED_SOURCE_ARRAYS) - set(source)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"transition dataset is missing arrays: {names}")

    raw_states = np.asarray(source["states"])
    if raw_states.ndim != 2:
        raise ValueError("states must have shape [N, 4]")
    count = int(raw_states.shape[0])
    if count == 0:
        raise ValueError("transition dataset must contain at least one row")

    states = _require_numeric_shape(source, "states", (count, 4))
    actions = _require_numeric_shape(source, "actions", (count, 2))
    next_states = _require_numeric_shape(source, "next_states", (count, 4))
    rewards = _require_numeric_shape(source, "rewards", (count,))

    dones = np.asarray(source["dones"])
    if dones.shape != (count,) or dones.dtype.kind != "b":
        raise ValueError("dones must be a boolean array with shape [N]")

    episode_ids = np.asarray(source["episode_ids"])
    step_ids = np.asarray(source["step_ids"])
    for name, values in (("episode_ids", episode_ids), ("step_ids", step_ids)):
        if values.shape != (count,) or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer array with shape [N]")

    terminal_reasons = np.asarray(source["terminal_reasons"])
    if terminal_reasons.shape != (count,) or terminal_reasons.dtype.kind != "U":
        raise ValueError(
            "terminal_reasons must be a Unicode array with shape [N]"
        )

    episodes: list[OrderedEpisode] = []
    for raw_episode_id in np.unique(episode_ids):
        episode_id = int(raw_episode_id)
        row_indices = np.flatnonzero(episode_ids == raw_episode_id)
        order = np.argsort(step_ids[row_indices], kind="stable")
        indices = row_indices[order]
        ordered_steps = np.asarray(step_ids[indices], dtype=np.int64)
        expected_steps = np.arange(indices.size, dtype=np.int64)
        if not np.array_equal(ordered_steps, expected_steps):
            raise ValueError(
                f"episode {episode_id} step_ids must be unique, non-negative, "
                "and contiguous from zero"
            )

        episode_states = np.asarray(states[indices], dtype=np.float64)
        episode_next_states = np.asarray(next_states[indices], dtype=np.float64)
        if indices.size > 1 and not np.allclose(
            episode_states[1:],
            episode_next_states[:-1],
            rtol=0.0,
            atol=1e-10,
        ):
            matching = np.all(
                np.isclose(
                    episode_states[1:],
                    episode_next_states[:-1],
                    rtol=0.0,
                    atol=1e-10,
                ),
                axis=1,
            )
            failing_step = int(np.flatnonzero(~matching)[0])
            raise ValueError(
                f"episode {episode_id} is discontinuous after step {failing_step}"
            )

        episode_dones = np.asarray(dones[indices], dtype=np.bool_)
        episode_reasons = np.asarray(terminal_reasons[indices], dtype=np.str_)
        if np.any(episode_dones[:-1]):
            failing_step = int(np.flatnonzero(episode_dones[:-1])[0])
            raise ValueError(
                f"episode {episode_id} terminates before its final row "
                f"at step {failing_step}"
            )
        if not bool(episode_dones[-1]):
            raise ValueError(f"episode {episode_id} final row is not terminal")
        if np.any(episode_reasons[:-1] != ""):
            failing_step = int(np.flatnonzero(episode_reasons[:-1] != "")[0])
            raise ValueError(
                f"episode {episode_id} has terminal reason before final row "
                f"at step {failing_step}"
            )
        if str(episode_reasons[-1]) not in VALID_TERMINAL_REASONS:
            raise ValueError(
                f"episode {episode_id} final terminal reason is invalid: "
                f"{episode_reasons[-1]!s}"
            )

        episodes.append(
            OrderedEpisode(
                episode_id=episode_id,
                states=episode_states.copy(),
                actions=np.asarray(actions[indices], dtype=np.float64).copy(),
                next_states=episode_next_states.copy(),
                rewards=np.asarray(rewards[indices], dtype=np.float64).copy(),
                dones=episode_dones.copy(),
                step_ids=ordered_steps.copy(),
                terminal_reasons=episode_reasons.copy(),
            )
        )
    return tuple(episodes)
