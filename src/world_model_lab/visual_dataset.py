"""Episode-oriented visual artifacts built from transition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .car_env import CarEnv
from .visual_observation import (
    IMAGE_SIZE,
    PILLOW_VERSION,
    RENDERER_VERSION,
    render_observation,
    scene_from_env,
)


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
VISUAL_SCHEMA_VERSION = 1
CONTEXT_FRAMES = 4
TERMINAL_REASON_DTYPE = np.dtype("<U13")
REQUIRED_VISUAL_ARRAYS = (
    "schema_version",
    "image_size",
    "context_frames",
    "renderer_version",
    "pillow_version",
    "frames",
    "states",
    "actions",
    "rewards",
    "dones",
    "terminal_reasons",
    "episode_ids",
    "frame_offsets",
    "transition_offsets",
    "scene_world_bounds",
    "scene_obstacle",
    "scene_obstacle_radius",
    "scene_goal",
    "scene_goal_radius",
    "scene_car_radius",
    "scene_dt",
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


def _scalar(
    dataset: Mapping[str, np.ndarray],
    name: str,
) -> object:
    values = np.asarray(dataset[name])
    if values.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return values.item()


def _require_dtype(
    dataset: Mapping[str, np.ndarray],
    name: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    values = np.asarray(dataset[name])
    expected = np.dtype(dtype)
    if values.dtype != expected:
        raise ValueError(f"{name} must have dtype {expected.name}")
    return values


def validate_visual_dataset(dataset: Mapping[str, np.ndarray]) -> None:
    """Reject unsupported, malformed, or misaligned visual artifacts."""

    missing = set(REQUIRED_VISUAL_ARRAYS) - set(dataset)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"visual dataset is missing arrays: {names}")

    schema_values = _require_dtype(dataset, "schema_version", np.int64)
    image_size_values = _require_dtype(dataset, "image_size", np.int64)
    context_values = _require_dtype(dataset, "context_frames", np.int64)
    schema_version = int(_scalar(dataset, "schema_version"))
    image_size = int(_scalar(dataset, "image_size"))
    context_frames = int(_scalar(dataset, "context_frames"))
    if schema_values.shape != () or schema_version != VISUAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    if image_size_values.shape != () or image_size != IMAGE_SIZE:
        raise ValueError(f"schema version 1 requires image_size={IMAGE_SIZE}")
    if context_values.shape != () or context_frames != CONTEXT_FRAMES:
        raise ValueError(
            f"schema version 1 requires context_frames={CONTEXT_FRAMES}"
        )

    for name in ("renderer_version", "pillow_version"):
        values = np.asarray(dataset[name])
        if values.shape != () or values.dtype.kind != "U":
            raise ValueError(f"{name} must be a Unicode scalar")
        if not str(values.item()).strip():
            raise ValueError(f"{name} must be a non-empty string")
    renderer_version = str(np.asarray(dataset["renderer_version"]).item())
    if renderer_version != RENDERER_VERSION:
        raise ValueError(
            "schema version 1 requires "
            f"renderer_version={RENDERER_VERSION}"
        )

    frames = _require_dtype(dataset, "frames", np.uint8)
    states = _require_dtype(dataset, "states", np.float64)
    actions = _require_dtype(dataset, "actions", np.float64)
    rewards = _require_dtype(dataset, "rewards", np.float64)
    dones = _require_dtype(dataset, "dones", np.bool_)
    episode_ids = _require_dtype(dataset, "episode_ids", np.int64)
    frame_offsets = _require_dtype(dataset, "frame_offsets", np.int64)
    transition_offsets = _require_dtype(
        dataset,
        "transition_offsets",
        np.int64,
    )
    terminal_reasons = np.asarray(dataset["terminal_reasons"])
    if terminal_reasons.dtype.kind != "U":
        raise ValueError("terminal_reasons must have a Unicode dtype")

    episode_count = int(episode_ids.size)
    transition_count = int(actions.shape[0]) if actions.ndim == 2 else -1
    frame_count = int(frames.shape[0]) if frames.ndim == 4 else -1
    if frames.shape != (frame_count, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("frames must have shape [F, 64, 64, 3]")
    if states.shape != (frame_count, 4):
        raise ValueError("states must have shape [F, 4]")
    if actions.shape != (transition_count, 2):
        raise ValueError("actions must have shape [N, 2]")
    for name, values in (
        ("rewards", rewards),
        ("dones", dones),
        ("terminal_reasons", terminal_reasons),
    ):
        if values.shape != (transition_count,):
            raise ValueError(f"{name} must have shape [N]")
    if episode_ids.shape != (episode_count,):
        raise ValueError("episode_ids must have shape [E]")
    if frame_offsets.shape != (episode_count + 1,):
        raise ValueError("frame_offsets must have shape [E + 1]")
    if transition_offsets.shape != (episode_count + 1,):
        raise ValueError("transition_offsets must have shape [E + 1]")
    if episode_count == 0:
        raise ValueError("visual dataset must contain at least one episode")
    if not np.array_equal(episode_ids, np.unique(episode_ids)):
        raise ValueError("episode_ids must be strictly increasing")

    for name, values in (
        ("states", states),
        ("actions", actions),
        ("rewards", rewards),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")

    if (
        frame_offsets[0] != 0
        or frame_offsets[-1] != frame_count
        or np.any(np.diff(frame_offsets) <= 0)
    ):
        raise ValueError("frame_offsets must cover every frame exactly once")
    if (
        transition_offsets[0] != 0
        or transition_offsets[-1] != transition_count
        or np.any(np.diff(transition_offsets) <= 0)
    ):
        raise ValueError(
            "transition_offsets must cover every transition exactly once"
        )

    frame_lengths = np.diff(frame_offsets)
    transition_lengths = np.diff(transition_offsets)
    for index, episode_id in enumerate(episode_ids):
        if frame_lengths[index] != transition_lengths[index] + 1:
            raise ValueError(
                f"episode {int(episode_id)} must own T plus one frames"
            )
        start = int(transition_offsets[index])
        stop = int(transition_offsets[index + 1])
        if np.any(dones[start : stop - 1]) or not bool(dones[stop - 1]):
            raise ValueError(
                f"episode {int(episode_id)} has invalid terminal placement"
            )
        if (
            np.any(terminal_reasons[start : stop - 1] != "")
            or str(terminal_reasons[stop - 1]) not in VALID_TERMINAL_REASONS
        ):
            raise ValueError(
                f"episode {int(episode_id)} has invalid terminal reasons"
            )

    if frame_count != transition_count + episode_count:
        raise ValueError("visual dataset requires F = N + E")

    expected_scene = scene_from_env(CarEnv())
    scene_arrays = {
        "scene_world_bounds": (
            np.asarray(expected_scene.world_bounds, dtype=np.float64),
            (4,),
        ),
        "scene_obstacle": (
            np.asarray(expected_scene.obstacle, dtype=np.float64),
            (2,),
        ),
        "scene_goal": (
            np.asarray(expected_scene.goal, dtype=np.float64),
            (2,),
        ),
    }
    for name, (expected, shape) in scene_arrays.items():
        values = _require_dtype(dataset, name, np.float64)
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")
        if not np.array_equal(values, expected):
            raise ValueError(
                f"{name} does not match the schema-v1 default scene"
            )

    scene_scalars = {
        "scene_obstacle_radius": expected_scene.obstacle_radius,
        "scene_goal_radius": expected_scene.goal_radius,
        "scene_car_radius": expected_scene.car_radius,
        "scene_dt": expected_scene.dt,
    }
    for name, expected in scene_scalars.items():
        values = _require_dtype(dataset, name, np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a scalar")
        if float(values.item()) != expected:
            raise ValueError(
                f"{name} does not match the schema-v1 default scene"
            )


def build_visual_dataset(
    source: Mapping[str, np.ndarray],
) -> VisualDataset:
    """Render canonical episodes into one flattened schema-v1 artifact."""

    episodes = reconstruct_episodes(source)
    scene = scene_from_env(CarEnv())
    episode_count = len(episodes)
    transition_count = sum(
        episode.transition_count for episode in episodes
    )
    frame_count = transition_count + episode_count

    frames = np.empty(
        (frame_count, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )
    aligned_states = np.empty((frame_count, 4), dtype=np.float64)
    actions = np.empty((transition_count, 2), dtype=np.float64)
    rewards = np.empty((transition_count,), dtype=np.float64)
    dones = np.empty((transition_count,), dtype=np.bool_)
    terminal_reasons = np.empty(
        (transition_count,),
        dtype=TERMINAL_REASON_DTYPE,
    )
    episode_ids = np.empty((episode_count,), dtype=np.int64)
    frame_offsets = np.zeros((episode_count + 1,), dtype=np.int64)
    transition_offsets = np.zeros((episode_count + 1,), dtype=np.int64)

    frame_cursor = 0
    transition_cursor = 0
    for episode_index, episode in enumerate(episodes):
        physical_states = np.concatenate(
            (episode.states[:1], episode.next_states),
            axis=0,
        )
        next_frame_cursor = frame_cursor + physical_states.shape[0]
        next_transition_cursor = (
            transition_cursor + episode.transition_count
        )
        aligned_states[frame_cursor:next_frame_cursor] = physical_states
        for local_index, state in enumerate(physical_states):
            frames[frame_cursor + local_index] = render_observation(
                state,
                scene=scene,
            )
        actions[transition_cursor:next_transition_cursor] = episode.actions
        rewards[transition_cursor:next_transition_cursor] = episode.rewards
        dones[transition_cursor:next_transition_cursor] = episode.dones
        terminal_reasons[
            transition_cursor:next_transition_cursor
        ] = episode.terminal_reasons
        episode_ids[episode_index] = episode.episode_id

        frame_cursor = next_frame_cursor
        transition_cursor = next_transition_cursor
        frame_offsets[episode_index + 1] = frame_cursor
        transition_offsets[episode_index + 1] = transition_cursor

    dataset: VisualDataset = {
        "schema_version": np.asarray(
            VISUAL_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "image_size": np.asarray(IMAGE_SIZE, dtype=np.int64),
        "context_frames": np.asarray(CONTEXT_FRAMES, dtype=np.int64),
        "renderer_version": np.asarray(RENDERER_VERSION, dtype=np.str_),
        "pillow_version": np.asarray(PILLOW_VERSION, dtype=np.str_),
        "frames": frames,
        "states": aligned_states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "terminal_reasons": terminal_reasons,
        "episode_ids": episode_ids,
        "frame_offsets": frame_offsets,
        "transition_offsets": transition_offsets,
        "scene_world_bounds": np.asarray(
            scene.world_bounds,
            dtype=np.float64,
        ),
        "scene_obstacle": np.asarray(scene.obstacle, dtype=np.float64),
        "scene_obstacle_radius": np.asarray(
            scene.obstacle_radius,
            dtype=np.float64,
        ),
        "scene_goal": np.asarray(scene.goal, dtype=np.float64),
        "scene_goal_radius": np.asarray(
            scene.goal_radius,
            dtype=np.float64,
        ),
        "scene_car_radius": np.asarray(
            scene.car_radius,
            dtype=np.float64,
        ),
        "scene_dt": np.asarray(scene.dt, dtype=np.float64),
    }
    validate_visual_dataset(dataset)
    return dataset


def summarize_visual_dataset(
    dataset: Mapping[str, np.ndarray],
) -> dict[str, int | str]:
    """Return JSON-safe schema and temporal-window counts."""

    validate_visual_dataset(dataset)
    transition_lengths = np.diff(dataset["transition_offsets"])
    return {
        "schema_version": int(dataset["schema_version"].item()),
        "renderer_version": str(dataset["renderer_version"].item()),
        "image_size": int(dataset["image_size"].item()),
        "context_frames": int(dataset["context_frames"].item()),
        "episodes": int(dataset["episode_ids"].size),
        "transitions": int(dataset["actions"].shape[0]),
        "frames": int(dataset["frames"].shape[0]),
        "four_frame_eligible_episodes": int(
            np.count_nonzero(transition_lengths >= CONTEXT_FRAMES)
        ),
        "one_step_visual_samples": int(
            np.maximum(
                0,
                transition_lengths - (CONTEXT_FRAMES - 1),
            ).sum()
        ),
    }


def save_visual_dataset(
    dataset: Mapping[str, np.ndarray],
    output: Path | str,
) -> Path:
    """Validate and exclusively create one compressed visual NPZ."""

    validate_visual_dataset(dataset)
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"visual dataset already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise NotADirectoryError(
            f"visual dataset parent is not a directory: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(values)
                for name, values in dataset.items()
            },
        )
    return path


def load_visual_dataset(path: Path | str) -> VisualDataset:
    """Load and validate one schema-versioned visual artifact."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"visual dataset is not a regular file: {input_path}"
        )
    with np.load(input_path, allow_pickle=False) as loaded:
        missing = set(REQUIRED_VISUAL_ARRAYS) - set(loaded.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"visual dataset is missing arrays: {names}")
        dataset = {
            name: np.asarray(loaded[name])
            for name in loaded.files
        }
    validate_visual_dataset(dataset)
    return dataset
