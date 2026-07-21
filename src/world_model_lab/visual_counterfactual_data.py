"""Generate matched visual counterfactual targets with the real simulator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .car_env import CarEnv
from .diagnose_visual_rollout import VisualRolloutWindow
from .visual_dataset import validate_visual_dataset
from .visual_observation import (
    IMAGE_SIZE,
    render_observation,
    scene_from_env,
)


_TERMINAL_REASON_DTYPE = np.dtype("<U16")


@dataclass(frozen=True)
class MatchedCounterfactualBatch:
    """One seed's matched simulator branches for selected visual windows."""

    recipient_episode_ids: np.ndarray
    recipient_start_steps: np.ndarray
    donor_window_indices: np.ndarray
    requested_actions: np.ndarray
    applied_actions: np.ndarray
    true_states: np.ndarray
    true_frames: np.ndarray
    valid_steps: np.ndarray
    terminal_steps: np.ndarray
    terminal_reasons: np.ndarray

    def __post_init__(self) -> None:
        requested = np.asarray(self.requested_actions)
        if (
            requested.ndim != 3
            or requested.shape[0] == 0
            or requested.shape[1] == 0
            or requested.shape[2] != 2
            or requested.dtype != np.dtype(np.float64)
        ):
            raise ValueError(
                "requested_actions must be float64 [N, H, 2]"
            )
        count, horizon, _ = requested.shape
        expected = {
            "recipient_episode_ids": ((count,), np.dtype(np.int64)),
            "recipient_start_steps": ((count,), np.dtype(np.int64)),
            "donor_window_indices": ((count,), np.dtype(np.int64)),
            "applied_actions": (
                (count, horizon, 2),
                np.dtype(np.float64),
            ),
            "true_states": (
                (count, horizon, 4),
                np.dtype(np.float64),
            ),
            "true_frames": (
                (count, horizon, IMAGE_SIZE, IMAGE_SIZE, 3),
                np.dtype(np.uint8),
            ),
            "valid_steps": ((count, horizon), np.dtype(np.bool_)),
            "terminal_steps": ((count,), np.dtype(np.int64)),
        }
        for name, (shape, dtype) in expected.items():
            values = np.asarray(getattr(self, name))
            if values.shape != shape or values.dtype != dtype:
                raise ValueError(
                    f"{name} must have shape {list(shape)} and dtype {dtype}"
                )
        reasons = np.asarray(self.terminal_reasons)
        if reasons.shape != (count,) or reasons.dtype.kind != "U":
            raise ValueError(
                "terminal_reasons must be a unicode vector matching N"
            )
        if not (
            np.all(np.isfinite(requested))
            and np.all(np.isfinite(self.applied_actions))
        ):
            raise ValueError("counterfactual actions must be finite")
        donors = np.asarray(self.donor_window_indices)
        if (
            not np.array_equal(np.sort(donors), np.arange(count))
            or np.any(donors == np.arange(count))
        ):
            raise ValueError(
                "donor_window_indices must be a no-fixed-point permutation"
            )
        if (
            np.any(self.recipient_episode_ids < 0)
            or np.any(self.recipient_start_steps < 3)
        ):
            raise ValueError("recipient episode IDs or start steps are invalid")

        states = np.asarray(self.true_states)
        frames = np.asarray(self.true_frames)
        valid = np.asarray(self.valid_steps)
        terminal_steps = np.asarray(self.terminal_steps)
        for index in range(count):
            valid_count = int(np.sum(valid[index]))
            if (
                valid_count <= 0
                or not np.all(valid[index, :valid_count])
                or np.any(valid[index, valid_count:])
            ):
                raise ValueError(
                    "valid_steps must be a non-empty true prefix per branch"
                )
            if not np.all(np.isfinite(states[index, :valid_count])):
                raise ValueError("valid counterfactual states must be finite")
            if not np.all(np.isnan(states[index, valid_count:])):
                raise ValueError("invalid counterfactual states must be NaN")
            if np.any(frames[index, valid_count:] != 0):
                raise ValueError("invalid counterfactual frames must be zero")
            terminal_step = int(terminal_steps[index])
            reason = str(reasons[index])
            if terminal_step == -1:
                if valid_count != horizon or reason:
                    raise ValueError(
                        "non-terminal branches must cover the full horizon"
                    )
            elif (
                terminal_step != valid_count
                or terminal_step < 1
                or terminal_step > horizon
                or not reason
            ):
                raise ValueError(
                    "terminal metadata must match the final valid step"
                )

        for name in self.__dataclass_fields__:
            values = np.asarray(getattr(self, name)).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def count(self) -> int:
        return int(self.requested_actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.requested_actions.shape[1])


def _terminal_reason(info: Mapping[str, float | int | bool]) -> str:
    for flag, label in (
        ("reached_goal", "goal"),
        ("collision", "collision"),
        ("out_of_bounds", "out_of_bounds"),
        ("time_limit", "time_limit"),
    ):
        if bool(info[flag]):
            return label
    raise ValueError("terminal simulator step has no terminal reason")


def _validate_windows(
    dataset: Mapping[str, np.ndarray],
    windows: Iterable[VisualRolloutWindow],
) -> tuple[VisualRolloutWindow, ...]:
    selected = tuple(windows)
    if not selected:
        raise ValueError("counterfactual windows must not be empty")
    horizon = int(selected[0].future_actions.shape[0])
    if any(
        window.future_actions.shape != (horizon, 2)
        for window in selected
    ):
        raise ValueError(
            "counterfactual windows must share one positive horizon"
        )
    frame_count = int(np.asarray(dataset["frames"]).shape[0])
    episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    frame_offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
    positions = {
        int(episode_id): index
        for index, episode_id in enumerate(episode_ids.tolist())
    }
    for window in selected:
        if window.episode_id not in positions:
            raise ValueError(
                f"recipient episode {window.episode_id} is missing"
            )
        if (
            window.initial_frame_index < 0
            or window.initial_frame_index >= frame_count
        ):
            raise ValueError(
                "counterfactual initial frame index is out of range"
            )
        position = positions[window.episode_id]
        if not (
            frame_offsets[position]
            <= window.initial_frame_index
            < frame_offsets[position + 1]
        ):
            raise ValueError(
                "counterfactual initial frame belongs to another episode"
            )
        if not np.all(np.isfinite(window.future_actions)):
            raise ValueError("counterfactual future actions must be finite")
    return selected


def build_matched_counterfactual_batch(
    dataset: Mapping[str, np.ndarray],
    windows: Iterable[VisualRolloutWindow],
    donor_window_indices: np.ndarray,
) -> MatchedCounterfactualBatch:
    """Execute donated action rows from each recipient's exact anchor state."""

    validate_visual_dataset(dataset)
    selected = _validate_windows(dataset, windows)
    count = len(selected)
    donors = np.asarray(donor_window_indices)
    if (
        donors.shape != (count,)
        or donors.dtype != np.dtype(np.int64)
        or not np.array_equal(np.sort(donors), np.arange(count))
        or np.any(donors == np.arange(count))
    ):
        raise ValueError(
            "donor_window_indices must be a no-fixed-point int64 permutation"
        )
    requested = np.stack(
        [selected[int(index)].future_actions for index in donors]
    ).astype(np.float64, copy=False)
    horizon = int(requested.shape[1])
    template = CarEnv()
    applied = requested.copy()
    applied[:, :, 0] = np.clip(
        applied[:, :, 0],
        -template.max_steering,
        template.max_steering,
    )
    applied[:, :, 1] = np.clip(
        applied[:, :, 1],
        -template.max_acceleration,
        template.max_acceleration,
    )
    true_states = np.full(
        (count, horizon, 4),
        np.nan,
        dtype=np.float64,
    )
    true_frames = np.zeros(
        (count, horizon, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )
    valid_steps = np.zeros((count, horizon), dtype=np.bool_)
    terminal_steps = np.full(count, -1, dtype=np.int64)
    terminal_reasons = np.full(
        count,
        "",
        dtype=_TERMINAL_REASON_DTYPE,
    )
    states = np.asarray(dataset["states"], dtype=np.float64)
    scene = scene_from_env(template)
    for recipient_index, window in enumerate(selected):
        environment = CarEnv(
            initial_state=states[window.initial_frame_index]
        )
        for step in range(horizon):
            next_state, _, done, info = environment.step(
                float(applied[recipient_index, step, 0]),
                float(applied[recipient_index, step, 1]),
            )
            actual = np.asarray(
                [
                    info["applied_steering"],
                    info["applied_acceleration"],
                ],
                dtype=np.float64,
            )
            if not np.array_equal(
                actual,
                applied[recipient_index, step],
            ):
                raise RuntimeError(
                    "simulator applied actions differ from clipped actions"
                )
            true_states[recipient_index, step] = next_state
            true_frames[recipient_index, step] = render_observation(
                next_state,
                scene=scene,
            )
            valid_steps[recipient_index, step] = True
            if done:
                terminal_steps[recipient_index] = step + 1
                terminal_reasons[recipient_index] = _terminal_reason(info)
                break

    return MatchedCounterfactualBatch(
        recipient_episode_ids=np.asarray(
            [window.episode_id for window in selected],
            dtype=np.int64,
        ),
        recipient_start_steps=np.asarray(
            [window.start_step for window in selected],
            dtype=np.int64,
        ),
        donor_window_indices=donors,
        requested_actions=requested,
        applied_actions=applied,
        true_states=true_states,
        true_frames=true_frames,
        valid_steps=valid_steps,
        terminal_steps=terminal_steps,
        terminal_reasons=terminal_reasons,
    )
