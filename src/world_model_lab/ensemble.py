"""Inference and disagreement for compatible world-model ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset import wrap_angle
from .diagnostics import predict_next_states as predict_member_next_states
from .train_world_model import LoadedWorldModel, load_checkpoint


DISAGREEMENT_NAMES = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)


@dataclass(frozen=True)
class WorldModelEnsemble:
    members: tuple[LoadedWorldModel, ...]
    seeds: tuple[int, ...]
    target_std: np.ndarray
    checkpoint_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class EnsemblePrediction:
    member_next_states: np.ndarray
    mean_next_states: np.ndarray
    disagreement: dict[str, np.ndarray]


@dataclass(frozen=True)
class EnsembleRollout:
    member_states: np.ndarray
    mean_states: np.ndarray
    disagreement: dict[str, np.ndarray]


def _circular_mean(values: np.ndarray, *, axis: int) -> np.ndarray:
    return np.arctan2(
        np.mean(np.sin(values), axis=axis),
        np.mean(np.cos(values), axis=axis),
    )


def _aggregate_member_states(
    member_states: np.ndarray,
    *,
    target_std: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    members = np.asarray(member_states, dtype=np.float64)
    if members.ndim != 3 or members.shape[2] != 4:
        raise ValueError("member states must have shape [M, N, 4]")
    if not np.all(np.isfinite(members)):
        raise ValueError("member states must contain only finite values")

    mean = np.mean(members, axis=0)
    mean[:, 2] = _circular_mean(members[:, :, 2], axis=0)
    deviations = members - mean[None, :, :]
    deviations[:, :, 2] = wrap_angle(deviations[:, :, 2])
    position = np.sqrt(
        np.mean(np.sum(np.square(deviations[:, :, :2]), axis=2), axis=0)
    )
    heading = np.degrees(
        np.sqrt(np.mean(np.square(deviations[:, :, 2]), axis=0))
    )
    velocity = np.sqrt(np.mean(np.square(deviations[:, :, 3]), axis=0))
    normalized_total = np.sqrt(
        np.mean(
            np.square(deviations / target_std[None, None, :]),
            axis=(0, 2),
        )
    )
    return mean, {
        "position": position,
        "heading_degrees": heading,
        "velocity": velocity,
        "normalized_total": normalized_total,
    }


def _config_value(member: LoadedWorldModel, name: str):
    if name not in member.training_config:
        raise ValueError(f"checkpoint training_config is missing {name}")
    return member.training_config[name]


def _validate_normalizer_array(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    label: str,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"checkpoint {label} must have shape {list(expected_shape)} "
            "and contain only finite values"
        )
    if positive and np.any(array <= 0.0):
        raise ValueError(f"checkpoint {label} values must be positive")
    return array


def _validate_member(member: LoadedWorldModel) -> None:
    _validate_normalizer_array(
        member.input_normalizer.mean,
        expected_shape=(7,),
        label="input mean",
    )
    _validate_normalizer_array(
        member.input_normalizer.std,
        expected_shape=(7,),
        label="input std",
        positive=True,
    )
    _validate_normalizer_array(
        member.target_normalizer.mean,
        expected_shape=(4,),
        label="target mean",
    )
    _validate_normalizer_array(
        member.target_normalizer.std,
        expected_shape=(4,),
        label="target std",
        positive=True,
    )
    if member.model.input_size != 7 or member.model.output_size != 4:
        raise ValueError("checkpoint model input/output contract differs")
    hidden_size = _config_value(member, "hidden_size")
    if member.model.hidden_size != hidden_size:
        raise ValueError("checkpoint model hidden_size differs from training_config")
    for split_name in ("train", "validation", "test"):
        split_ids = member.split_episode_ids.get(split_name)
        if split_ids is None or np.asarray(split_ids).ndim != 1:
            raise ValueError(
                f"checkpoint {split_name} split episode IDs "
                "must be one-dimensional"
            )


def build_ensemble(
    members: Iterable[LoadedWorldModel],
    *,
    checkpoint_paths: Iterable[Path | str] = (),
) -> WorldModelEnsemble:
    loaded = list(members)
    paths = tuple(Path(path).resolve() for path in checkpoint_paths)
    if len(loaded) < 2:
        raise ValueError("ensemble requires at least two checkpoints")
    if paths and len(paths) != len(loaded):
        raise ValueError("checkpoint path count must match member count")

    paired = []
    for index, member in enumerate(loaded):
        _validate_member(member)
        seed = _config_value(member, "seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
        ):
            raise ValueError(
                "checkpoint training seeds must be non-negative integers"
            )
        paired.append((int(seed), member, paths[index] if paths else None))
    paired.sort(key=lambda item: item[0])
    seeds = tuple(item[0] for item in paired)
    if len(set(seeds)) != len(seeds):
        raise ValueError("checkpoint training seeds must be unique")

    reference = paired[0][1]
    if int(_config_value(reference, "rollout_horizon")) != 10:
        raise ValueError("rollout_horizon must equal 10")
    for _, member, _ in paired[1:]:
        for name in (
            "split_seed",
            "rollout_horizon",
            "rollout_loss_weight",
            "hidden_size",
        ):
            if _config_value(member, name) != _config_value(reference, name):
                raise ValueError(f"checkpoint {name} values differ")
        if int(_config_value(member, "rollout_horizon")) != 10:
            raise ValueError("rollout_horizon must equal 10")
        for split_name in ("train", "validation", "test"):
            if not np.array_equal(
                member.split_episode_ids.get(split_name),
                reference.split_episode_ids.get(split_name),
            ):
                raise ValueError(
                    f"checkpoint {split_name} split episode IDs differ"
                )
        for label, left, right in (
            (
                "input mean",
                member.input_normalizer.mean,
                reference.input_normalizer.mean,
            ),
            (
                "input std",
                member.input_normalizer.std,
                reference.input_normalizer.std,
            ),
            (
                "target mean",
                member.target_normalizer.mean,
                reference.target_normalizer.mean,
            ),
            (
                "target std",
                member.target_normalizer.std,
                reference.target_normalizer.std,
            ),
        ):
            if not np.array_equal(left, right):
                raise ValueError(f"checkpoint {label} values differ")

    ordered_members = tuple(item[1] for item in paired)
    ordered_paths = tuple(item[2] for item in paired if item[2] is not None)
    return WorldModelEnsemble(
        members=ordered_members,
        seeds=seeds,
        target_std=reference.target_normalizer.std.copy(),
        checkpoint_paths=ordered_paths,
    )


def predict_ensemble_next_states(
    ensemble: WorldModelEnsemble,
    states: np.ndarray,
    actions: np.ndarray,
) -> EnsemblePrediction:
    member_next_states = np.stack(
        [
            predict_member_next_states(member, states, actions)
            for member in ensemble.members
        ]
    )
    mean, disagreement = _aggregate_member_states(
        member_next_states,
        target_std=ensemble.target_std,
    )
    return EnsemblePrediction(member_next_states, mean, disagreement)


def load_ensemble(paths: Iterable[Path | str]) -> WorldModelEnsemble:
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("checkpoint paths must be unique")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint is not a regular file: {path}"
            )
    return build_ensemble(
        (load_checkpoint(path) for path in resolved),
        checkpoint_paths=resolved,
    )


def rollout_ensemble(
    ensemble: WorldModelEnsemble,
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> EnsembleRollout:
    initial = np.asarray(initial_state, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    if (
        initial.shape != (4,)
        or action_array.ndim != 2
        or action_array.shape[1] != 2
    ):
        raise ValueError(
            "initial_state must have shape [4] and actions [H, 2]"
        )
    if not np.all(np.isfinite(initial)) or not np.all(
        np.isfinite(action_array)
    ):
        raise ValueError("initial_state and actions must contain only finite values")

    member_states = []
    for member in ensemble.members:
        trajectory = [initial.copy()]
        for action in action_array:
            trajectory.append(
                predict_member_next_states(
                    member,
                    trajectory[-1][None, :],
                    action[None, :],
                )[0]
            )
        member_states.append(np.asarray(trajectory))
    stacked = np.stack(member_states)
    mean, disagreement = _aggregate_member_states(
        stacked,
        target_std=ensemble.target_std,
    )
    return EnsembleRollout(stacked, mean, disagreement)
