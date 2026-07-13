"""Reusable diagnostics for learned world-model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dataset import build_model_inputs, wrap_angle
from .train_world_model import LoadedWorldModel, predict_deltas


ERROR_NAMES = ("position", "heading_degrees", "velocity")


@dataclass(frozen=True)
class RolloutWindow:
    """One fixed-length, contiguous section of a recorded episode."""

    episode_id: int
    start_step: int
    true_states: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class WindowSelection:
    """Selected rollout windows plus eligibility metadata."""

    windows: tuple[RolloutWindow, ...]
    eligible_episode_ids: np.ndarray
    skipped_episode_ids: np.ndarray


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    """Return JSON-safe distribution statistics for a finite vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def compute_state_errors(
    predicted_states: np.ndarray,
    true_states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute absolute errors in physical units for matching state batches."""

    predicted = np.asarray(predicted_states, dtype=np.float64)
    true = np.asarray(true_states, dtype=np.float64)
    if (
        predicted.shape != true.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 4
    ):
        raise ValueError("predicted and true states must have matching shape [N, 4]")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(true)):
        raise ValueError("predicted and true states must contain only finite values")

    difference = predicted - true
    return {
        "position": np.linalg.norm(difference[:, :2], axis=1),
        "heading_degrees": np.degrees(np.abs(wrap_angle(difference[:, 2]))),
        "velocity": np.abs(difference[:, 3]),
    }


def linear_bin_edges(values: np.ndarray, *, bin_count: int) -> np.ndarray:
    """Build stable linear bin edges that also handle a constant feature."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bin values must be a non-empty finite vector")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    lower = float(np.min(array))
    upper = float(np.max(array))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, bin_count + 1, dtype=np.float64)


def _validate_errors(
    errors: Mapping[str, np.ndarray],
    *,
    expected_count: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in ERROR_NAMES:
        if name not in errors:
            raise ValueError(f"errors are missing {name}")
        array = np.asarray(errors[name], dtype=np.float64)
        if array.shape != (expected_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} errors must be a finite vector of length N")
        arrays[name] = array
    return arrays


def _validate_edges(edges: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(edges, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size < 2
        or not np.all(np.isfinite(array))
        or not np.all(np.diff(array) > 0.0)
    ):
        raise ValueError(f"{name} edges must be a finite increasing vector")
    return array


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if np.any(values < edges[0]) or np.any(values > edges[-1]):
        raise ValueError("bin values must lie within the provided edges")
    indices = np.searchsorted(edges, values, side="right") - 1
    return np.clip(indices, 0, edges.size - 2)


def _summarize_cell(
    errors: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    min_bin_count: int,
) -> dict[str, object]:
    count = int(np.count_nonzero(mask))
    cell: dict[str, object] = {"count": count}
    for name in ERROR_NAMES:
        cell[name] = (
            summarize_values(errors[name][mask])
            if count >= min_bin_count
            else None
        )
    return cell


def build_feature_slice(
    values: np.ndarray,
    errors: Mapping[str, np.ndarray],
    *,
    edges: np.ndarray,
    min_bin_count: int,
) -> dict[str, object]:
    """Group error distributions by one state or action feature."""

    feature = np.asarray(values, dtype=np.float64)
    if feature.ndim != 1 or feature.size == 0 or not np.all(np.isfinite(feature)):
        raise ValueError("feature values must be a non-empty finite vector")
    if min_bin_count <= 0:
        raise ValueError("min_bin_count must be positive")
    edge_array = _validate_edges(edges, name="feature")
    error_arrays = _validate_errors(errors, expected_count=feature.size)
    indices = _bin_indices(feature, edge_array)

    bins = []
    for bin_index in range(edge_array.size - 1):
        cell = _summarize_cell(
            error_arrays,
            indices == bin_index,
            min_bin_count=min_bin_count,
        )
        cell.update(
            {
                "lower": float(edge_array[bin_index]),
                "upper": float(edge_array[bin_index + 1]),
            }
        )
        bins.append(cell)
    return {"edges": edge_array.tolist(), "bins": bins}


def build_xy_grid(
    xy: np.ndarray,
    errors: Mapping[str, np.ndarray],
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_bin_count: int,
) -> dict[str, object]:
    """Group errors on a rectangular XY grid stored as rows of Y then X."""

    positions = np.asarray(xy, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or positions.shape[0] == 0
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("xy positions must have shape [N, 2] and be finite")
    if min_bin_count <= 0:
        raise ValueError("min_bin_count must be positive")
    x_edge_array = _validate_edges(x_edges, name="x")
    y_edge_array = _validate_edges(y_edges, name="y")
    error_arrays = _validate_errors(errors, expected_count=positions.shape[0])
    x_indices = _bin_indices(positions[:, 0], x_edge_array)
    y_indices = _bin_indices(positions[:, 1], y_edge_array)

    cells = []
    for y_index in range(y_edge_array.size - 1):
        row = []
        for x_index in range(x_edge_array.size - 1):
            row.append(
                _summarize_cell(
                    error_arrays,
                    (x_indices == x_index) & (y_indices == y_index),
                    min_bin_count=min_bin_count,
                )
            )
        cells.append(row)
    return {
        "x_edges": x_edge_array.tolist(),
        "y_edges": y_edge_array.tolist(),
        "cells": cells,
    }


def _evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    if limit == 1:
        return np.asarray([0], dtype=np.int64)
    return np.rint(np.linspace(0, count - 1, limit)).astype(np.int64)


def select_rollout_windows(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    episode_ids: np.ndarray,
    step_ids: np.ndarray,
    selected_episode_ids: np.ndarray,
    max_horizon: int,
    windows_per_episode: int,
) -> WindowSelection:
    """Select deterministic, equal-length windows from complete episodes."""

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    step_ids = np.asarray(step_ids)
    selected_ids = np.asarray(selected_episode_ids)
    count = states.shape[0]
    if states.shape != (count, 4) or next_states.shape != (count, 4):
        raise ValueError("states and next_states must have shape [N, 4]")
    if actions.shape != (count, 2):
        raise ValueError("actions must have shape [N, 2]")
    if episode_ids.shape != (count,) or step_ids.shape != (count,):
        raise ValueError("episode_ids and step_ids must have shape [N]")
    if selected_ids.ndim != 1 or selected_ids.size == 0:
        raise ValueError("selected_episode_ids must be a non-empty vector")
    if np.unique(selected_ids).size != selected_ids.size:
        raise ValueError("selected_episode_ids must not contain duplicates")
    if max_horizon <= 0 or windows_per_episode <= 0:
        raise ValueError("max_horizon and windows_per_episode must be positive")
    if not all(
        np.all(np.isfinite(values))
        for values in (states, actions, next_states)
    ):
        raise ValueError("transition arrays must contain only finite values")

    windows: list[RolloutWindow] = []
    eligible_ids: list[int] = []
    skipped_ids: list[int] = []
    for episode_id_value in selected_ids.tolist():
        episode_id = int(episode_id_value)
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size == 0:
            raise ValueError(f"episode {episode_id} is missing from the dataset")
        indices = indices[np.argsort(step_ids[indices], kind="stable")]
        ordered_steps = step_ids[indices]
        if not np.array_equal(ordered_steps, np.arange(indices.size)):
            raise ValueError(f"episode {episode_id} step_ids must be contiguous from zero")

        episode_states = states[indices]
        episode_next_states = next_states[indices]
        if indices.size > 1 and not np.allclose(
            episode_states[1:],
            episode_next_states[:-1],
            atol=1e-10,
        ):
            raise ValueError(f"episode {episode_id} transitions are not contiguous")
        if indices.size < max_horizon:
            skipped_ids.append(episode_id)
            continue

        eligible_ids.append(episode_id)
        episode_actions = actions[indices]
        true_states = np.vstack((episode_states[0], episode_next_states))
        valid_start_count = indices.size - max_horizon + 1
        for start_step in _evenly_spaced_indices(
            valid_start_count,
            windows_per_episode,
        ):
            start = int(start_step)
            windows.append(
                RolloutWindow(
                    episode_id=episode_id,
                    start_step=start,
                    true_states=true_states[start : start + max_horizon + 1].copy(),
                    actions=episode_actions[start : start + max_horizon].copy(),
                )
            )

    if not windows:
        raise ValueError("no selected episode is long enough for the maximum horizon")
    return WindowSelection(
        windows=tuple(windows),
        eligible_episode_ids=np.asarray(eligible_ids, dtype=np.int64),
        skipped_episode_ids=np.asarray(skipped_ids, dtype=np.int64),
    )


def predict_next_states(
    world_model: LoadedWorldModel,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Predict next states from raw state/action batches."""

    state_array = np.asarray(states, dtype=np.float64)
    inputs = build_model_inputs(state_array, actions)
    deltas = predict_deltas(world_model, inputs)
    predictions = state_array + deltas
    predictions[:, 2] = wrap_angle(predictions[:, 2])
    return predictions


def _free_rollout(
    world_model: LoadedWorldModel,
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    predicted_states = [np.asarray(initial_state, dtype=np.float64).copy()]
    for action in np.asarray(actions, dtype=np.float64):
        predicted_states.append(
            predict_next_states(
                world_model,
                predicted_states[-1][None, :],
                action[None, :],
            )[0]
        )
    return np.asarray(predicted_states, dtype=np.float64)


def _summarize_error_components(
    errors: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    return {name: summarize_values(np.asarray(errors[name])) for name in ERROR_NAMES}


def _xy_counts(
    xy: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> list[list[int]]:
    counts, _, _ = np.histogram2d(
        np.asarray(xy)[:, 1],
        np.asarray(xy)[:, 0],
        bins=(y_edges, x_edges),
    )
    return counts.astype(np.int64).tolist()


def _empty_episode_records() -> dict[str, list[float]]:
    return {name: [] for name in ERROR_NAMES}


def _summarize_episode_records(
    records: Mapping[int, Mapping[str, list[float]]],
) -> dict[str, object]:
    episode_ids = sorted(records)
    window_count = sum(len(records[episode_id]["position"]) for episode_id in episode_ids)
    summary: dict[str, object] = {
        "episodes": len(episode_ids),
        "windows": window_count,
    }
    for name in ERROR_NAMES:
        per_episode_means = np.asarray(
            [np.mean(records[episode_id][name]) for episode_id in episode_ids],
            dtype=np.float64,
        )
        summary[name] = summarize_values(per_episode_means)
    return summary


def _validate_horizons(horizons: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)
    if not values or any(value <= 0 for value in values):
        raise ValueError("horizons must contain positive integers")
    if values != tuple(sorted(set(values))):
        raise ValueError("horizons must be unique and strictly increasing")
    return values


def build_diagnostic_metrics(
    world_model: LoadedWorldModel,
    *,
    arrays: Mapping[str, np.ndarray],
    split_episode_ids: Mapping[str, np.ndarray],
    horizons: tuple[int, ...] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    xy_bins: int = 12,
    feature_bins: int = 8,
    min_bin_count: int = 5,
) -> dict[str, Any]:
    """Build a JSON-safe one-step and fixed-window diagnostic report."""

    required_arrays = {
        "states",
        "actions",
        "next_states",
        "episode_ids",
        "step_ids",
    }
    missing = required_arrays - set(arrays)
    if missing:
        raise ValueError(f"diagnostic arrays are missing: {', '.join(sorted(missing))}")
    if "train" not in split_episode_ids or "test" not in split_episode_ids:
        raise ValueError("split_episode_ids must contain train and test IDs")
    horizon_values = _validate_horizons(tuple(horizons))
    if windows_per_episode <= 0:
        raise ValueError("windows_per_episode must be positive")
    if xy_bins <= 0 or feature_bins <= 0 or min_bin_count <= 0:
        raise ValueError("bin counts and min_bin_count must be positive")

    states = np.asarray(arrays["states"], dtype=np.float64)
    actions = np.asarray(arrays["actions"], dtype=np.float64)
    next_states = np.asarray(arrays["next_states"], dtype=np.float64)
    episode_ids = np.asarray(arrays["episode_ids"])
    step_ids = np.asarray(arrays["step_ids"])
    count = states.shape[0]
    if states.shape != (count, 4) or next_states.shape != (count, 4):
        raise ValueError("states and next_states must have shape [N, 4]")
    if actions.shape != (count, 2):
        raise ValueError("actions must have shape [N, 2]")
    if episode_ids.shape != (count,) or step_ids.shape != (count,):
        raise ValueError("episode_ids and step_ids must have shape [N]")
    if count == 0:
        raise ValueError("diagnostic arrays must not be empty")
    if not all(
        np.all(np.isfinite(values))
        for values in (states, actions, next_states)
    ):
        raise ValueError("diagnostic transition arrays must contain only finite values")

    train_ids = np.asarray(split_episode_ids["train"])
    test_ids = np.asarray(split_episode_ids["test"])
    if train_ids.ndim != 1 or test_ids.ndim != 1 or train_ids.size == 0 or test_ids.size == 0:
        raise ValueError("train and test episode IDs must be non-empty vectors")
    available_ids = set(np.asarray(np.unique(episode_ids)).tolist())
    missing_train_ids = set(train_ids.tolist()) - available_ids
    missing_test_ids = set(test_ids.tolist()) - available_ids
    if missing_train_ids or missing_test_ids:
        missing_ids = sorted(missing_train_ids | missing_test_ids)
        raise ValueError(f"split episodes are missing from the dataset: {missing_ids}")
    train_mask = np.isin(episode_ids, train_ids)
    test_mask = np.isin(episode_ids, test_ids)

    test_predictions = predict_next_states(
        world_model,
        states[test_mask],
        actions[test_mask],
    )
    one_step_errors = compute_state_errors(test_predictions, next_states[test_mask])
    x_edges = linear_bin_edges(states[:, 0], bin_count=xy_bins)
    y_edges = linear_bin_edges(states[:, 1], bin_count=xy_bins)
    velocity_edges = linear_bin_edges(states[:, 3], bin_count=feature_bins)
    steering_edges = linear_bin_edges(actions[:, 0], bin_count=feature_bins)
    acceleration_edges = linear_bin_edges(actions[:, 1], bin_count=feature_bins)

    selection = select_rollout_windows(
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        selected_episode_ids=test_ids,
        max_horizon=horizon_values[-1],
        windows_per_episode=windows_per_episode,
    )
    window_predictions = []
    for window in selection.windows:
        window_predictions.append(
            (
                window,
                predict_next_states(
                    world_model,
                    window.true_states[:-1],
                    window.actions,
                ),
                _free_rollout(world_model, window.true_states[0], window.actions),
            )
        )

    horizon_metrics: dict[str, object] = {}
    for horizon in horizon_values:
        teacher_records: dict[int, dict[str, list[float]]] = {}
        free_records: dict[int, dict[str, list[float]]] = {}
        for window, teacher_predictions, free_predictions in window_predictions:
            teacher_errors = compute_state_errors(
                teacher_predictions[horizon - 1 : horizon],
                window.true_states[horizon : horizon + 1],
            )
            free_errors = compute_state_errors(
                free_predictions[horizon : horizon + 1],
                window.true_states[horizon : horizon + 1],
            )
            teacher_episode = teacher_records.setdefault(
                window.episode_id,
                _empty_episode_records(),
            )
            free_episode = free_records.setdefault(
                window.episode_id,
                _empty_episode_records(),
            )
            for name in ERROR_NAMES:
                teacher_episode[name].append(float(teacher_errors[name][0]))
                free_episode[name].append(float(free_errors[name][0]))
        horizon_metrics[str(horizon)] = {
            "teacher_forcing": _summarize_episode_records(teacher_records),
            "free_rollout": _summarize_episode_records(free_records),
        }

    return {
        "schema_version": 1,
        "population": {
            "train_episode_ids": [int(value) for value in train_ids.tolist()],
            "test_episode_ids": [int(value) for value in test_ids.tolist()],
            "test_transitions": int(np.count_nonzero(test_mask)),
        },
        "coverage": {
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "train_counts": _xy_counts(
                states[train_mask, :2],
                x_edges=x_edges,
                y_edges=y_edges,
            ),
            "test_counts": _xy_counts(
                states[test_mask, :2],
                x_edges=x_edges,
                y_edges=y_edges,
            ),
        },
        "one_step": {
            "overall": _summarize_error_components(one_step_errors),
            "xy_grid": build_xy_grid(
                states[test_mask, :2],
                one_step_errors,
                x_edges=x_edges,
                y_edges=y_edges,
                min_bin_count=min_bin_count,
            ),
            "feature_slices": {
                "velocity": build_feature_slice(
                    states[test_mask, 3],
                    one_step_errors,
                    edges=velocity_edges,
                    min_bin_count=min_bin_count,
                ),
                "steering": build_feature_slice(
                    actions[test_mask, 0],
                    one_step_errors,
                    edges=steering_edges,
                    min_bin_count=min_bin_count,
                ),
                "acceleration": build_feature_slice(
                    actions[test_mask, 1],
                    one_step_errors,
                    edges=acceleration_edges,
                    min_bin_count=min_bin_count,
                ),
            },
        },
        "rollout": {
            "protocol": {
                "horizons": list(horizon_values),
                "max_horizon": horizon_values[-1],
                "windows_per_episode": windows_per_episode,
                "eligible_episode_ids": selection.eligible_episode_ids.tolist(),
                "skipped_episode_ids": selection.skipped_episode_ids.tolist(),
                "windows": [
                    {
                        "episode_id": window.episode_id,
                        "start_step": window.start_step,
                    }
                    for window in selection.windows
                ],
            },
            "horizons": horizon_metrics,
        },
    }
