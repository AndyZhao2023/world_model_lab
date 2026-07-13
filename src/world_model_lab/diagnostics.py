"""Reusable diagnostics for learned world-model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .dataset import wrap_angle


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
