"""Held-out calibration and rollout diagnostics for H10 ensembles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import (
    RolloutWindow,
    compute_normalized_squared_errors,
    compute_state_errors,
    summarize_values,
)
from .ensemble import (
    DISAGREEMENT_NAMES,
    WorldModelEnsemble,
    predict_ensemble_next_states,
    rollout_ensemble,
)


METRIC_NAMES = DISAGREEMENT_NAMES
METRIC_LABELS = {
    "position": ("Position", "error (m)", "disagreement (m)"),
    "heading_degrees": (
        "Heading",
        "error (degrees)",
        "disagreement (degrees)",
    ),
    "velocity": ("Velocity", "error (m/s)", "disagreement (m/s)"),
    "normalized_total": (
        "Normalized total",
        "normalized MSE",
        "normalized RMS disagreement",
    ),
}


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError("correlation values must be matching non-empty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation values must be finite")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient = float(np.corrcoef(x, y)[0, 1])
    return coefficient if np.isfinite(coefficient) else None


def build_calibration_bins(
    disagreement: np.ndarray,
    errors: np.ndarray,
    *,
    bin_count: int,
) -> list[dict[str, float | int]]:
    uncertainty = np.asarray(disagreement, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    if bin_count <= 0:
        raise ValueError("calibration_bins must be positive")
    if (
        uncertainty.ndim != 1
        or error.shape != uncertainty.shape
        or uncertainty.size == 0
    ):
        raise ValueError("calibration inputs must be matching non-empty vectors")
    if not np.all(np.isfinite(uncertainty)) or not np.all(np.isfinite(error)):
        raise ValueError("calibration inputs must be finite")
    ordered = np.argsort(uncertainty, kind="stable")
    return [
        {
            "count": int(indices.size),
            "disagreement_mean": float(np.mean(uncertainty[indices])),
            "error_mean": float(np.mean(error[indices])),
        }
        for indices in np.array_split(ordered, min(bin_count, ordered.size))
        if indices.size
    ]


def _state_error_arrays(
    predicted_states: np.ndarray,
    true_states: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, np.ndarray]:
    errors = compute_state_errors(predicted_states, true_states)
    errors["normalized_total"] = compute_normalized_squared_errors(
        predicted_states,
        true_states,
        target_std,
    )["total"]
    return errors


def evaluate_one_step_calibration(
    ensemble: WorldModelEnsemble,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    true_next_states: np.ndarray,
    calibration_bins: int,
) -> dict[str, Any]:
    prediction = predict_ensemble_next_states(ensemble, states, actions)
    ensemble_errors = _state_error_arrays(
        prediction.mean_next_states,
        true_next_states,
        ensemble.target_std,
    )
    member_errors = {
        name: np.stack(
            [
                _state_error_arrays(
                    member_prediction,
                    true_next_states,
                    ensemble.target_std,
                )[name]
                for member_prediction in prediction.member_next_states
            ]
        )
        for name in METRIC_NAMES
    }
    metrics = {}
    for name in METRIC_NAMES:
        mean_member_per_sample = np.mean(member_errors[name], axis=0)
        bins = build_calibration_bins(
            prediction.disagreement[name],
            ensemble_errors[name],
            bin_count=calibration_bins,
        )
        lowest = float(bins[0]["error_mean"])
        highest = float(bins[-1]["error_mean"])
        metrics[name] = {
            "ensemble_error": summarize_values(ensemble_errors[name]),
            "mean_member_error": summarize_values(mean_member_per_sample),
            "member_error_means": {
                str(seed): float(np.mean(values))
                for seed, values in zip(ensemble.seeds, member_errors[name])
            },
            "ensemble_gain_mean": float(
                np.mean(mean_member_per_sample) - np.mean(ensemble_errors[name])
            ),
            "pearson_correlation": pearson_correlation(
                prediction.disagreement[name],
                ensemble_errors[name],
            ),
            "lowest_bin_error_mean": lowest,
            "highest_bin_error_mean": highest,
            "highest_to_lowest_risk_ratio": (
                None if lowest == 0.0 else highest / lowest
            ),
            "calibration_bins": bins,
        }
    return {"samples": int(np.asarray(states).shape[0]), "metrics": metrics}


def _empty_episode_metric_records(
    episode_ids: tuple[int, ...],
) -> dict[int, list[np.ndarray]]:
    return {episode_id: [] for episode_id in episode_ids}


def _episode_macro_curve(
    records: Mapping[int, list[np.ndarray]],
    *,
    episode_ids: tuple[int, ...],
) -> np.ndarray:
    per_episode = []
    for episode_id in episode_ids:
        values = records[episode_id]
        if not values:
            raise ValueError(f"eligible episode {episode_id} has no rollout windows")
        per_episode.append(np.mean(np.stack(values), axis=0))
    curve = np.mean(np.stack(per_episode), axis=0)
    if not np.all(np.isfinite(curve)):
        raise ValueError("rollout metric curves must contain only finite values")
    return curve


def evaluate_ensemble_rollouts(
    ensemble: WorldModelEnsemble,
    *,
    windows: Iterable[RolloutWindow],
    eligible_episode_ids: np.ndarray,
) -> dict[str, Any]:
    selected_windows = tuple(windows)
    eligible = np.asarray(eligible_episode_ids)
    if eligible.ndim != 1 or eligible.size == 0:
        raise ValueError("eligible_episode_ids must be a non-empty vector")
    episode_ids = tuple(int(value) for value in eligible.tolist())
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("eligible_episode_ids must not contain duplicates")
    if not selected_windows:
        raise ValueError("rollout windows must not be empty")

    horizon = int(np.asarray(selected_windows[0].actions).shape[0])
    if horizon <= 0:
        raise ValueError("rollout windows must have a positive horizon")

    ensemble_records = {
        name: _empty_episode_metric_records(episode_ids)
        for name in METRIC_NAMES
    }
    disagreement_records = {
        name: _empty_episode_metric_records(episode_ids)
        for name in METRIC_NAMES
    }
    member_records = {
        name: [
            _empty_episode_metric_records(episode_ids)
            for _ in ensemble.members
        ]
        for name in METRIC_NAMES
    }
    window_ensemble_errors = {name: [] for name in METRIC_NAMES}
    window_disagreement = {name: [] for name in METRIC_NAMES}

    for window in selected_windows:
        episode_id = int(window.episode_id)
        if episode_id not in ensemble_records[METRIC_NAMES[0]]:
            raise ValueError(
                f"rollout window episode {episode_id} is not eligible"
            )
        actions = np.asarray(window.actions, dtype=np.float64)
        true_states = np.asarray(window.true_states, dtype=np.float64)
        if actions.shape != (horizon, 2) or true_states.shape != (horizon + 1, 4):
            raise ValueError(
                "rollout windows must share one horizon with true_states [H+1, 4] "
                "and actions [H, 2]"
            )
        if not np.all(np.isfinite(true_states)):
            raise ValueError("rollout true_states must contain only finite values")

        rollout = rollout_ensemble(ensemble, true_states[0], actions)
        expected_states = true_states[1:]
        ensemble_errors = _state_error_arrays(
            rollout.mean_states[1:],
            expected_states,
            ensemble.target_std,
        )
        errors_by_member = [
            _state_error_arrays(
                member_states[1:],
                expected_states,
                ensemble.target_std,
            )
            for member_states in rollout.member_states
        ]
        for name in METRIC_NAMES:
            ensemble_values = ensemble_errors[name]
            disagreement_values = np.asarray(rollout.disagreement[name][1:])
            ensemble_records[name][episode_id].append(ensemble_values)
            disagreement_records[name][episode_id].append(disagreement_values)
            window_ensemble_errors[name].append(ensemble_values)
            window_disagreement[name].append(disagreement_values)
            for member_index, errors in enumerate(errors_by_member):
                member_records[name][member_index][episode_id].append(errors[name])

    metrics: dict[str, dict[str, list[float | None]]] = {}
    for name in METRIC_NAMES:
        ensemble_curve = _episode_macro_curve(
            ensemble_records[name],
            episode_ids=episode_ids,
        )
        disagreement_curve = _episode_macro_curve(
            disagreement_records[name],
            episode_ids=episode_ids,
        )
        member_curves = np.stack(
            [
                _episode_macro_curve(records, episode_ids=episode_ids)
                for records in member_records[name]
            ]
        )
        per_window_errors = np.stack(window_ensemble_errors[name])
        per_window_disagreement = np.stack(window_disagreement[name])
        metrics[name] = {
            "ensemble_error_mean": ensemble_curve.tolist(),
            "mean_member_error_mean": np.mean(member_curves, axis=0).tolist(),
            "min_member_error_mean": np.min(member_curves, axis=0).tolist(),
            "max_member_error_mean": np.max(member_curves, axis=0).tolist(),
            "disagreement_mean": disagreement_curve.tolist(),
            "pearson_correlation": [
                pearson_correlation(
                    per_window_disagreement[:, step_index],
                    per_window_errors[:, step_index],
                )
                for step_index in range(horizon)
            ],
        }

    return {
        "steps": list(range(1, horizon + 1)),
        "episodes": len(episode_ids),
        "windows": len(selected_windows),
        "aggregation": {
            "accuracy": "window mean within episode, then equal episode mean",
            "correlation": "rollout windows",
        },
        "metrics": metrics,
    }


def _required_mapping(
    values: Mapping[str, Any],
    field: str,
    *,
    parent: str,
) -> Mapping[str, Any]:
    if field not in values:
        raise ValueError(f"{parent} is missing {field}")
    result = values[field]
    if not isinstance(result, Mapping):
        raise ValueError(f"{parent}.{field} must be a mapping")
    return result


def _finite_plot_vector(
    values: Any,
    *,
    field: str,
    expected_size: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or (expected_size is not None and array.size != expected_size)
        or not np.all(np.isfinite(array))
    ):
        size_text = (
            "a non-empty finite vector"
            if expected_size is None
            else f"a finite vector of length {expected_size}"
        )
        raise ValueError(f"{field} must be {size_text}")
    return array


def _one_step_plot_values(
    metrics: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    metric_records = _required_mapping(metrics, "metrics", parent="one-step metrics")
    result = {}
    for name in METRIC_NAMES:
        record = _required_mapping(metric_records, name, parent="metrics")
        if "calibration_bins" not in record:
            raise ValueError(f"{name} is missing calibration_bins")
        bins = record["calibration_bins"]
        if not isinstance(bins, (list, tuple)) or not bins:
            raise ValueError(f"{name}.calibration_bins must be a non-empty list")
        disagreement = []
        errors = []
        for index, item in enumerate(bins):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"{name}.calibration_bins[{index}] must be a mapping"
                )
            for field, destination in (
                ("disagreement_mean", disagreement),
                ("error_mean", errors),
            ):
                if field not in item:
                    raise ValueError(
                        f"{name}.calibration_bins[{index}] is missing {field}"
                    )
                destination.append(item[field])
        result[name] = (
            _finite_plot_vector(
                disagreement,
                field=f"{name}.calibration_bins.disagreement_mean",
            ),
            _finite_plot_vector(
                errors,
                field=f"{name}.calibration_bins.error_mean",
            ),
        )
    return result


def _rollout_plot_values(
    metrics: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if "steps" not in metrics:
        raise ValueError("rollout metrics is missing steps")
    steps = _finite_plot_vector(metrics["steps"], field="steps")
    metric_records = _required_mapping(metrics, "metrics", parent="rollout metrics")
    result = {}
    for name in METRIC_NAMES:
        record = _required_mapping(metric_records, name, parent="metrics")
        curves = []
        for field in (
            "ensemble_error_mean",
            "mean_member_error_mean",
            "disagreement_mean",
        ):
            if field not in record:
                raise ValueError(f"{name} is missing {field}")
            curves.append(
                _finite_plot_vector(
                    record[field],
                    field=f"{name}.{field}",
                    expected_size=steps.size,
                )
            )
        result[name] = tuple(curves)
    return steps, result


def plot_one_step_calibration(
    metrics: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    values = _one_step_plot_values(metrics)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    try:
        for axis, name in zip(axes.flat, METRIC_NAMES):
            title, error_label, disagreement_label = METRIC_LABELS[name]
            disagreement, errors = values[name]
            axis.plot(disagreement, errors, marker="o")
            axis.set_title(title)
            axis.set_xlabel(disagreement_label)
            axis.set_ylabel(error_label)
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output, dpi=160)
    finally:
        plt.close(figure)
    return output


def plot_rollout_uncertainty(
    metrics: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    steps, values = _rollout_plot_values(metrics)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    try:
        for axis, name in zip(axes.flat, METRIC_NAMES):
            title, error_label, disagreement_label = METRIC_LABELS[name]
            ensemble_error, member_error, disagreement = values[name]
            ensemble_line = axis.plot(
                steps,
                ensemble_error,
                marker="o",
                label="Ensemble mean",
            )[0]
            member_line = axis.plot(
                steps,
                member_error,
                marker="o",
                label="Mean member",
            )[0]
            axis.set_title(title)
            axis.set_xlabel("Rollout step")
            axis.set_ylabel(error_label)
            axis.grid(alpha=0.3)

            disagreement_axis = axis.twinx()
            disagreement_line = disagreement_axis.plot(
                steps,
                disagreement,
                color="tab:green",
                linestyle="--",
                label="Disagreement",
            )[0]
            disagreement_axis.set_ylabel(disagreement_label)
            axis.legend(
                [ensemble_line, member_line, disagreement_line],
                ["Ensemble mean", "Mean member", "Disagreement"],
                loc="best",
            )
        figure.tight_layout()
        figure.savefig(output, dpi=160)
    finally:
        plt.close(figure)
    return output
