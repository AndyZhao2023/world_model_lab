"""Held-out calibration and rollout diagnostics for H10 ensembles."""

from __future__ import annotations

from typing import Any

import numpy as np

from .diagnostics import (
    compute_normalized_squared_errors,
    compute_state_errors,
    summarize_values,
)
from .ensemble import (
    DISAGREEMENT_NAMES,
    WorldModelEnsemble,
    predict_ensemble_next_states,
)


METRIC_NAMES = DISAGREEMENT_NAMES


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError("correlation values must be matching non-empty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation values must be finite")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


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
