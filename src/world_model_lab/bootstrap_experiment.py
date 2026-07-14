"""Compare seed-only and episode-bootstrap ensemble diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .ensemble import DISAGREEMENT_NAMES


METRIC_NAMES = DISAGREEMENT_NAMES
COMPARISON_FIELDS = (
    "evaluation",
    "horizon",
    "metric",
    "baseline_error",
    "bootstrap_error",
    "error_delta",
    "baseline_disagreement",
    "bootstrap_disagreement",
    "disagreement_delta",
    "baseline_correlation",
    "bootstrap_correlation",
    "correlation_delta",
)


def write_comparison_csv(
    comparison: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    """Write the comparison in a stable tidy layout."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for metric in METRIC_NAMES:
            writer.writerow(
                {
                    "evaluation": "one_step",
                    "horizon": "",
                    "metric": metric,
                    **comparison["one_step"][metric],
                }
            )
        for horizon in comparison["horizons"]:
            for metric in METRIC_NAMES:
                writer.writerow(
                    {
                        "evaluation": "rollout",
                        "horizon": horizon,
                        "metric": metric,
                        **comparison["rollout"][str(horizon)][metric],
                    }
                )
    return output


def plot_bootstrap_comparison(
    comparison: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    """Plot baseline and bootstrap rollout diagnostics."""

    horizons = np.asarray(comparison["horizons"], dtype=np.int64)
    metric_specs = (
        ("position", "Position", "error (m)"),
        ("heading_degrees", "Heading", "error (degrees)"),
        ("velocity", "Velocity", "error (m/s)"),
        ("normalized_total", "Normalized total", "normalized error"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for index, (axis, (metric, title, error_label)) in enumerate(
        zip(axes.flat, metric_specs, strict=True)
    ):
        records = [
            comparison["rollout"][str(int(horizon))][metric]
            for horizon in horizons
        ]
        baseline_error = np.asarray(
            [record["baseline_error"] for record in records],
            dtype=np.float64,
        )
        bootstrap_error = np.asarray(
            [record["bootstrap_error"] for record in records],
            dtype=np.float64,
        )
        baseline_correlation = np.asarray(
            [
                np.nan
                if record["baseline_correlation"] is None
                else record["baseline_correlation"]
                for record in records
            ],
            dtype=np.float64,
        )
        bootstrap_correlation = np.asarray(
            [
                np.nan
                if record["bootstrap_correlation"] is None
                else record["bootstrap_correlation"]
                for record in records
            ],
            dtype=np.float64,
        )

        axis.plot(
            horizons,
            baseline_error,
            color="#4c78a8",
            linewidth=2,
            label="Baseline error",
        )
        axis.plot(
            horizons,
            bootstrap_error,
            color="#f58518",
            linewidth=2,
            label="Bootstrap error",
        )
        axis.set_title(title)
        axis.set_ylabel(error_label)
        axis.grid(True, alpha=0.3)
        if index >= 2:
            axis.set_xlabel("rollout horizon")

        correlation_axis = axis.twinx()
        correlation_axis.plot(
            horizons,
            baseline_correlation,
            color="#4c78a8",
            linestyle="--",
            label="Baseline correlation",
        )
        correlation_axis.plot(
            horizons,
            bootstrap_correlation,
            color="#f58518",
            linestyle="--",
            label="Bootstrap correlation",
        )
        correlation_axis.set_ylabel("Pearson correlation")

        lines = axis.lines + correlation_axis.lines
        axis.legend(lines, [line.get_label() for line in lines], fontsize=8)

    figure.suptitle("Seed-only vs Episode-bootstrap Ensemble")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _required_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_value(
    values: Mapping[str, Any],
    field: str,
    *,
    parent: str,
) -> Any:
    if field not in values:
        raise ValueError(f"{parent}.{field} is missing")
    return values[field]


def _required_child_mapping(
    values: Mapping[str, Any],
    field: str,
    *,
    parent: str,
) -> Mapping[str, Any]:
    name = f"{parent}.{field}"
    return _required_mapping(
        _required_value(values, field, parent=parent),
        name=name,
    )


def _finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _correlation(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name=name)


def _finite_delta(
    current: float,
    baseline: float,
    *,
    name: str,
) -> float:
    result = current - baseline
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _correlation_delta(
    current: float | None,
    baseline: float | None,
    *,
    name: str,
) -> float | None:
    if current is None or baseline is None:
        return None
    return _finite_delta(current, baseline, name=name)


def _validate_schema(
    value: object,
    *,
    name: str,
) -> Mapping[str, Any]:
    metrics = _required_mapping(value, name=name)
    version = _required_value(metrics, "schema_version", parent=name)
    if (
        isinstance(version, (bool, np.bool_))
        or not isinstance(version, (int, np.integer))
        or int(version) != 1
    ):
        raise ValueError(f"{name}.schema_version must equal 1")
    return metrics


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(horizons)
    except TypeError:
        raise ValueError("horizons must be an iterable") from None
    if not values:
        raise ValueError("horizons must be non-empty")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in values
    ):
        raise ValueError("horizons must contain positive integers")
    normalized = tuple(int(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("horizons must be unique")
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError("horizons must be strictly increasing")
    return normalized


def _extract_one_step_metric(
    metrics: Mapping[str, Any],
    *,
    source: str,
    metric: str,
) -> tuple[float, float | None]:
    one_step = _required_child_mapping(metrics, "one_step", parent=source)
    records = _required_child_mapping(
        one_step,
        "metrics",
        parent=f"{source}.one_step",
    )
    record_path = f"{source}.one_step.metrics.{metric}"
    record = _required_child_mapping(
        records,
        metric,
        parent=f"{source}.one_step.metrics",
    )
    error_path = f"{record_path}.ensemble_error"
    error_record = _required_child_mapping(
        record,
        "ensemble_error",
        parent=record_path,
    )
    error = _finite_float(
        _required_value(error_record, "mean", parent=error_path),
        name=f"{error_path}.mean",
    )
    correlation_path = f"{record_path}.pearson_correlation"
    correlation = _correlation(
        _required_value(record, "pearson_correlation", parent=record_path),
        name=correlation_path,
    )
    return error, correlation


def _extract_rollout_metric(
    metrics: Mapping[str, Any],
    *,
    source: str,
    horizon: int,
    metric: str,
) -> tuple[float, float, float | None]:
    rollout = _required_child_mapping(metrics, "rollout", parent=source)
    snapshots = _required_child_mapping(
        rollout,
        "horizons",
        parent=f"{source}.rollout",
    )
    horizon_key = str(horizon)
    snapshot_path = f"{source}.rollout.horizons.{horizon_key}"
    snapshot = _required_child_mapping(
        snapshots,
        horizon_key,
        parent=f"{source}.rollout.horizons",
    )
    records = _required_child_mapping(
        snapshot,
        "metrics",
        parent=snapshot_path,
    )
    record_path = f"{snapshot_path}.metrics.{metric}"
    record = _required_child_mapping(
        records,
        metric,
        parent=f"{snapshot_path}.metrics",
    )

    error_name = f"{record_path}.ensemble_error_mean"
    error = _finite_float(
        _required_value(record, "ensemble_error_mean", parent=record_path),
        name=error_name,
    )
    disagreement_name = f"{record_path}.disagreement_mean"
    disagreement = _finite_float(
        _required_value(record, "disagreement_mean", parent=record_path),
        name=disagreement_name,
    )
    correlation_name = f"{record_path}.pearson_correlation"
    correlation = _correlation(
        _required_value(record, "pearson_correlation", parent=record_path),
        name=correlation_name,
    )
    return error, disagreement, correlation

def build_bootstrap_comparison(
    baseline_metrics: Mapping[str, Any],
    bootstrap_metrics: Mapping[str, Any],
    *,
    horizons: Iterable[int],
) -> dict[str, Any]:
    """Build a JSON-safe baseline-versus-bootstrap comparison."""

    baseline = _validate_schema(baseline_metrics, name="baseline")
    bootstrap = _validate_schema(bootstrap_metrics, name="bootstrap")
    horizon_values = _validate_horizons(horizons)

    one_step = {}
    for metric in METRIC_NAMES:
        baseline_error, baseline_correlation = _extract_one_step_metric(
            baseline,
            source="baseline",
            metric=metric,
        )
        bootstrap_error, bootstrap_correlation = _extract_one_step_metric(
            bootstrap,
            source="bootstrap",
            metric=metric,
        )
        output_path = f"one_step.{metric}"
        one_step[metric] = {
            "baseline_error": baseline_error,
            "bootstrap_error": bootstrap_error,
            "error_delta": _finite_delta(
                bootstrap_error,
                baseline_error,
                name=f"{output_path}.error_delta",
            ),
            "baseline_correlation": baseline_correlation,
            "bootstrap_correlation": bootstrap_correlation,
            "correlation_delta": _correlation_delta(
                bootstrap_correlation,
                baseline_correlation,
                name=f"{output_path}.correlation_delta",
            ),
        }

    rollout = {}
    for horizon in horizon_values:
        horizon_metrics = {}
        for metric in METRIC_NAMES:
            (
                baseline_error,
                baseline_disagreement,
                baseline_correlation,
            ) = _extract_rollout_metric(
                baseline,
                source="baseline",
                horizon=horizon,
                metric=metric,
            )
            (
                bootstrap_error,
                bootstrap_disagreement,
                bootstrap_correlation,
            ) = _extract_rollout_metric(
                bootstrap,
                source="bootstrap",
                horizon=horizon,
                metric=metric,
            )
            output_path = f"rollout.{horizon}.{metric}"
            horizon_metrics[metric] = {
                "baseline_error": baseline_error,
                "bootstrap_error": bootstrap_error,
                "error_delta": _finite_delta(
                    bootstrap_error,
                    baseline_error,
                    name=f"{output_path}.error_delta",
                ),
                "baseline_disagreement": baseline_disagreement,
                "bootstrap_disagreement": bootstrap_disagreement,
                "disagreement_delta": _finite_delta(
                    bootstrap_disagreement,
                    baseline_disagreement,
                    name=f"{output_path}.disagreement_delta",
                ),
                "baseline_correlation": baseline_correlation,
                "bootstrap_correlation": bootstrap_correlation,
                "correlation_delta": _correlation_delta(
                    bootstrap_correlation,
                    baseline_correlation,
                    name=f"{output_path}.correlation_delta",
                ),
            }
        rollout[str(horizon)] = horizon_metrics

    return {
        "schema_version": 1,
        "delta_definition": "bootstrap minus baseline",
        "horizons": list(horizon_values),
        "one_step": one_step,
        "rollout": rollout,
    }
