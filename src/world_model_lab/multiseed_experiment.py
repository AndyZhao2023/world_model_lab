"""Aggregate and plot paired H1/H10 multi-seed diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)
COMPARISON_TOLERANCE = 1e-12


def _dense_curves(metrics: dict[str, Any]) -> dict[str, Any]:
    rollout = metrics["rollout"]
    # ``step_curves`` is the diagnostics schema-v2 name. ``curves`` remains an
    # accepted alias for compact synthetic or previously serialized records.
    if "step_curves" in rollout:
        return rollout["step_curves"]
    return rollout["curves"]


def _metric_curve(metrics: dict[str, Any], name: str) -> np.ndarray:
    curves = _dense_curves(metrics)["free_rollout"]
    if name == "normalized_total":
        values = curves["normalized_mse"]["total"]
    else:
        values = curves["physical"][name]
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"invalid curve for metric: {name}")
    return result


def _series_statistics(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=1).tolist(),
    }


def _steps(metrics: dict[str, Any]) -> np.ndarray:
    result = np.asarray(_dense_curves(metrics)["steps"])
    if result.ndim != 1 or result.size == 0:
        raise ValueError("rollout steps must be a non-empty one-dimensional array")
    return result


def _validate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: record["seed"])
    if len(ordered) < 2:
        raise ValueError("at least two training seeds are required")

    seeds = [record["seed"] for record in ordered]
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
        for seed in seeds
    ):
        raise ValueError("training seeds must be non-negative integers")
    if len(set(map(int, seeds))) != len(seeds):
        raise ValueError("training seeds must be unique")
    return ordered


def _paired_values(
    h1: np.ndarray,
    h10: np.ndarray,
) -> dict[str, list[float] | list[int]]:
    delta = h10 - h1
    return {
        **_series_statistics(delta),
        "improved_seed_count": np.count_nonzero(
            delta < -COMPARISON_TOLERANCE,
            axis=0,
        ).tolist(),
        "worse_seed_count": np.count_nonzero(
            delta > COMPARISON_TOLERANCE,
            axis=0,
        ).tolist(),
        "equal_seed_count": np.count_nonzero(
            np.abs(delta) <= COMPARISON_TOLERANCE,
            axis=0,
        ).tolist(),
    }


def build_experiment_summary(
    records: Iterable[dict[str, Any]],
    snapshot_horizons: Iterable[int],
    split_seed: int = 0,
) -> dict[str, Any]:
    """Build a JSON-safe summary from paired diagnostics records."""

    ordered = _validate_records(records)
    reference_steps = _steps(ordered[0]["h1_metrics"])
    for record in ordered:
        for model_name in ("h1", "h10"):
            if not np.array_equal(
                _steps(record[f"{model_name}_metrics"]),
                reference_steps,
            ):
                raise ValueError("all diagnostics must use identical steps arrays")

    metric_arrays: dict[str, dict[str, np.ndarray]] = {}
    metric_summaries: dict[str, Any] = {}
    for name in METRIC_NAMES:
        h1 = np.stack(
            [_metric_curve(record["h1_metrics"], name) for record in ordered]
        )
        h10 = np.stack(
            [_metric_curve(record["h10_metrics"], name) for record in ordered]
        )
        if (
            h1.shape[1:] != reference_steps.shape
            or h10.shape[1:] != reference_steps.shape
        ):
            raise ValueError(f"curve length does not match steps for metric: {name}")
        metric_arrays[name] = {"h1": h1, "h10": h10}
        metric_summaries[name] = {
            "h1": _series_statistics(h1),
            "h10": _series_statistics(h10),
            "paired_delta": _paired_values(h1, h10),
        }

    step_indexes = {
        int(step): index for index, step in enumerate(reference_steps.tolist())
    }
    sparse_steps = []
    for horizon in snapshot_horizons:
        value = int(horizon)
        if value in step_indexes and value not in sparse_steps:
            sparse_steps.append(value)

    sparse_horizons: dict[str, Any] = {}
    for horizon in sparse_steps:
        index = step_indexes[horizon]
        sparse_horizons[str(horizon)] = {
            name: {
                "h1": {
                    statistic: metric_summaries[name]["h1"][statistic][index]
                    for statistic in ("mean", "std")
                },
                "h10": {
                    statistic: metric_summaries[name]["h10"][statistic][index]
                    for statistic in ("mean", "std")
                },
                "paired_delta": {
                    statistic: metric_summaries[name]["paired_delta"][statistic][
                        index
                    ]
                    for statistic in (
                        "mean",
                        "std",
                        "improved_seed_count",
                        "worse_seed_count",
                        "equal_seed_count",
                    )
                },
            }
            for name in METRIC_NAMES
        }

    per_seed: dict[str, Any] = {}
    for seed_index, record in enumerate(ordered):
        heading_delta = (
            metric_arrays["heading_degrees"]["h10"][seed_index]
            - metric_arrays["heading_degrees"]["h1"][seed_index]
        )
        regression_indexes = np.flatnonzero(heading_delta > COMPARISON_TOLERANCE)
        first_regression = (
            None
            if regression_indexes.size == 0
            else int(reference_steps[int(regression_indexes[0])])
        )
        seed_snapshots: dict[str, Any] = {}
        for horizon in sparse_steps:
            index = step_indexes[horizon]
            seed_snapshots[str(horizon)] = {}
            for name in METRIC_NAMES:
                h1_value = float(metric_arrays[name]["h1"][seed_index, index])
                h10_value = float(metric_arrays[name]["h10"][seed_index, index])
                seed_snapshots[str(horizon)][name] = {
                    "h1": h1_value,
                    "h10": h10_value,
                    "delta_h10_minus_h1": h10_value - h1_value,
                }
        per_seed[str(int(record["seed"]))] = {
            "h1_metrics_path": str(record["h1_metrics_path"]),
            "h10_metrics_path": str(record["h10_metrics_path"]),
            "first_heading_regression_step": first_regression,
            "sparse_horizons": seed_snapshots,
        }

    summary = {
        "schema_version": 1,
        "seeds": [int(record["seed"]) for record in ordered],
        "split_seed": int(split_seed),
        "steps": reference_steps.tolist(),
        "metrics": metric_summaries,
        "sparse_horizons": sparse_horizons,
        "per_seed": per_seed,
    }
    json.dumps(summary, allow_nan=False)
    return summary


def write_summary_csv(summary: dict[str, Any], output_path: Path | str) -> Path:
    """Write sparse per-seed paired values in a stable tidy layout."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "seed",
        "horizon",
        "metric",
        "h1",
        "h10",
        "delta_h10_minus_h1",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in sorted(summary["seeds"]):
            seed_summary = summary["per_seed"][str(seed)]
            for horizon in sorted(
                seed_summary["sparse_horizons"],
                key=int,
            ):
                for name in METRIC_NAMES:
                    values = seed_summary["sparse_horizons"][horizon][name]
                    writer.writerow(
                        {
                            "seed": seed,
                            "horizon": int(horizon),
                            "metric": name,
                            **values,
                        }
                    )
    return output


def plot_multiseed_comparison(
    summary: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Plot H1 and H10 dense means with sample-standard-deviation bands."""

    steps = np.asarray(summary["steps"], dtype=np.int64)
    metric_specs = (
        ("position", "Position error", "macro mean (m)"),
        ("heading_degrees", "Heading error", "macro mean (degrees)"),
        ("velocity", "Velocity error", "macro mean (m/s)"),
        ("normalized_total", "Normalized total error", "normalized MSE"),
    )
    model_specs = (
        ("h1", "H1", "#4c78a8"),
        ("h10", "H10", "#e45756"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (name, title, ylabel) in zip(
        axes.flat,
        metric_specs,
        strict=True,
    ):
        for model_name, label, color in model_specs:
            statistics = summary["metrics"][name][model_name]
            mean = np.asarray(statistics["mean"], dtype=np.float64)
            std = np.asarray(statistics["std"], dtype=np.float64)
            axis.plot(steps, mean, linewidth=2, label=label, color=color)
            axis.fill_between(
                steps,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
            )
        axis.set(title=title, xlabel="rollout step", ylabel=ylabel)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("H1 vs H10 Multi-Seed Free Rollout")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
