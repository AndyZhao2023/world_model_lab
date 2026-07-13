"""Plot JSON-style output from the model diagnostics benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(figure, output_path: Path | str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _plot_grid(
    figure,
    axis,
    values: np.ndarray,
    *,
    extent: list[float],
    title: str,
    colorbar_label: str,
) -> None:
    image = axis.imshow(
        values,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    axis.set(title=title, xlabel="x (m)", ylabel="y (m)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def _position_mean_grid(cells: list[list[Mapping[str, Any]]]) -> np.ndarray:
    return np.asarray(
        [
            [
                np.nan
                if cell.get("position") is None
                else float(cell["position"]["mean"])
                for cell in row
            ]
            for row in cells
        ],
        dtype=np.float64,
    )


def _plot_feature_slice(
    axis,
    feature_slice: Mapping[str, Any],
    *,
    title: str,
    xlabel: str,
) -> None:
    edges = np.asarray(feature_slice["edges"], dtype=np.float64)
    bins = feature_slice["bins"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.asarray(
        [
            np.nan
            if cell.get("position") is None
            else float(cell["position"]["mean"])
            for cell in bins
        ],
        dtype=np.float64,
    )
    counts = np.asarray([cell["count"] for cell in bins], dtype=np.int64)
    axis.plot(centers, means, marker="o", linewidth=2, color="#d95f02")
    axis.set(
        title=title,
        xlabel=xlabel,
        ylabel="position MAE (m)",
    )
    axis.grid(True, alpha=0.3)

    count_axis = axis.twinx()
    count_axis.bar(
        centers,
        counts,
        width=np.diff(edges) * 0.75,
        color="#4c78a8",
        alpha=0.2,
    )
    count_axis.set_ylabel("samples")


def plot_diagnostic_overview(
    metrics: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    """Plot coverage, XY error, and one-dimensional error slices."""

    coverage = metrics["coverage"]
    x_edges = np.asarray(coverage["x_edges"], dtype=np.float64)
    y_edges = np.asarray(coverage["y_edges"], dtype=np.float64)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    _plot_grid(
        figure,
        axes[0, 0],
        np.asarray(coverage["train_counts"], dtype=np.float64),
        extent=extent,
        title="Train XY coverage",
        colorbar_label="transitions",
    )
    _plot_grid(
        figure,
        axes[0, 1],
        np.asarray(coverage["test_counts"], dtype=np.float64),
        extent=extent,
        title="Test XY coverage",
        colorbar_label="transitions",
    )
    _plot_grid(
        figure,
        axes[0, 2],
        _position_mean_grid(metrics["one_step"]["xy_grid"]["cells"]),
        extent=extent,
        title="Test one-step position error",
        colorbar_label="MAE (m)",
    )

    feature_slices = metrics["one_step"]["feature_slices"]
    feature_specs = (
        ("velocity", "Velocity slice", "velocity (m/s)"),
        ("steering", "Steering slice", "steering (rad)"),
        ("acceleration", "Acceleration slice", "acceleration (m/s²)"),
    )
    for axis, (name, title, xlabel) in zip(axes[1], feature_specs, strict=True):
        _plot_feature_slice(
            axis,
            feature_slices[name],
            title=title,
            xlabel=xlabel,
        )

    figure.suptitle("World Model Diagnostic Overview")
    return _save_figure(figure, output_path)


def _physical_rollout_curve(
    rollout: Mapping[str, Any],
    mode_name: str,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    step_curves = rollout.get("step_curves")
    if step_curves is not None:
        return (
            np.asarray(step_curves["steps"], dtype=np.int64),
            np.asarray(
                step_curves[mode_name]["physical"][metric_name],
                dtype=np.float64,
            ),
        )

    horizons = np.asarray(rollout["protocol"]["horizons"], dtype=np.int64)
    values = np.asarray(
        [
            rollout["horizons"][str(int(horizon))][mode_name][metric_name]["mean"]
            for horizon in horizons
        ],
        dtype=np.float64,
    )
    return horizons, values


def plot_rollout_errors(
    metrics: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    """Compare teacher-forced and free-rollout errors over horizon."""

    rollout = metrics["rollout"]
    if int(metrics.get("schema_version", 1)) >= 2 and "step_curves" not in rollout:
        raise ValueError("schema version 2 rollout metrics require step_curves")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = (
        ("position", "Position error", "macro mean (m)"),
        ("heading_degrees", "Heading error", "macro mean (degrees)"),
        ("velocity", "Velocity error", "macro mean (m/s)"),
    )
    modes = (
        ("teacher_forcing", "Teacher forcing", "#4c78a8"),
        ("free_rollout", "Free rollout", "#e45756"),
    )
    for axis, (metric_name, title, ylabel) in zip(
        axes,
        metric_specs,
        strict=True,
    ):
        for mode_name, label, color in modes:
            steps, values = _physical_rollout_curve(
                rollout,
                mode_name,
                metric_name,
            )
            axis.plot(
                steps,
                values,
                marker=None if "step_curves" in rollout else "o",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set(title=title, xlabel="horizon (steps)", ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("Teacher Forcing vs Free Rollout")
    return _save_figure(figure, output_path)


def plot_rollout_loss_components(
    metrics: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    """Plot normalized rollout MSE components for every diagnostic step."""

    rollout = metrics["rollout"]
    step_curves = rollout.get("step_curves")
    if step_curves is None:
        raise ValueError("normalized rollout components require step_curves")
    steps = np.asarray(step_curves["steps"], dtype=np.int64)
    modes = (
        ("teacher_forcing", "Teacher forcing", "#4c78a8"),
        ("free_rollout", "Free rollout", "#e45756"),
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, component_name in zip(
        axes.flat,
        ("x", "y", "heading", "velocity"),
        strict=True,
    ):
        for mode_name, label, color in modes:
            values = np.asarray(
                step_curves[mode_name]["normalized_mse"][component_name],
                dtype=np.float64,
            )
            axis.plot(
                steps,
                values,
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set(
            title=component_name,
            xlabel="rollout step",
            ylabel="normalized MSE",
        )
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("Normalized Rollout Loss Components")
    return _save_figure(figure, output_path)
