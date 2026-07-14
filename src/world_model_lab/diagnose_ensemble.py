"""Held-out calibration and rollout diagnostics for H10 ensembles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import (
    RolloutWindow,
    compute_normalized_squared_errors,
    compute_state_errors,
    select_rollout_windows,
    summarize_values,
)
from .diagnose_model import sha256_file
from .ensemble import (
    DISAGREEMENT_NAMES,
    WorldModelEnsemble,
    load_ensemble,
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


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _is_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(horizons)
    if not values:
        raise ValueError("horizons must be non-empty")
    if any(not _is_integer(value) or int(value) <= 0 for value in values):
        raise ValueError("horizons must contain positive integers")
    normalized = tuple(int(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("horizons must be unique")
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError("horizons must be strictly increasing")
    return normalized


def _validate_positive_integer(name: str, value: Any) -> int:
    if not _is_integer(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_dataset_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    validated = {name: np.asarray(values) for name, values in arrays.items()}
    states = validated["states"]
    actions = validated["actions"]
    next_states = validated["next_states"]
    episode_ids = validated["episode_ids"]
    step_ids = validated["step_ids"]

    if states.ndim != 2 or states.shape[1:] != (4,):
        raise ValueError("states must have shape [N, 4]")
    count = states.shape[0]
    if count == 0:
        raise ValueError("dataset arrays must not be empty")
    if actions.shape != (count, 2):
        raise ValueError("actions must have shape [N, 2]")
    if next_states.shape != (count, 4):
        raise ValueError("next_states must have shape [N, 4]")
    if episode_ids.shape != (count,):
        raise ValueError("episode_ids must have shape [N]")
    if step_ids.shape != (count,):
        raise ValueError("step_ids must have shape [N]")

    for name, values in validated.items():
        try:
            finite = bool(np.all(np.isfinite(values)))
        except TypeError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must contain only finite values")

    for name, values in (("episode_ids", episode_ids), ("step_ids", step_ids)):
        if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
            values.dtype,
            np.integer,
        ):
            raise ValueError(f"{name} must use a non-boolean integer dtype")
        if np.any(values < 0):
            raise ValueError(f"{name} must contain only non-negative values")
    return validated


def _rollout_horizon_snapshots(
    rollout: Mapping[str, Any],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    metric_records = _required_mapping(rollout, "metrics", parent="rollout")
    snapshots = {}
    curve_fields = (
        "ensemble_error_mean",
        "mean_member_error_mean",
        "min_member_error_mean",
        "max_member_error_mean",
        "disagreement_mean",
        "pearson_correlation",
    )
    for horizon in horizons:
        index = horizon - 1
        snapshots[str(horizon)] = {
            "episodes": int(rollout["episodes"]),
            "windows": int(rollout["windows"]),
            "metrics": {
                name: {
                    field: _required_mapping(
                        metric_records,
                        name,
                        parent="rollout.metrics",
                    )[field][index]
                    for field in curve_fields
                }
                for name in METRIC_NAMES
            },
        }
    return snapshots


def write_calibration_csv(
    metrics: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    """Write one-step calibration bins in a stable tidy CSV layout."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "metric",
        "bin_index",
        "count",
        "disagreement_mean",
        "error_mean",
    )
    metric_records = _required_mapping(metrics, "metrics", parent="one-step")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in METRIC_NAMES:
            record = _required_mapping(metric_records, name, parent="one-step.metrics")
            if "calibration_bins" not in record:
                raise ValueError(f"{name} is missing calibration_bins")
            bins = record["calibration_bins"]
            if not isinstance(bins, (list, tuple)) or not bins:
                raise ValueError(f"{name}.calibration_bins must be a non-empty list")
            for index, item in enumerate(bins):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"{name}.calibration_bins[{index}] must be a mapping"
                    )
                writer.writerow(
                    {
                        "metric": name,
                        "bin_index": index,
                        "count": item["count"],
                        "disagreement_mean": item["disagreement_mean"],
                        "error_mean": item["error_mean"],
                    }
                )
    return output


def run_ensemble_diagnostics(
    *,
    data_path: Path | str,
    checkpoint_paths: Iterable[Path | str],
    output_dir: Path | str,
    horizons: Iterable[int] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate compatible checkpoints and write one atomic artifact bundle."""

    horizon_values = _validate_horizons(horizons)
    window_count = _validate_positive_integer(
        "windows_per_episode",
        windows_per_episode,
    )
    bin_count = _validate_positive_integer("calibration_bins", calibration_bins)
    data = Path(data_path)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")
    if not data.is_file():
        raise FileNotFoundError(f"dataset is not a regular file: {data}")
    required_arrays = {
        "states",
        "actions",
        "next_states",
        "episode_ids",
        "step_ids",
    }
    with np.load(data, allow_pickle=False) as loaded:
        missing = required_arrays - set(loaded.files)
        if missing:
            raise ValueError(
                f"dataset is missing arrays: {', '.join(sorted(missing))}"
            )
        arrays = {name: loaded[name] for name in required_arrays}
    arrays = _validate_dataset_arrays(arrays)

    ensemble = load_ensemble(tuple(checkpoint_paths))
    test_ids = np.asarray(ensemble.members[0].split_episode_ids["test"])
    available_ids = np.unique(np.asarray(arrays["episode_ids"]))
    missing_test_ids = [
        int(value)
        for value in test_ids.tolist()
        if not np.any(available_ids == value)
    ]
    if missing_test_ids:
        raise ValueError(
            "checkpoint test episode IDs are missing from the dataset: "
            + ", ".join(map(str, missing_test_ids))
        )

    test_mask = np.isin(arrays["episode_ids"], test_ids)
    one_step = evaluate_one_step_calibration(
        ensemble,
        states=arrays["states"][test_mask],
        actions=arrays["actions"][test_mask],
        true_next_states=arrays["next_states"][test_mask],
        calibration_bins=bin_count,
    )
    selection = select_rollout_windows(
        states=arrays["states"],
        actions=arrays["actions"],
        next_states=arrays["next_states"],
        episode_ids=arrays["episode_ids"],
        step_ids=arrays["step_ids"],
        selected_episode_ids=test_ids,
        max_horizon=horizon_values[-1],
        windows_per_episode=window_count,
    )
    rollout = evaluate_ensemble_rollouts(
        ensemble,
        windows=selection.windows,
        eligible_episode_ids=selection.eligible_episode_ids,
    )
    rollout["horizons"] = _rollout_horizon_snapshots(rollout, horizon_values)
    metrics = {
        "schema_version": 1,
        "one_step": one_step,
        "rollout": rollout,
    }

    reference_config = ensemble.members[0].training_config
    checkpoints = [
        {
            "seed": int(seed),
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for seed, path in zip(ensemble.seeds, ensemble.checkpoint_paths)
    ]
    manifest = {
        "schema_version": 1,
        "dataset": {
            "path": str(data.resolve()),
            "sha256": sha256_file(data),
        },
        "checkpoints": checkpoints,
        "member_seeds": [int(seed) for seed in ensemble.seeds],
        "training_config": {
            "split_seed": int(reference_config["split_seed"]),
            "rollout_horizon": int(reference_config["rollout_horizon"]),
            "rollout_loss_weight": float(
                reference_config["rollout_loss_weight"]
            ),
            "hidden_size": int(reference_config["hidden_size"]),
        },
        "test_episode_ids": sorted(int(value) for value in test_ids.tolist()),
        "diagnostics": {
            "horizons": list(horizon_values),
            "windows_per_episode": window_count,
            "calibration_bins": bin_count,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    try:
        metrics_path = staging / "metrics.json"
        calibration_csv_path = staging / "one_step_calibration.csv"
        calibration_plot_path = staging / "one_step_calibration.png"
        rollout_plot_path = staging / "rollout_uncertainty.png"
        manifest_path = staging / "manifest.json"

        _write_json(metrics_path, metrics)
        write_calibration_csv(one_step, calibration_csv_path)
        plot_one_step_calibration(one_step, calibration_plot_path)
        plot_rollout_uncertainty(rollout, rollout_plot_path)
        _write_json(manifest_path, manifest)

        if output.exists():
            output.rmdir()
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "output_dir": str(output),
        "manifest": str(output / "manifest.json"),
        "metrics": str(output / "metrics.json"),
        "calibration_csv": str(output / "one_step_calibration.csv"),
        "calibration_plot": str(output / "one_step_calibration.png"),
        "rollout_plot": str(output / "rollout_uncertainty.png"),
        "member_seeds": [int(seed) for seed in ensemble.seeds],
        "horizons": list(horizon_values),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/diagnostics/h10-ensemble"),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50],
    )
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()
    try:
        result = run_ensemble_diagnostics(
            data_path=args.data,
            checkpoint_paths=args.checkpoints,
            output_dir=args.output_dir,
            horizons=args.horizons,
            windows_per_episode=args.windows_per_episode,
            calibration_bins=args.calibration_bins,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
