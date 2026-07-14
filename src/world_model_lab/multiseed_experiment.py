"""Aggregate and plot paired H1/H10 multi-seed diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .diagnose_model import run_diagnostics, sha256_file
from .train_world_model import LoadedWorldModel, load_checkpoint, run_training


METRIC_NAMES = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)
COMPARISON_TOLERANCE = 1e-12


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _validate_split_invariant(
    checkpoints: Iterable[LoadedWorldModel],
) -> None:
    loaded = list(checkpoints)
    if not loaded:
        raise ValueError("at least one checkpoint is required")
    reference = loaded[0].split_episode_ids
    for checkpoint in loaded[1:]:
        for name in ("train", "validation", "test"):
            if name not in reference or name not in checkpoint.split_episode_ids:
                raise ValueError("checkpoint split episode IDs are incomplete")
            if not np.array_equal(
                reference[name], checkpoint.split_episode_ids[name]
            ):
                raise ValueError("checkpoint split episode IDs differ")


def _validate_dataset_hash_invariant(
    manifests: Iterable[dict[str, Any]],
) -> str:
    loaded = list(manifests)
    if not loaded:
        raise ValueError("at least one diagnostics manifest is required")
    try:
        hashes = [manifest["dataset"]["sha256"] for manifest in loaded]
    except (KeyError, TypeError) as error:
        raise ValueError("diagnostics manifest is missing dataset SHA-256") from error
    if any(value != hashes[0] for value in hashes[1:]):
        raise ValueError("diagnostics dataset SHA-256 values differ")
    return str(hashes[0])


def _is_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _validate_positive_integer(name: str, value: Any) -> None:
    if not _is_integer(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")


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


def run_multiseed_experiment(
    *,
    data_path: Path | str,
    output_dir: Path | str,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    split_seed: int = 0,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    rollout_loss_weight: float = 1.0,
    diagnostic_horizons: Iterable[int] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    xy_bins: int = 12,
    feature_bins: int = 8,
    min_bin_count: int = 5,
) -> dict[str, Any]:
    """Train paired H1/H10 models and aggregate held-out diagnostics."""

    data = Path(data_path)
    output = Path(output_dir)
    seed_values = tuple(seeds)
    horizon_values = tuple(diagnostic_horizons)

    if not data.is_file():
        raise FileNotFoundError(f"dataset is not a regular file: {data}")
    if len(seed_values) < 2:
        raise ValueError("at least two training seeds are required")
    if any(not _is_integer(seed) or int(seed) < 0 for seed in seed_values):
        raise ValueError("training seeds must be non-negative integers")
    if len(set(map(int, seed_values))) != len(seed_values):
        raise ValueError("training seeds must be unique")
    if not _is_integer(split_seed) or int(split_seed) < 0:
        raise ValueError("split_seed must be a non-negative integer")
    for name, value in (
        ("hidden_size", hidden_size),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("windows_per_episode", windows_per_episode),
        ("xy_bins", xy_bins),
        ("feature_bins", feature_bins),
        ("min_bin_count", min_bin_count),
    ):
        _validate_positive_integer(name, value)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if (
        not math.isfinite(rollout_loss_weight)
        or rollout_loss_weight < 0.0
    ):
        raise ValueError(
            "rollout_loss_weight must be finite and non-negative"
        )
    if not horizon_values:
        raise ValueError("diagnostic_horizons must be non-empty")
    if any(
        not _is_integer(horizon) or int(horizon) <= 0
        for horizon in horizon_values
    ):
        raise ValueError(
            "diagnostic_horizons must contain positive integers"
        )
    if len(set(map(int, horizon_values))) != len(horizon_values):
        raise ValueError("diagnostic_horizons must be unique")
    if any(
        int(left) >= int(right)
        for left, right in zip(horizon_values, horizon_values[1:])
    ):
        raise ValueError("diagnostic_horizons must be strictly increasing")
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise ValueError("output directory must be absent or empty")

    normalized_seeds = sorted(int(seed) for seed in seed_values)
    normalized_horizons = [int(horizon) for horizon in horizon_values]
    normalized_split_seed = int(split_seed)
    dataset_hash = sha256_file(data)
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    checkpoints: list[LoadedWorldModel] = []
    diagnostics_manifests: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        paired_metrics: dict[str, dict[str, Any]] = {}
        paired_metrics_paths: dict[str, str] = {}
        for model_name, rollout_horizon, effective_rollout_weight in (
            ("h1", 1, 0.0),
            ("h10", 10, rollout_loss_weight),
        ):
            model_dir = output / "runs" / f"seed_{seed}" / model_name
            checkpoint_path = model_dir / "world_model.pt"
            diagnostics_dir = model_dir / "diagnostics"
            metrics_path = diagnostics_dir / "metrics.json"
            diagnostics_manifest_path = diagnostics_dir / "manifest.json"
            run_training(
                data_path=data,
                output_path=checkpoint_path,
                hidden_size=hidden_size,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                split_seed=normalized_split_seed,
                rollout_horizon=rollout_horizon,
                rollout_loss_weight=effective_rollout_weight,
            )
            run_diagnostics(
                data_path=data,
                checkpoint_path=checkpoint_path,
                output_dir=diagnostics_dir,
                horizons=normalized_horizons,
                windows_per_episode=windows_per_episode,
                xy_bins=xy_bins,
                feature_bins=feature_bins,
                min_bin_count=min_bin_count,
            )

            checkpoints.append(load_checkpoint(checkpoint_path))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            diagnostics_manifest = json.loads(
                diagnostics_manifest_path.read_text(encoding="utf-8")
            )
            diagnostics_manifests.append(diagnostics_manifest)
            relative_checkpoint = checkpoint_path.relative_to(output).as_posix()
            relative_metrics = metrics_path.relative_to(output).as_posix()
            relative_manifest = (
                diagnostics_manifest_path.relative_to(output).as_posix()
            )
            paired_metrics[model_name] = metrics
            paired_metrics_paths[model_name] = relative_metrics
            manifest_runs.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "checkpoint": relative_checkpoint,
                    "metrics": relative_metrics,
                    "manifest": relative_manifest,
                }
            )
        records.append(
            {
                "seed": seed,
                "h1_metrics": paired_metrics["h1"],
                "h10_metrics": paired_metrics["h10"],
                "h1_metrics_path": paired_metrics_paths["h1"],
                "h10_metrics_path": paired_metrics_paths["h10"],
            }
        )

    _validate_split_invariant(checkpoints)
    diagnostics_dataset_hash = _validate_dataset_hash_invariant(
        diagnostics_manifests
    )
    if diagnostics_dataset_hash != dataset_hash:
        raise ValueError("diagnostics dataset SHA-256 does not match input dataset")

    summary = build_experiment_summary(
        records,
        snapshot_horizons=normalized_horizons,
        split_seed=normalized_split_seed,
    )
    manifest = {
        "schema_version": 1,
        "dataset": {
            "path": str(data.resolve()),
            "sha256": dataset_hash,
        },
        "training": {
            "seeds": normalized_seeds,
            "split_seed": normalized_split_seed,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "models": {
                "h1": {
                    "rollout_horizon": 1,
                    "rollout_loss_weight": 0.0,
                },
                "h10": {
                    "rollout_horizon": 10,
                    "rollout_loss_weight": rollout_loss_weight,
                },
            },
        },
        "diagnostics": {
            "horizons": normalized_horizons,
            "windows_per_episode": windows_per_episode,
            "xy_bins": xy_bins,
            "feature_bins": feature_bins,
            "min_bin_count": min_bin_count,
        },
        "runs": manifest_runs,
    }

    manifest_path = output / "experiment_manifest.json"
    summary_path = output / "summary.json"
    csv_path = output / "summary.csv"
    plot_path = output / "multiseed_comparison.png"
    _write_json(summary_path, summary)
    write_summary_csv(summary, csv_path)
    plot_multiseed_comparison(summary, plot_path)
    _write_json(manifest_path, manifest)

    longest_horizon = normalized_horizons[-1]
    longest_snapshot = summary["sparse_horizons"][str(longest_horizon)]
    longest_horizon_metrics = {
        name: {
            "h1_mean": longest_snapshot[name]["h1"]["mean"],
            "h10_mean": longest_snapshot[name]["h10"]["mean"],
            "paired_delta_mean": longest_snapshot[name]["paired_delta"]["mean"],
            "improved_seed_count": longest_snapshot[name]["paired_delta"][
                "improved_seed_count"
            ],
            "worse_seed_count": longest_snapshot[name]["paired_delta"][
                "worse_seed_count"
            ],
        }
        for name in METRIC_NAMES
    }
    return {
        "output_dir": str(output),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "seeds": normalized_seeds,
        "split_seed": normalized_split_seed,
        "longest_horizon": longest_horizon,
        "longest_horizon_metrics": longest_horizon_metrics,
        "first_heading_regression_steps": {
            str(seed): summary["per_seed"][str(seed)][
                "first_heading_regression_step"
            ]
            for seed in normalized_seeds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/h1-vs-h10-seeds-0-4"),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--diagnostic-horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50],
    )
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--xy-bins", type=int, default=12)
    parser.add_argument("--feature-bins", type=int, default=8)
    parser.add_argument("--min-bin-count", type=int, default=5)
    args = parser.parse_args()

    try:
        summary = run_multiseed_experiment(
            data_path=args.data,
            output_dir=args.output_dir,
            seeds=args.seeds,
            split_seed=args.split_seed,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            rollout_loss_weight=args.rollout_loss_weight,
            diagnostic_horizons=args.diagnostic_horizons,
            windows_per_episode=args.windows_per_episode,
            xy_bins=args.xy_bins,
            feature_bins=args.feature_bins,
            min_bin_count=args.min_bin_count,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
