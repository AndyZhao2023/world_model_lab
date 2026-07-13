"""Generate a reproducible diagnostic bundle for a learned world model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .diagnostic_plots import plot_diagnostic_overview, plot_rollout_errors
from .diagnostics import build_diagnostic_metrics
from .train_world_model import load_checkpoint


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_diagnostics(
    *,
    data_path: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    horizons: Iterable[int] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    xy_bins: int = 12,
    feature_bins: int = 8,
    min_bin_count: int = 5,
) -> dict[str, Any]:
    """Evaluate a checkpoint and write metrics, manifest, and PNG reports."""

    data = Path(data_path)
    checkpoint = Path(checkpoint_path)
    output = Path(output_dir)
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

    horizon_values = tuple(int(value) for value in horizons)
    world_model = load_checkpoint(checkpoint)
    metrics = build_diagnostic_metrics(
        world_model,
        arrays=arrays,
        split_episode_ids=world_model.split_episode_ids,
        horizons=horizon_values,
        windows_per_episode=windows_per_episode,
        xy_bins=xy_bins,
        feature_bins=feature_bins,
        min_bin_count=min_bin_count,
    )
    manifest = {
        "schema_version": 1,
        "dataset": {
            "path": str(data.resolve()),
            "sha256": sha256_file(data),
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "hidden_size": world_model.model.hidden_size,
            "training_config": world_model.training_config,
            "test_episode_ids": [
                int(value)
                for value in world_model.split_episode_ids["test"].tolist()
            ],
        },
        "diagnostics": {
            "horizons": list(horizon_values),
            "windows_per_episode": windows_per_episode,
            "xy_bins": xy_bins,
            "feature_bins": feature_bins,
            "min_bin_count": min_bin_count,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    manifest_path = output / "manifest.json"
    overview_path = output / "overview.png"
    rollout_path = output / "rollout_errors.png"
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    plot_diagnostic_overview(metrics, overview_path)
    plot_rollout_errors(metrics, rollout_path)

    longest_horizon = horizon_values[-1]
    longest_metrics = metrics["rollout"]["horizons"][str(longest_horizon)]
    return {
        "output_dir": str(output),
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
        "overview_plot": str(overview_path),
        "rollout_plot": str(rollout_path),
        "one_step": {
            name: metrics["one_step"]["overall"][name]["mean"]
            for name in ("position", "heading_degrees", "velocity")
        },
        "longest_horizon": longest_horizon,
        "teacher_forcing": {
            name: longest_metrics["teacher_forcing"][name]["mean"]
            for name in ("position", "heading_degrees", "velocity")
        },
        "free_rollout": {
            name: longest_metrics["free_rollout"][name]["mean"]
            for name in ("position", "heading_degrees", "velocity")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/world_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/diagnostics/baseline"),
    )
    parser.add_argument(
        "--horizons",
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
        summary = run_diagnostics(
            data_path=args.data,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            horizons=args.horizons,
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
