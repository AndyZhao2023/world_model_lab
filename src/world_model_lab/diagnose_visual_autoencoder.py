"""Compare spatial autoencoders on object and oracle-rollout reconstruction."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
import math
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

from .diagnose_model import sha256_file
from .diagnose_visual_rollout import select_visual_rollout_windows
from .train_visual_latent_model import (
    LoadedVisualLatentModel,
    evaluate_autoencoder,
    load_visual_latent_checkpoint,
)
from .visual_dataset import load_visual_dataset
from .visual_latent_data import (
    encode_all_frames,
    frames_to_tensor,
    renderer_object_masks,
)
from .visual_latent_model import SpatialConvAutoencoder


def _episode_equal_mean(
    values: np.ndarray,
    episode_ids: np.ndarray,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episode_ids, dtype=np.int64)
    if array.ndim != 1 or episodes.shape != array.shape or array.size == 0:
        raise ValueError("metric values and episode_ids must align")
    if not np.all(np.isfinite(array)):
        raise ValueError("metric values must be finite")
    episode_means = [
        float(np.mean(array[episodes == episode_id]))
        for episode_id in np.unique(episodes)
    ]
    return float(np.mean(episode_means))


def summarize_autoencoder_reconstructions(
    *,
    reconstructed_frames: np.ndarray,
    true_initial_frames: np.ndarray,
    true_target_frames: np.ndarray,
    object_masks: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize oracle reconstructions with episode-equal aggregation."""

    reconstructed = np.asarray(reconstructed_frames, dtype=np.float64)
    initial = np.asarray(true_initial_frames)
    targets = np.asarray(true_target_frames)
    masks = np.asarray(object_masks)
    episodes = np.asarray(episode_ids)
    if reconstructed.ndim != 5 or reconstructed.shape[2] != 3:
        raise ValueError(
            "reconstructed_frames must have shape [N, H, 3, Y, X]"
        )
    count, horizon, _, height, width = reconstructed.shape
    if initial.shape != (count, 3, height, width):
        raise ValueError("true_initial_frames must have shape [N, 3, Y, X]")
    if targets.shape != reconstructed.shape:
        raise ValueError(
            "true_target_frames must align with reconstructed_frames"
        )
    if masks.shape != (count, horizon, 1, height, width):
        raise ValueError("object_masks must have shape [N, H, 1, Y, X]")
    if (
        episodes.shape != (count,)
        or episodes.dtype.kind not in "iu"
        or np.unique(episodes).size == 0
    ):
        raise ValueError("episode_ids must be an integer vector of length N")
    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("reconstructed_frames must be finite")
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError("object_masks must be binary")

    initial_values = initial.astype(np.float64) / 255.0
    target_values = targets.astype(np.float64) / 255.0
    result: dict[str, dict[str, float | int]] = {}
    for step in range(horizon):
        errors = reconstructed[:, step] - target_values[:, step]
        squared = np.square(errors)
        absolute = np.abs(errors)
        expanded_object = np.broadcast_to(
            masks[:, step].astype(bool),
            errors.shape,
        )
        expanded_background = ~expanded_object
        object_counts = np.sum(expanded_object, axis=(1, 2, 3))
        background_counts = np.sum(expanded_background, axis=(1, 2, 3))
        if np.any(object_counts == 0) or np.any(background_counts == 0):
            raise ValueError(
                "every oracle window requires object and background pixels"
            )
        cumulative_changed = np.any(
            targets[:, step] != initial,
            axis=1,
            keepdims=True,
        )
        expanded_changed = np.broadcast_to(
            cumulative_changed,
            errors.shape,
        )
        changed_counts = np.sum(expanded_changed, axis=(1, 2, 3))
        changed_absolute = np.sum(
            absolute * expanded_changed,
            axis=(1, 2, 3),
        )
        changed_mae = np.divide(
            changed_absolute,
            changed_counts,
            out=np.zeros_like(changed_absolute),
            where=changed_counts > 0,
        )
        per_window = {
            "pixel_mse": np.mean(squared, axis=(1, 2, 3)),
            "pixel_mae": np.mean(absolute, axis=(1, 2, 3)),
            "object_pixel_mse": (
                np.sum(squared * expanded_object, axis=(1, 2, 3))
                / object_counts
            ),
            "object_pixel_mae": (
                np.sum(absolute * expanded_object, axis=(1, 2, 3))
                / object_counts
            ),
            "background_pixel_mse": (
                np.sum(squared * expanded_background, axis=(1, 2, 3))
                / background_counts
            ),
            "background_pixel_mae": (
                np.sum(absolute * expanded_background, axis=(1, 2, 3))
                / background_counts
            ),
            "cumulative_changed_pixel_mae": changed_mae,
        }
        result[str(step + 1)] = {
            "episodes": int(np.unique(episodes).size),
            "windows": count,
            **{
                name: _episode_equal_mean(values, episodes)
                for name, values in per_window.items()
            },
        }
    return result


def _build_representation_decision(
    *,
    source_static: Mapping[str, float | int],
    candidate_static: Mapping[str, float | int],
    source_steps: Mapping[str, Mapping[str, float | int]],
    candidate_steps: Mapping[str, Mapping[str, float | int]],
    decision_horizon: int,
) -> dict[str, object]:
    """Apply the four pre-registered representation gates."""

    step = str(decision_horizon)
    if step not in source_steps or step not in candidate_steps:
        raise ValueError("decision_horizon is missing from rollout metrics")
    gate_specs = (
        (
            "held_out_object_mse_improvement",
            "object_pixel_mse",
            "<",
            float(source_static["object_pixel_mse"]),
            float(candidate_static["object_pixel_mse"]),
        ),
        (
            "held_out_full_frame_mse_stability",
            "pixel_mse",
            "<=",
            1.10 * float(source_static["pixel_mse"]),
            float(candidate_static["pixel_mse"]),
        ),
        (
            "held_out_background_mse_stability",
            "background_pixel_mse",
            "<=",
            1.10 * float(source_static["background_pixel_mse"]),
            float(candidate_static["background_pixel_mse"]),
        ),
        (
            "horizon_cumulative_changed_pixel_mae_improvement",
            "cumulative_changed_pixel_mae",
            "<",
            float(
                source_steps[step]["cumulative_changed_pixel_mae"]
            ),
            float(
                candidate_steps[step]["cumulative_changed_pixel_mae"]
            ),
        ),
    )
    gates = []
    for name, metric, operator, limit, candidate in gate_specs:
        if not all(math.isfinite(value) for value in (limit, candidate)):
            raise ValueError("representation gate values must be finite")
        passed = candidate < limit if operator == "<" else candidate <= limit
        gates.append(
            {
                "name": name,
                "metric": metric,
                "operator": operator,
                "limit": limit,
                "candidate": candidate,
                "passed": bool(passed),
            }
        )
    return {
        "decision_horizon": int(decision_horizon),
        "passed": all(bool(gate["passed"]) for gate in gates),
        "gates": gates,
    }


def _validate_protocol(
    *,
    horizons: Iterable[int],
    windows_per_episode: int,
    decision_horizon: int,
    batch_size: int,
) -> tuple[int, ...]:
    raw = tuple(horizons)
    if (
        not raw
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in raw
        )
    ):
        raise ValueError("horizons must be positive integers")
    values = tuple(int(value) for value in raw)
    if len(set(values)) != len(values) or tuple(sorted(values)) != values:
        raise ValueError("horizons must be unique and strictly increasing")
    if (
        isinstance(windows_per_episode, (bool, np.bool_))
        or not isinstance(windows_per_episode, (int, np.integer))
        or int(windows_per_episode) <= 0
    ):
        raise ValueError("windows_per_episode must be positive")
    if (
        isinstance(decision_horizon, (bool, np.bool_))
        or not isinstance(decision_horizon, (int, np.integer))
        or int(decision_horizon) not in values
    ):
        raise ValueError("decision_horizon must be listed in horizons")
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or int(batch_size) <= 0
    ):
        raise ValueError("batch_size must be positive")
    return values


def _validate_checkpoint_pair(
    *,
    source: LoadedVisualLatentModel,
    candidate: LoadedVisualLatentModel,
    dataset_sha256: str,
) -> None:
    for name, model in (("source", source), ("candidate", candidate)):
        if not isinstance(model.autoencoder, SpatialConvAutoencoder):
            raise ValueError(
                f"{name} checkpoint must use a spatial autoencoder"
            )
        if model.dataset_metadata.get("sha256") != dataset_sha256:
            raise ValueError(
                f"{name} dataset SHA-256 does not match the input dataset"
            )
    if not np.array_equal(
        source.split_episode_ids["test"],
        candidate.split_episode_ids["test"],
    ):
        raise ValueError("checkpoint test split episode IDs do not match")
    if (
        source.autoencoder.latent_channels
        != candidate.autoencoder.latent_channels
        or source.autoencoder.base_channels
        != candidate.autoencoder.base_channels
    ):
        raise ValueError("checkpoint spatial autoencoder dimensions do not match")


def _reconstruct_all_frames(
    model: SpatialConvAutoencoder,
    frames: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(frames)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            images = frames_to_tensor(values[start : start + batch_size])
            reconstructed = model(images)
            if not torch.all(torch.isfinite(reconstructed)):
                raise ValueError("autoencoder produced non-finite pixels")
            batches.append(reconstructed.cpu().numpy())
    return np.concatenate(batches, axis=0)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _plot_comparison(
    *,
    metrics: Mapping[str, object],
    output_path: Path,
) -> Path:
    models = metrics["models"]
    if not isinstance(models, Mapping):
        raise ValueError("metrics models record is malformed")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    colors = {"source": "#4C78A8", "candidate": "#F58518"}
    plots = (
        ("pixel_mse", "Oracle full-frame reconstruction", "MSE"),
        ("object_pixel_mse", "Oracle object reconstruction", "MSE"),
        ("background_pixel_mse", "Oracle background reconstruction", "MSE"),
        (
            "cumulative_changed_pixel_mae",
            "Oracle cumulative changed pixels",
            "MAE",
        ),
    )
    for axis, (metric, title, ylabel) in zip(axes.flat, plots):
        for name in ("source", "candidate"):
            record = models[name]
            if not isinstance(record, Mapping):
                raise ValueError("model record is malformed")
            steps = record["steps"]
            if not isinstance(steps, Mapping):
                raise ValueError("model steps record is malformed")
            x = np.asarray([int(step) for step in steps], dtype=np.int64)
            y = np.asarray(
                [float(steps[str(step)][metric]) for step in x],
                dtype=np.float64,
            )
            axis.plot(x, y, color=colors[name], label=name)
        axis.set_title(title)
        axis.set_xlabel("rollout step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Spatial autoencoder oracle diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def run_visual_autoencoder_diagnostics(
    *,
    data_path: Path | str,
    source_checkpoint_path: Path | str,
    candidate_checkpoint_path: Path | str,
    output_dir: Path | str,
    horizons: Iterable[int] = (1, 5, 10),
    windows_per_episode: int = 8,
    decision_horizon: int = 5,
    batch_size: int = 256,
) -> dict[str, object]:
    """Publish matched source/candidate oracle reconstruction diagnostics."""

    horizon_values = _validate_protocol(
        horizons=horizons,
        windows_per_episode=windows_per_episode,
        decision_horizon=decision_horizon,
        batch_size=batch_size,
    )
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")
    data = Path(data_path)
    source_path = Path(source_checkpoint_path)
    candidate_path = Path(candidate_checkpoint_path)
    dataset = load_visual_dataset(data)
    dataset_sha = sha256_file(data)
    source = load_visual_latent_checkpoint(source_path)
    candidate = load_visual_latent_checkpoint(candidate_path)
    _validate_checkpoint_pair(
        source=source,
        candidate=candidate,
        dataset_sha256=dataset_sha,
    )
    test_ids = source.split_episode_ids["test"]
    available_ids = set(
        int(value) for value in dataset["episode_ids"].tolist()
    )
    missing = sorted(
        set(int(value) for value in test_ids.tolist()) - available_ids
    )
    if missing:
        raise ValueError(
            "checkpoint test episode IDs are missing from the dataset: "
            + ", ".join(map(str, missing))
        )
    source_latents = encode_all_frames(
        source.autoencoder,
        dataset["frames"],
        batch_size=batch_size,
    )
    selection = select_visual_rollout_windows(
        dataset=dataset,
        latent_frames=source_latents,
        selected_episode_ids=test_ids,
        max_horizon=horizon_values[-1],
        windows_per_episode=windows_per_episode,
    )
    episode_ids = np.asarray(
        [window.episode_id for window in selection.windows],
        dtype=np.int64,
    )
    initial_indices = np.asarray(
        [window.initial_frame_index for window in selection.windows],
        dtype=np.int64,
    )
    target_indices = np.stack(
        [window.target_frame_indices for window in selection.windows]
    )
    frames = np.asarray(dataset["frames"])
    initial_frames = np.transpose(
        frames[initial_indices],
        (0, 3, 1, 2),
    )
    target_hwc = frames[target_indices]
    target_frames = np.transpose(target_hwc, (0, 1, 4, 2, 3))
    object_masks = (
        renderer_object_masks(
            target_hwc.reshape(-1, target_hwc.shape[2], target_hwc.shape[3], 3)
        )
        .numpy()
        .reshape(
            target_hwc.shape[0],
            target_hwc.shape[1],
            1,
            target_hwc.shape[2],
            target_hwc.shape[3],
        )
    )

    records: dict[str, dict[str, object]] = {}
    for name, loaded in (("source", source), ("candidate", candidate)):
        assert isinstance(loaded.autoencoder, SpatialConvAutoencoder)
        reconstructed = _reconstruct_all_frames(
            loaded.autoencoder,
            frames,
            batch_size=batch_size,
        )
        reconstructed_targets = reconstructed[
            target_indices.reshape(-1)
        ].reshape(
            target_indices.shape[0],
            target_indices.shape[1],
            3,
            frames.shape[1],
            frames.shape[2],
        )
        records[name] = {
            "static": evaluate_autoencoder(
                loaded.autoencoder,
                dataset,
                selected_episode_ids=test_ids,
                batch_size=batch_size,
            ),
            "steps": summarize_autoencoder_reconstructions(
                reconstructed_frames=reconstructed_targets,
                true_initial_frames=initial_frames,
                true_target_frames=target_frames,
                object_masks=object_masks,
                episode_ids=episode_ids,
            ),
        }
    decision = _build_representation_decision(
        source_static=records["source"]["static"],
        candidate_static=records["candidate"]["static"],
        source_steps=records["source"]["steps"],
        candidate_steps=records["candidate"]["steps"],
        decision_horizon=decision_horizon,
    )
    metrics: dict[str, object] = {
        "schema_version": 1,
        "models": records,
        "decision": decision,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "path": str(data.resolve()),
            "sha256": dataset_sha,
        },
        "checkpoints": {
            "source": {
                "path": str(source_path.resolve()),
                "sha256": sha256_file(source_path),
            },
            "candidate": {
                "path": str(candidate_path.resolve()),
                "sha256": sha256_file(candidate_path),
            },
        },
        "protocol": {
            "snapshot_horizons": list(horizon_values),
            "max_horizon": horizon_values[-1],
            "decision_horizon": int(decision_horizon),
            "windows_per_episode": int(windows_per_episode),
            "aggregation": "windows-within-episode-then-episodes-equally",
            "object_mask": "exact-renderer-car-or-heading-colour",
            "latent_comparison": "not-used-across-representations",
        },
        "test_episode_ids": [int(value) for value in test_ids.tolist()],
        "eligible_episode_ids": [
            int(value)
            for value in selection.eligible_episode_ids.tolist()
        ],
        "skipped_episode_ids": [
            int(value)
            for value in selection.skipped_episode_ids.tolist()
        ],
        "windows": [
            {
                "episode_id": window.episode_id,
                "start_step": window.start_step,
            }
            for window in selection.windows
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    try:
        _write_json(staging / "metrics.json", metrics)
        _write_json(staging / "manifest.json", manifest)
        _plot_comparison(
            metrics=metrics,
            output_path=staging / "visual_autoencoder_comparison.png",
        )
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
        "plot": str(output / "visual_autoencoder_comparison.png"),
        "horizons": list(horizon_values),
        "windows": len(selection.windows),
        "eligible_episodes": int(selection.eligible_episode_ids.size),
        "decision": decision,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 5, 10],
    )
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--decision-horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        summary = run_visual_autoencoder_diagnostics(
            data_path=args.data,
            source_checkpoint_path=args.source_checkpoint,
            candidate_checkpoint_path=args.candidate_checkpoint,
            output_dir=args.output_dir,
            horizons=args.horizons,
            windows_per_episode=args.windows_per_episode,
            decision_horizon=args.decision_horizon,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
