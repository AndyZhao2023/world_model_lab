"""Run a frozen V-JEPA 2 state-readability and temporal-order probe."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from ._artifact_io import write_new_file_atomically
from .dataset import split_episode_ids
from .diagnose_model import sha256_file
from .visual_dataset import load_visual_dataset
from .visual_windows import build_visual_window_index
from .vjepa_probe import (
    DEFAULT_VJEPA_MODEL_ID,
    FrozenVJEPAEncoder,
    build_probe_clip_batch,
    fit_linear_state_probe,
    mean_target_predictions,
    representation_probe_gates,
    select_evenly_spaced_positions,
    state_probe_metrics,
    state_to_probe_targets,
)


FEATURE_SCHEMA_VERSION = 1


class ProbeEncoder(Protocol):
    metadata: Mapping[str, str | int]

    def encode(self, clips: np.ndarray) -> np.ndarray: ...


EncoderFactory = Callable[..., ProbeEncoder]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _preflight_paths(feature_path: Path, result_path: Path) -> None:
    if feature_path.absolute() == result_path.absolute():
        raise ValueError("feature_path and result_path must be distinct")
    for path, label in (
        (feature_path, "feature cache"),
        (result_path, "result"),
    ):
        if path.exists():
            raise FileExistsError(f"{label} already exists: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"{label} parent directory does not exist: {path.parent}"
            )


def _encode_in_batches(
    encoder: ProbeEncoder,
    frames: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    batches: list[np.ndarray] = []
    feature_dim: int | None = None
    for start in range(0, int(frames.shape[0]), batch_size):
        features = np.asarray(
            encoder.encode(frames[start : start + batch_size]),
            dtype=np.float32,
        )
        expected_count = min(batch_size, int(frames.shape[0]) - start)
        if (
            features.ndim != 2
            or features.shape[0] != expected_count
            or features.shape[1] == 0
            or not np.all(np.isfinite(features))
        ):
            raise ValueError("encoder returned malformed or non-finite features")
        if feature_dim is None:
            feature_dim = int(features.shape[1])
        elif features.shape[1] != feature_dim:
            raise ValueError("encoder feature dimension changed between batches")
        batches.append(features.copy())
    if not batches:
        raise ValueError("feature extraction requires non-empty frames")
    return np.concatenate(batches), float(time.perf_counter() - started)


def _extract_variant(
    *,
    dataset: Mapping[str, np.ndarray],
    index: Any,
    positions: np.ndarray,
    order: str,
    encoder: ProbeEncoder,
    batch_size: int,
) -> tuple[Any, np.ndarray, float]:
    clips = build_probe_clip_batch(
        dataset,
        index,
        positions,
        order=order,
    )
    features, seconds = _encode_in_batches(
        encoder,
        clips.frames,
        batch_size=batch_size,
    )
    return clips, features, seconds


def _json_payload(summary: Mapping[str, Any]) -> bytes:
    return json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def run_probe(
    *,
    data_path: Path | str,
    feature_path: Path | str,
    result_path: Path | str,
    model_id: str = DEFAULT_VJEPA_MODEL_ID,
    model_revision: str = "main",
    device: str = "cpu",
    batch_size: int = 1,
    max_train: int = 128,
    max_validation: int = 32,
    max_test: int = 32,
    ridge: float = 1e-3,
    split_seed: int = 42,
    encoder_factory: EncoderFactory = FrozenVJEPAEncoder.from_pretrained,
) -> dict[str, Any]:
    """Extract frozen features, fit one ridge probe, and publish results."""

    feature_output = Path(feature_path)
    result_output = Path(result_path)
    _preflight_paths(feature_output, result_output)
    batch = _positive_int("batch_size", batch_size)
    limits = {
        "train": _positive_int("max_train", max_train),
        "validation": _positive_int("max_validation", max_validation),
        "test": _positive_int("max_test", max_test),
    }
    penalty = float(ridge)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if isinstance(split_seed, (bool, np.bool_)) or int(split_seed) < 0:
        raise ValueError("split_seed must be a non-negative integer")
    seed = int(split_seed)
    if not str(model_id).strip():
        raise ValueError("model_id must be non-empty")
    if not str(model_revision).strip():
        raise ValueError("model_revision must be non-empty")
    if not str(device).strip():
        raise ValueError("device must be non-empty")

    source = Path(data_path)
    dataset = load_visual_dataset(source)
    split_ids = split_episode_ids(
        np.asarray(dataset["episode_ids"], dtype=np.int64),
        seed=seed,
    )
    indexes = {
        name: build_visual_window_index(dataset, split_ids[name])
        for name in ("train", "validation", "test")
    }
    positions = {
        name: select_evenly_spaced_positions(
            indexes[name].count,
            limit=limits[name],
        )
        for name in ("train", "validation", "test")
    }
    encoder = encoder_factory(
        model_id=str(model_id),
        revision=str(model_revision),
        device=str(device),
    )
    metadata = dict(encoder.metadata)
    required_metadata = {
        "model_id",
        "requested_revision",
        "resolved_revision",
        "device",
        "tubelet_size",
        "crop_size",
        "patch_size",
        "hidden_size",
        "feature_dim",
        "pooling",
    }
    if set(metadata) != required_metadata:
        raise ValueError("encoder metadata does not match the probe contract")

    extracted: dict[str, tuple[Any, np.ndarray, float]] = {}
    for split_name in ("train", "validation", "test"):
        extracted[f"{split_name}_recorded"] = _extract_variant(
            dataset=dataset,
            index=indexes[split_name],
            positions=positions[split_name],
            order="recorded",
            encoder=encoder,
            batch_size=batch,
        )
    for order in ("reversed", "repeat_last"):
        extracted[f"test_{order}"] = _extract_variant(
            dataset=dataset,
            index=indexes["test"],
            positions=positions["test"],
            order=order,
            encoder=encoder,
            batch_size=batch,
        )

    train_clips, train_features, _ = extracted["train_recorded"]
    validation_clips, validation_features, _ = extracted[
        "validation_recorded"
    ]
    test_clips, test_features, _ = extracted["test_recorded"]
    _, reversed_features, _ = extracted["test_reversed"]
    _, repeated_features, _ = extracted["test_repeat_last"]
    train_targets = state_to_probe_targets(train_clips.states)
    validation_targets = state_to_probe_targets(validation_clips.states)
    test_targets = state_to_probe_targets(test_clips.states)
    probe = fit_linear_state_probe(
        train_features,
        train_targets,
        ridge=penalty,
    )
    world_bounds = np.asarray(dataset["scene_world_bounds"], dtype=np.float64)
    validation_metrics = state_probe_metrics(
        probe.predict(validation_features),
        validation_targets,
        world_bounds=world_bounds,
    )
    test_metrics = {
        "recorded": state_probe_metrics(
            probe.predict(test_features),
            test_targets,
            world_bounds=world_bounds,
        ),
        "reversed": state_probe_metrics(
            probe.predict(reversed_features),
            test_targets,
            world_bounds=world_bounds,
        ),
        "repeat_last": state_probe_metrics(
            probe.predict(repeated_features),
            test_targets,
            world_bounds=world_bounds,
        ),
    }
    baseline_metrics = state_probe_metrics(
        mean_target_predictions(train_targets, count=test_targets.shape[0]),
        test_targets,
        world_bounds=world_bounds,
    )
    gates = representation_probe_gates(
        recorded=test_metrics["recorded"],
        reversed_metrics=test_metrics["reversed"],
        repeat_last_metrics=test_metrics["repeat_last"],
    )
    timings = {
        name: {
            "seconds": seconds,
            "samples": int(clips.states.shape[0]),
            "seconds_per_clip": seconds / int(clips.states.shape[0]),
        }
        for name, (clips, _, seconds) in extracted.items()
    }
    summary: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "dataset": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "schema_version": int(dataset["schema_version"].item()),
            "renderer_version": str(dataset["renderer_version"].item()),
        },
        "encoder": metadata,
        "protocol": {
            "split_seed": seed,
            "batch_size": batch,
            "ridge": penalty,
            "context_frames": 4,
            "orders": ["recorded", "reversed", "repeat_last"],
        },
        "samples": {
            name: int(positions[name].size)
            for name in ("train", "validation", "test")
        },
        "sample_ids": {
            name: {
                "episode_ids": extracted[f"{name}_recorded"][0]
                .episode_ids.tolist(),
                "step_ids": extracted[f"{name}_recorded"][0]
                .step_ids.tolist(),
            }
            for name in ("train", "validation", "test")
        },
        "probe": {
            "feature_dim": probe.feature_dim,
            "solver": probe.solver,
            "ridge": probe.ridge,
        },
        "baseline": {"mean_target": baseline_metrics},
        "validation": validation_metrics,
        "test": test_metrics,
        "timing": timings,
        "decision": {
            "gates": gates,
            "passed": all(gates.values()),
            "next": (
                "micro_action_predictor"
                if all(gates.values())
                else "reject_or_redesign_representation"
            ),
        },
        "features": str(feature_output.resolve()),
        "result": str(result_output.resolve()),
    }
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int64),
        "dataset_sha256": np.asarray(summary["dataset"]["sha256"]),
        "model_id": np.asarray(str(metadata["model_id"])),
        "requested_revision": np.asarray(str(metadata["requested_revision"])),
        "resolved_revision": np.asarray(str(metadata["resolved_revision"])),
        "split_seed": np.asarray(seed, dtype=np.int64),
        "probe_weight": np.asarray(probe.weight, dtype=np.float64),
        "probe_bias": np.asarray(probe.bias, dtype=np.float64),
    }
    for name in ("train", "validation", "test"):
        clips, features, _ = extracted[f"{name}_recorded"]
        arrays[f"{name}_recorded_features"] = np.asarray(
            features,
            dtype=np.float32,
        )
        arrays[f"{name}_targets"] = state_to_probe_targets(clips.states)
        arrays[f"{name}_episode_ids"] = clips.episode_ids
        arrays[f"{name}_step_ids"] = clips.step_ids
        arrays[f"{name}_positions"] = positions[name]
    arrays["test_reversed_features"] = np.asarray(
        reversed_features,
        dtype=np.float32,
    )
    arrays["test_repeat_last_features"] = np.asarray(
        repeated_features,
        dtype=np.float32,
    )
    write_new_file_atomically(
        feature_output,
        writer=lambda handle: np.savez_compressed(handle, **arrays),
        exists_message=f"feature cache already exists: {feature_output}",
    )
    try:
        payload = _json_payload(summary)
        write_new_file_atomically(
            result_output,
            writer=lambda handle: handle.write(payload),
            exists_message=f"result already exists: {result_output}",
        )
    except Exception:
        feature_output.unlink(missing_ok=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/visual_episodes.npz"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/vjepa2_probe_features.npz"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/vjepa2_probe_result.json"),
    )
    parser.add_argument("--model", default=DEFAULT_VJEPA_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=128)
    parser.add_argument("--max-validation", type=int, default=32)
    parser.add_argument("--max-test", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--split-seed", type=int, default=42)
    arguments = parser.parse_args()
    try:
        summary = run_probe(
            data_path=arguments.data,
            feature_path=arguments.features,
            result_path=arguments.result,
            model_id=arguments.model,
            model_revision=arguments.revision,
            device=arguments.device,
            batch_size=arguments.batch_size,
            max_train=arguments.max_train,
            max_validation=arguments.max_validation,
            max_test=arguments.max_test,
            ridge=arguments.ridge,
            split_seed=arguments.split_seed,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
