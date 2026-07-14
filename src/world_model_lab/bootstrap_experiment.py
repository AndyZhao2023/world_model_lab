"""Compare seed-only and episode-bootstrap ensemble diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import csv
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .dataset import (
    build_model_arrays,
    build_sequence_windows,
    fit_normalizer,
    split_episode_ids,
)
from .diagnose_ensemble import run_ensemble_diagnostics
from .diagnose_model import sha256_file
from .ensemble import DISAGREEMENT_NAMES, WorldModelEnsemble, load_ensemble
from .train_world_model import run_training


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


def _validate_comparable_ensembles(
    baseline: WorldModelEnsemble,
    bootstrap: WorldModelEnsemble,
) -> None:
    """Validate that bootstrap sampling is the only ensemble difference."""

    if baseline.seeds != bootstrap.seeds:
        raise ValueError("baseline and bootstrap training seeds differ")

    comparable_config_fields = (
        "split_seed",
        "hidden_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "rollout_horizon",
        "rollout_loss_weight",
    )
    bootstrap_fields = {
        "bootstrap_seed",
        "bootstrap_episode_draws",
        "bootstrap_unique_episodes",
        "bootstrap_episode_counts",
    }
    for seed, baseline_member, bootstrap_member in zip(
        baseline.seeds,
        baseline.members,
        bootstrap.members,
        strict=True,
    ):
        baseline_config = baseline_member.training_config
        bootstrap_config = bootstrap_member.training_config
        unexpected = bootstrap_fields.intersection(baseline_config)
        if unexpected:
            raise ValueError(
                "baseline checkpoints must not contain bootstrap provenance"
            )
        missing = bootstrap_fields.difference(bootstrap_config)
        if missing:
            raise ValueError(
                "bootstrap checkpoints are missing bootstrap provenance: "
                + ", ".join(sorted(missing))
            )
        if (
            not _is_integer(bootstrap_config["bootstrap_seed"])
            or int(bootstrap_config["bootstrap_seed"]) != seed
        ):
            raise ValueError("bootstrap_seed must equal the training seed")
        for name in comparable_config_fields:
            if name not in baseline_config or name not in bootstrap_config:
                raise ValueError(
                    f"baseline and bootstrap {name} values are required"
                )
            if baseline_config[name] != bootstrap_config[name]:
                raise ValueError(
                    f"baseline and bootstrap {name} values differ"
                )
        baseline_data_path = baseline_config.get("data_path")
        bootstrap_data_path = bootstrap_config.get("data_path")
        if not isinstance(baseline_data_path, (str, Path)) or not isinstance(
            bootstrap_data_path,
            (str, Path),
        ):
            raise ValueError("baseline and bootstrap data_path values are required")
        if (
            Path(baseline_data_path).expanduser().resolve()
            != Path(bootstrap_data_path).expanduser().resolve()
        ):
            raise ValueError("baseline and bootstrap data_path values differ")
        for split_name in ("train", "validation", "test"):
            if not np.array_equal(
                baseline_member.split_episode_ids.get(split_name),
                bootstrap_member.split_episode_ids.get(split_name),
            ):
                raise ValueError(
                    "baseline and bootstrap "
                    f"{split_name} split episode IDs differ"
                )
        for label, baseline_values, bootstrap_values in (
            (
                "input mean",
                baseline_member.input_normalizer.mean,
                bootstrap_member.input_normalizer.mean,
            ),
            (
                "input std",
                baseline_member.input_normalizer.std,
                bootstrap_member.input_normalizer.std,
            ),
            (
                "target mean",
                baseline_member.target_normalizer.mean,
                bootstrap_member.target_normalizer.mean,
            ),
            (
                "target std",
                baseline_member.target_normalizer.std,
                bootstrap_member.target_normalizer.std,
            ),
        ):
            if not np.array_equal(baseline_values, bootstrap_values):
                raise ValueError(
                    f"baseline and bootstrap {label} values differ"
                )


def _is_integer(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, np.integer)
    )


def _positive_integer(name: str, value: object) -> int:
    if not _is_integer(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(
    name: str,
    value: object,
    *,
    positive: bool,
) -> float:
    requirement = "positive" if positive else "non-negative"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and {requirement}")
    normalized = float(value)
    if not math.isfinite(normalized) or (
        normalized <= 0.0 if positive else normalized < 0.0
    ):
        raise ValueError(f"{name} must be finite and {requirement}")
    return normalized


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(seeds)
    except TypeError:
        raise ValueError("training seeds must be an iterable") from None
    if len(values) < 2:
        raise ValueError("at least two training seeds are required")
    if any(not _is_integer(seed) or int(seed) < 0 for seed in values):
        raise ValueError("training seeds must be non-negative integers")
    normalized = tuple(int(seed) for seed in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("training seeds must be unique")
    return tuple(sorted(normalized))


def _validate_baseline_protocol(
    baseline: WorldModelEnsemble,
    *,
    data_path: Path,
    seeds: tuple[int, ...],
    split_seed: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    rollout_loss_weight: float,
) -> None:
    if baseline.seeds != seeds:
        raise ValueError(
            "baseline checkpoint training seeds differ from requested seeds"
        )
    expected = {
        "split_seed": split_seed,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "rollout_horizon": 10,
        "rollout_loss_weight": rollout_loss_weight,
    }
    bootstrap_fields = {
        "bootstrap_seed",
        "bootstrap_episode_draws",
        "bootstrap_unique_episodes",
        "bootstrap_episode_counts",
    }
    for member in baseline.members:
        config = member.training_config
        if bootstrap_fields.intersection(config):
            raise ValueError(
                "baseline checkpoints must not contain bootstrap provenance"
            )
        for name, expected_value in expected.items():
            if config.get(name) != expected_value:
                raise ValueError(
                    f"baseline checkpoint {name} differs from requested {name}"
                )
        configured_data = config.get("data_path")
        if not isinstance(configured_data, (str, Path)):
            raise ValueError("baseline checkpoint data_path is missing")
        if Path(configured_data).expanduser().resolve() != data_path:
            raise ValueError(
                "baseline checkpoint data_path differs from the current data_path"
            )


def _sha256_value(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character hexadecimal string")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal string")
    return normalized


def _load_and_validate_baseline_manifest(
    manifest_path: Path,
    *,
    data_path: Path,
    dataset_hash: str,
    baseline: WorldModelEnsemble,
) -> dict[str, Any]:
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"baseline manifest must contain valid JSON: {manifest_path}"
        ) from error
    manifest = dict(_required_mapping(raw_manifest, name="baseline manifest"))
    schema_version = _required_value(
        manifest,
        "schema_version",
        parent="baseline manifest",
    )
    if not _is_integer(schema_version) or int(schema_version) != 1:
        raise ValueError("baseline manifest schema_version must equal 1")

    dataset = _required_child_mapping(
        manifest,
        "dataset",
        parent="baseline manifest",
    )
    recorded_data_path = _required_value(
        dataset,
        "path",
        parent="baseline manifest.dataset",
    )
    if not isinstance(recorded_data_path, (str, Path)):
        raise ValueError("baseline manifest dataset path must be a path string")
    if Path(recorded_data_path).expanduser().resolve() != data_path:
        raise ValueError(
            "baseline manifest dataset path differs from the current dataset"
        )
    recorded_dataset_hash = _sha256_value(
        _required_value(
            dataset,
            "sha256",
            parent="baseline manifest.dataset",
        ),
        name="baseline manifest dataset SHA-256",
    )
    if recorded_dataset_hash != dataset_hash:
        raise ValueError(
            "baseline manifest dataset SHA-256 differs from the current dataset"
        )

    raw_checkpoints = _required_value(
        manifest,
        "checkpoints",
        parent="baseline manifest",
    )
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise ValueError(
            "baseline manifest checkpoints must be a non-empty list"
        )
    recorded_checkpoints = []
    for index, raw_record in enumerate(raw_checkpoints):
        parent = f"baseline manifest.checkpoints[{index}]"
        record = _required_mapping(raw_record, name=parent)
        seed = _required_value(record, "seed", parent=parent)
        if not _is_integer(seed) or int(seed) < 0:
            raise ValueError(f"{parent}.seed must be a non-negative integer")
        checkpoint_path = _required_value(record, "path", parent=parent)
        if not isinstance(checkpoint_path, (str, Path)):
            raise ValueError(f"{parent}.path must be a path string")
        checkpoint_hash = _sha256_value(
            _required_value(record, "sha256", parent=parent),
            name=f"{parent}.sha256",
        )
        recorded_checkpoints.append(
            (
                int(seed),
                Path(checkpoint_path).expanduser().resolve(),
                checkpoint_hash,
            )
        )
    recorded_checkpoints.sort(key=lambda record: record[0])
    recorded_seeds = tuple(record[0] for record in recorded_checkpoints)
    if len(set(recorded_seeds)) != len(recorded_seeds):
        raise ValueError("baseline manifest checkpoint seeds must be unique")

    member_seeds = _required_value(
        manifest,
        "member_seeds",
        parent="baseline manifest",
    )
    if not isinstance(member_seeds, list) or any(
        not _is_integer(seed) for seed in member_seeds
    ):
        raise ValueError(
            "baseline manifest member_seeds must be an integer list"
        )
    if tuple(int(seed) for seed in member_seeds) != recorded_seeds:
        raise ValueError(
            "baseline manifest member_seeds differ from checkpoint records"
        )

    actual_checkpoints = tuple(
        (
            int(seed),
            path.resolve(),
            sha256_file(path),
        )
        for seed, path in zip(
            baseline.seeds,
            baseline.checkpoint_paths,
            strict=True,
        )
    )
    if recorded_seeds != baseline.seeds:
        raise ValueError(
            "baseline manifest checkpoint seeds differ from supplied checkpoints"
        )
    for recorded, actual in zip(
        recorded_checkpoints,
        actual_checkpoints,
        strict=True,
    ):
        if recorded[1] != actual[1]:
            raise ValueError(
                "baseline manifest checkpoint paths differ from supplied checkpoints"
            )
        if recorded[2] != actual[2]:
            raise ValueError(
                "baseline manifest checkpoint SHA-256 values differ from "
                "supplied checkpoints"
            )
    return manifest


def _validate_dataset_split_coverage(
    data_path: Path,
    ensemble: WorldModelEnsemble,
    *,
    split_seed: int,
    max_diagnostic_horizon: int = 10,
) -> None:
    required_arrays = {
        "states",
        "actions",
        "next_states",
        "episode_ids",
        "step_ids",
    }
    with np.load(data_path, allow_pickle=False) as loaded:
        missing = required_arrays - set(loaded.files)
        if missing:
            raise ValueError(
                f"dataset is missing arrays: {', '.join(sorted(missing))}"
            )
        arrays = {name: np.asarray(loaded[name]) for name in required_arrays}
    states = arrays["states"]
    actions = arrays["actions"]
    next_states = arrays["next_states"]
    episode_ids = arrays["episode_ids"]
    step_ids = arrays["step_ids"]
    inputs, targets = build_model_arrays(states, actions, next_states)
    if (
        episode_ids.ndim != 1
        or episode_ids.shape[0] != inputs.shape[0]
        or episode_ids.size == 0
        or np.issubdtype(episode_ids.dtype, np.bool_)
        or not np.issubdtype(episode_ids.dtype, np.integer)
        or np.any(episode_ids < 0)
    ):
        raise ValueError(
            "dataset episode_ids must be a non-empty non-negative integer vector"
        )
    available = set(int(value) for value in np.unique(episode_ids).tolist())
    reference = ensemble.members[0]
    expected_splits = split_episode_ids(episode_ids, seed=split_seed)
    for split_name in ("train", "validation", "test"):
        split_ids = np.asarray(reference.split_episode_ids[split_name])
        if not np.array_equal(split_ids, expected_splits[split_name]):
            raise ValueError(
                "split_seed recomputation differs from checkpoint "
                f"{split_name} split episode IDs"
            )
        missing = sorted(
            int(value) for value in split_ids.tolist() if int(value) not in available
        )
        if missing:
            raise ValueError(
                f"checkpoint {split_name} episode IDs are missing from the dataset: "
                + ", ".join(map(str, missing))
            )

    train_ids = np.asarray(reference.split_episode_ids["train"])
    train_mask = np.isin(episode_ids, train_ids)
    input_normalizer = fit_normalizer(inputs[train_mask])
    target_normalizer = fit_normalizer(targets[train_mask])
    for label, observed, expected in (
        (
            "input mean",
            input_normalizer.mean,
            reference.input_normalizer.mean,
        ),
        (
            "input std",
            input_normalizer.std,
            reference.input_normalizer.std,
        ),
        (
            "target mean",
            target_normalizer.mean,
            reference.target_normalizer.mean,
        ),
        (
            "target std",
            target_normalizer.std,
            reference.target_normalizer.std,
        ),
    ):
        if not np.allclose(observed, expected, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"dataset full-train {label} differs from baseline checkpoint"
            )

    validation_windows = build_sequence_windows(
        states,
        actions,
        next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        selected_episode_ids=np.asarray(
            reference.split_episode_ids["validation"]
        ),
        horizon=10,
    )
    train_windows = build_sequence_windows(
        states,
        actions,
        next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        selected_episode_ids=train_ids,
        horizon=10,
    )
    diagnostic_windows = build_sequence_windows(
        states,
        actions,
        next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        selected_episode_ids=np.asarray(reference.split_episode_ids["test"]),
        horizon=max_diagnostic_horizon,
    )
    if train_windows.count == 0 or validation_windows.count == 0:
        raise ValueError("dataset has no eligible H10 training or validation windows")
    if diagnostic_windows.count == 0:
        raise ValueError(
            "dataset has no test windows for the maximum diagnostic horizon"
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def run_bootstrap_experiment(
    *,
    data_path: Path | str,
    baseline_checkpoint_paths: Iterable[Path | str],
    baseline_manifest_path: Path | str,
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
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Run the complete seed-only versus episode-bootstrap experiment."""

    data = Path(data_path).expanduser()
    if not data.is_file():
        raise FileNotFoundError(f"dataset is not a regular file: {data}")
    data = data.resolve()
    baseline_manifest_file = Path(baseline_manifest_path).expanduser()
    if not baseline_manifest_file.is_file():
        raise FileNotFoundError(
            "baseline manifest is not a regular file: "
            f"{baseline_manifest_file}"
        )
    baseline_manifest_file = baseline_manifest_file.resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")

    seed_values = _validate_seeds(seeds)
    if not _is_integer(split_seed) or int(split_seed) < 0:
        raise ValueError("split_seed must be a non-negative integer")
    normalized_split_seed = int(split_seed)
    normalized_hidden_size = _positive_integer("hidden_size", hidden_size)
    normalized_epochs = _positive_integer("epochs", epochs)
    normalized_batch_size = _positive_integer("batch_size", batch_size)
    normalized_windows = _positive_integer(
        "windows_per_episode", windows_per_episode
    )
    normalized_bins = _positive_integer("calibration_bins", calibration_bins)
    normalized_learning_rate = _finite_real(
        "learning_rate",
        learning_rate,
        positive=True,
    )
    normalized_rollout_weight = _finite_real(
        "rollout_loss_weight",
        rollout_loss_weight,
        positive=False,
    )
    try:
        horizon_values = _validate_horizons(diagnostic_horizons)
    except ValueError as error:
        raise ValueError(
            str(error).replace("horizons", "diagnostic_horizons", 1)
        ) from None

    try:
        requested_baseline_paths = tuple(baseline_checkpoint_paths)
    except TypeError:
        raise ValueError("baseline_checkpoint_paths must be an iterable") from None
    for checkpoint_path in requested_baseline_paths:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint is not a regular file: {path}"
            )
    baseline = load_ensemble(requested_baseline_paths)
    dataset_hash = sha256_file(data)
    _load_and_validate_baseline_manifest(
        baseline_manifest_file,
        data_path=data,
        dataset_hash=dataset_hash,
        baseline=baseline,
    )
    _validate_baseline_protocol(
        baseline,
        data_path=data,
        seeds=seed_values,
        split_seed=normalized_split_seed,
        hidden_size=normalized_hidden_size,
        epochs=normalized_epochs,
        batch_size=normalized_batch_size,
        learning_rate=normalized_learning_rate,
        rollout_loss_weight=normalized_rollout_weight,
    )
    _validate_dataset_split_coverage(
        data,
        baseline,
        split_seed=normalized_split_seed,
        max_diagnostic_horizon=horizon_values[-1],
    )

    output.mkdir(parents=True, exist_ok=True)
    bootstrap_paths: list[Path] = []
    for seed in seed_values:
        checkpoint_path = output / "runs" / f"seed_{seed}" / "world_model.pt"
        run_training(
            data_path=data,
            output_path=checkpoint_path,
            hidden_size=normalized_hidden_size,
            epochs=normalized_epochs,
            batch_size=normalized_batch_size,
            learning_rate=normalized_learning_rate,
            seed=seed,
            split_seed=normalized_split_seed,
            rollout_horizon=10,
            rollout_loss_weight=normalized_rollout_weight,
            bootstrap_seed=seed,
        )
        bootstrap_paths.append(checkpoint_path)

    bootstrap = load_ensemble(bootstrap_paths)
    _validate_comparable_ensembles(baseline, bootstrap)

    diagnostic_arguments = {
        "data_path": data,
        "horizons": horizon_values,
        "windows_per_episode": normalized_windows,
        "calibration_bins": normalized_bins,
    }
    baseline_diagnostics = run_ensemble_diagnostics(
        checkpoint_paths=baseline.checkpoint_paths,
        output_dir=output / "baseline_diagnostics",
        **diagnostic_arguments,
    )
    bootstrap_diagnostics = run_ensemble_diagnostics(
        checkpoint_paths=bootstrap.checkpoint_paths,
        output_dir=output / "bootstrap_diagnostics",
        **diagnostic_arguments,
    )

    baseline_metrics_path = output / "baseline_diagnostics" / "metrics.json"
    bootstrap_metrics_path = output / "bootstrap_diagnostics" / "metrics.json"
    baseline_metrics = json.loads(
        baseline_metrics_path.read_text(encoding="utf-8")
    )
    bootstrap_metrics = json.loads(
        bootstrap_metrics_path.read_text(encoding="utf-8")
    )
    comparison = build_bootstrap_comparison(
        baseline_metrics,
        bootstrap_metrics,
        horizons=horizon_values,
    )

    comparison_json_path = output / "comparison.json"
    comparison_csv_path = output / "comparison.csv"
    comparison_plot_path = output / "comparison.png"
    _write_json(comparison_json_path, comparison)
    write_comparison_csv(comparison, comparison_csv_path)
    plot_bootstrap_comparison(comparison, comparison_plot_path)

    baseline_records = [
        {
            "seed": int(seed),
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for seed, path in zip(
            baseline.seeds,
            baseline.checkpoint_paths,
            strict=True,
        )
    ]
    bootstrap_records = [
        {
            "seed": int(seed),
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
        }
        for seed, path in zip(seed_values, bootstrap_paths, strict=True)
    ]
    manifest = {
        "schema_version": 1,
        "dataset": {"path": str(data), "sha256": dataset_hash},
        "baseline_manifest": {
            "path": str(baseline_manifest_file),
            "sha256": sha256_file(baseline_manifest_file),
        },
        "baseline_checkpoints": baseline_records,
        "bootstrap_checkpoints": bootstrap_records,
        "training": {
            "seeds": list(seed_values),
            "split_seed": normalized_split_seed,
            "hidden_size": normalized_hidden_size,
            "epochs": normalized_epochs,
            "batch_size": normalized_batch_size,
            "learning_rate": normalized_learning_rate,
            "rollout_horizon": 10,
            "rollout_loss_weight": normalized_rollout_weight,
            "bootstrap_strategy": "episode_with_replacement",
            "bootstrap_draws": "train_episode_count",
        },
        "diagnostics": {
            "horizons": list(horizon_values),
            "windows_per_episode": normalized_windows,
            "calibration_bins": normalized_bins,
            "baseline": "baseline_diagnostics",
            "bootstrap": "bootstrap_diagnostics",
        },
        "comparison": {
            "json": comparison_json_path.relative_to(output).as_posix(),
            "csv": comparison_csv_path.relative_to(output).as_posix(),
            "plot": comparison_plot_path.relative_to(output).as_posix(),
        },
    }
    manifest_path = output / "experiment_manifest.json"
    _write_json(manifest_path, manifest)

    return {
        "experiment_manifest.json": str(manifest_path),
        "comparison.json": str(comparison_json_path),
        "comparison.csv": str(comparison_csv_path),
        "comparison.png": str(comparison_plot_path),
        "baseline_diagnostics": baseline_diagnostics,
        "bootstrap_diagnostics": bootstrap_diagnostics,
        "runs": {
            str(seed): str(path)
            for seed, path in zip(seed_values, bootstrap_paths, strict=True)
        },
    }


def main() -> None:
    """Run the episode-bootstrap ensemble experiment from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    parser.add_argument(
        "--baseline-checkpoints",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/bootstrap-h10-seeds-0-4"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
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
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()

    try:
        result = run_bootstrap_experiment(
            data_path=args.data,
            baseline_checkpoint_paths=args.baseline_checkpoints,
            baseline_manifest_path=args.baseline_manifest,
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
            calibration_bins=args.calibration_bins,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
