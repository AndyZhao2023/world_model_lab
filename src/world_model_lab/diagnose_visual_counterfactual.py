"""Evaluate visual world models against matched simulator counterfactuals."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from .diagnose_model import sha256_file
from .diagnose_visual_rollout import (
    _decode_normalized_rollout_latents,
    _safe_relative_percent,
    _sattolo_permutation,
    _stack_rollout_windows,
    _validate_diagnostic_protocol,
    _validate_visual_checkpoint_pair,
    rollout_normalized_latents,
    select_visual_rollout_windows,
)
from .train_visual_latent_model import (
    LoadedVisualLatentModel,
    load_visual_latent_checkpoint,
)
from .visual_counterfactual_data import (
    MatchedCounterfactualBatch,
    build_matched_counterfactual_batch,
)
from .visual_dataset import load_visual_dataset
from .visual_latent_data import (
    encode_all_frames,
    frame_indices_for_episode_ids,
)
from .visual_object_position import (
    LinearObjectPositionProbe,
    evaluate_object_position_probe,
    fit_linear_object_position_probe,
    normalize_object_positions,
    object_position_probe_sha256,
    predict_normalized_object_positions,
    world_position_errors,
)


_METRIC_NAMES = (
    "normalized_latent_mse",
    "pixel_mse",
    "transition_changed_pixel_mae",
    "cumulative_changed_pixel_mae",
    "normalized_latent_effect_mse",
    "pixel_effect_mse",
    "normalized_position_mse",
    "world_position_error",
    "normalized_position_effect_mse",
)

_OBJECT_POSITION_PROBE_RIDGE = 1e-3


def _masked_episode_macro_curve(
    values: np.ndarray,
    *,
    valid_steps: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_steps)
    ids = np.asarray(episode_ids)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or valid.shape != array.shape
        or valid.dtype != np.dtype(np.bool_)
        or ids.shape != (array.shape[0],)
        or ids.dtype.kind not in "iu"
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("masked episode metric arrays are invalid")
    horizon = int(array.shape[1])
    curve = np.empty(horizon, dtype=np.float64)
    episode_counts = np.empty(horizon, dtype=np.int64)
    window_counts = np.empty(horizon, dtype=np.int64)
    for step in range(horizon):
        selected_ids = np.unique(ids[valid[:, step]])
        if selected_ids.size == 0:
            raise ValueError(
                f"counterfactual step {step + 1} has no valid episodes"
            )
        per_episode = [
            np.mean(
                array[
                    (ids == episode_id) & valid[:, step],
                    step,
                ]
            )
            for episode_id in selected_ids
        ]
        curve[step] = float(np.mean(per_episode))
        episode_counts[step] = int(selected_ids.size)
        window_counts[step] = int(np.sum(valid[:, step]))
    return curve, episode_counts, window_counts


def _masked_changed_pixel_curve(
    *,
    predicted_frames: np.ndarray,
    true_frames: np.ndarray,
    comparison_frames: np.ndarray,
    valid_steps: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(predicted_frames, dtype=np.float64)
    targets = np.asarray(true_frames)
    comparisons = np.asarray(comparison_frames)
    valid = np.asarray(valid_steps)
    ids = np.asarray(episode_ids)
    if (
        predictions.ndim != 5
        or targets.shape != predictions.shape
        or comparisons.shape != predictions.shape
        or targets.dtype != np.dtype(np.uint8)
        or comparisons.dtype != np.dtype(np.uint8)
        or valid.shape != predictions.shape[:2]
        or valid.dtype != np.dtype(np.bool_)
        or ids.shape != (predictions.shape[0],)
        or ids.dtype.kind not in "iu"
        or not np.all(np.isfinite(predictions))
    ):
        raise ValueError("masked changed-pixel arrays are invalid")
    target_values = targets.astype(np.float64) / 255.0
    errors = np.abs(predictions - target_values)
    changed = np.any(comparisons != targets, axis=2)
    channels = int(predictions.shape[2])
    horizon = int(predictions.shape[1])
    curve = np.empty(horizon, dtype=np.float64)
    for step in range(horizon):
        selected_ids = np.unique(ids[valid[:, step]])
        if selected_ids.size == 0:
            raise ValueError(
                f"counterfactual step {step + 1} has no valid episodes"
            )
        episode_values: list[float] = []
        for episode_id in selected_ids:
            selected = (ids == episode_id) & valid[:, step]
            masks = changed[selected, step]
            selected_errors = errors[selected, step]
            numerator = float(
                np.sum(selected_errors * masks[:, None, :, :])
            )
            denominator = float(np.sum(masks) * channels)
            episode_values.append(
                numerator / denominator if denominator > 0.0 else 0.0
            )
        curve[step] = float(np.mean(episode_values))
    return curve


def summarize_matched_counterfactual_predictions(
    *,
    predicted_counterfactual_normalized_latents: np.ndarray,
    true_counterfactual_normalized_latents: np.ndarray,
    predicted_factual_normalized_latents: np.ndarray,
    true_factual_normalized_latents: np.ndarray,
    predicted_counterfactual_frames: np.ndarray,
    true_counterfactual_frames: np.ndarray,
    predicted_factual_frames: np.ndarray,
    true_factual_frames: np.ndarray,
    true_initial_frames: np.ndarray,
    position_probe: LinearObjectPositionProbe,
    true_counterfactual_normalized_positions: np.ndarray,
    true_factual_normalized_positions: np.ndarray,
    world_bounds: np.ndarray,
    valid_steps: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize one seed with terminal masks and episode-equal weighting."""

    predicted_cf_latents = np.asarray(
        predicted_counterfactual_normalized_latents
    )
    true_cf_latents = np.asarray(
        true_counterfactual_normalized_latents
    )
    predicted_factual_latents = np.asarray(
        predicted_factual_normalized_latents
    )
    true_factual_latents = np.asarray(
        true_factual_normalized_latents
    )
    latent_shape = predicted_cf_latents.shape
    if (
        predicted_cf_latents.ndim != 3
        or latent_shape[0] == 0
        or latent_shape[1] == 0
        or latent_shape[2] == 0
        or true_cf_latents.shape != latent_shape
        or predicted_factual_latents.shape != latent_shape
        or true_factual_latents.shape != latent_shape
        or not all(
            np.all(np.isfinite(values))
            for values in (
                predicted_cf_latents,
                true_cf_latents,
                predicted_factual_latents,
                true_factual_latents,
            )
        )
    ):
        raise ValueError(
            "matched latent arrays must share finite non-empty [N, H, D]"
        )
    count, horizon, _ = latent_shape
    predicted_cf_frames = np.asarray(predicted_counterfactual_frames)
    true_cf_frames = np.asarray(true_counterfactual_frames)
    predicted_factual_image = np.asarray(predicted_factual_frames)
    true_factual_image = np.asarray(true_factual_frames)
    initial_frames = np.asarray(true_initial_frames)
    frame_shape = predicted_cf_frames.shape
    if (
        predicted_cf_frames.ndim != 5
        or frame_shape[:2] != (count, horizon)
        or frame_shape[2] != 3
        or true_cf_frames.shape != frame_shape
        or predicted_factual_image.shape != frame_shape
        or true_factual_image.shape != frame_shape
        or initial_frames.shape
        != (count, frame_shape[2], frame_shape[3], frame_shape[4])
        or true_cf_frames.dtype != np.dtype(np.uint8)
        or true_factual_image.dtype != np.dtype(np.uint8)
        or initial_frames.dtype != np.dtype(np.uint8)
        or not (
            np.all(np.isfinite(predicted_cf_frames))
            and np.all(np.isfinite(predicted_factual_image))
        )
    ):
        raise ValueError("matched frame arrays are invalid")
    valid = np.asarray(valid_steps)
    ids = np.asarray(episode_ids)
    if (
        valid.shape != (count, horizon)
        or valid.dtype != np.dtype(np.bool_)
        or ids.shape != (count,)
        or ids.dtype.kind not in "iu"
    ):
        raise ValueError("matched validity or episode arrays are invalid")
    true_cf_positions = np.asarray(
        true_counterfactual_normalized_positions
    )
    true_factual_positions = np.asarray(
        true_factual_normalized_positions
    )
    position_shape = (count, horizon, 2)
    if (
        not isinstance(position_probe, LinearObjectPositionProbe)
        or true_cf_positions.shape != position_shape
        or true_factual_positions.shape != position_shape
        or not np.all(np.isfinite(true_cf_positions))
        or not np.all(np.isfinite(true_factual_positions))
    ):
        raise ValueError("matched object-position arrays are invalid")

    latent_direct = np.mean(
        np.square(
            predicted_cf_latents.astype(np.float64)
            - true_cf_latents
        ),
        axis=2,
    )
    predicted_latent_effect = (
        predicted_cf_latents.astype(np.float64)
        - predicted_factual_latents
    )
    true_latent_effect = (
        true_cf_latents.astype(np.float64) - true_factual_latents
    )
    latent_effect = np.mean(
        np.square(predicted_latent_effect - true_latent_effect),
        axis=2,
    )
    true_cf_pixels = true_cf_frames.astype(np.float64) / 255.0
    true_factual_pixels = (
        true_factual_image.astype(np.float64) / 255.0
    )
    pixel_direct = np.mean(
        np.square(
            predicted_cf_frames.astype(np.float64) - true_cf_pixels
        ),
        axis=(2, 3, 4),
    )
    predicted_pixel_effect = (
        predicted_cf_frames.astype(np.float64)
        - predicted_factual_image
    )
    true_pixel_effect = true_cf_pixels - true_factual_pixels
    pixel_effect = np.mean(
        np.square(predicted_pixel_effect - true_pixel_effect),
        axis=(2, 3, 4),
    )
    predicted_cf_positions = predict_normalized_object_positions(
        position_probe,
        predicted_cf_latents,
    )
    predicted_factual_positions = predict_normalized_object_positions(
        position_probe,
        predicted_factual_latents,
    )
    position_direct = np.mean(
        np.square(
            predicted_cf_positions.astype(np.float64)
            - true_cf_positions
        ),
        axis=2,
    )
    world_position_direct = world_position_errors(
        predicted_cf_positions,
        true_cf_positions,
        world_bounds,
    )
    predicted_position_effect = (
        predicted_cf_positions.astype(np.float64)
        - predicted_factual_positions
    )
    true_position_effect = (
        true_cf_positions.astype(np.float64)
        - true_factual_positions
    )
    position_effect = np.mean(
        np.square(
            predicted_position_effect - true_position_effect
        ),
        axis=2,
    )
    latent_curve, episode_counts, window_counts = (
        _masked_episode_macro_curve(
            latent_direct,
            valid_steps=valid,
            episode_ids=ids,
        )
    )
    pixel_curve, _, _ = _masked_episode_macro_curve(
        pixel_direct,
        valid_steps=valid,
        episode_ids=ids,
    )
    latent_effect_curve, _, _ = _masked_episode_macro_curve(
        latent_effect,
        valid_steps=valid,
        episode_ids=ids,
    )
    pixel_effect_curve, _, _ = _masked_episode_macro_curve(
        pixel_effect,
        valid_steps=valid,
        episode_ids=ids,
    )
    position_curve, _, _ = _masked_episode_macro_curve(
        position_direct,
        valid_steps=valid,
        episode_ids=ids,
    )
    world_position_curve, _, _ = _masked_episode_macro_curve(
        world_position_direct,
        valid_steps=valid,
        episode_ids=ids,
    )
    position_effect_curve, _, _ = _masked_episode_macro_curve(
        position_effect,
        valid_steps=valid,
        episode_ids=ids,
    )
    transition_comparisons = np.concatenate(
        (initial_frames[:, None], true_cf_frames[:, :-1]),
        axis=1,
    )
    cumulative_comparisons = np.repeat(
        initial_frames[:, None],
        horizon,
        axis=1,
    )
    transition_changed = _masked_changed_pixel_curve(
        predicted_frames=predicted_cf_frames,
        true_frames=true_cf_frames,
        comparison_frames=transition_comparisons,
        valid_steps=valid,
        episode_ids=ids,
    )
    cumulative_changed = _masked_changed_pixel_curve(
        predicted_frames=predicted_cf_frames,
        true_frames=true_cf_frames,
        comparison_frames=cumulative_comparisons,
        valid_steps=valid,
        episode_ids=ids,
    )
    curves = {
        "normalized_latent_mse": latent_curve,
        "pixel_mse": pixel_curve,
        "transition_changed_pixel_mae": transition_changed,
        "cumulative_changed_pixel_mae": cumulative_changed,
        "normalized_latent_effect_mse": latent_effect_curve,
        "pixel_effect_mse": pixel_effect_curve,
        "normalized_position_mse": position_curve,
        "world_position_error": world_position_curve,
        "normalized_position_effect_mse": position_effect_curve,
    }
    return {
        str(step + 1): {
            "episodes": int(episode_counts[step]),
            "valid_windows": int(window_counts[step]),
            **{
                name: float(values[step])
                for name, values in curves.items()
            },
        }
        for step in range(horizon)
    }


def aggregate_counterfactual_seed_records(
    records: Iterable[Mapping[str, Mapping[str, float | int]]],
    *,
    seeds: Iterable[int],
) -> dict[str, object]:
    """Aggregate episode-macro counterfactual records across fixed seeds."""

    selected = tuple(records)
    seed_values = tuple(seeds)
    if (
        not selected
        or len(selected) != len(seed_values)
        or any(
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
            for seed in seed_values
        )
        or len(set(int(seed) for seed in seed_values)) != len(seed_values)
    ):
        raise ValueError("counterfactual records and seeds are invalid")
    steps = tuple(selected[0].keys())
    if not steps or any(tuple(record.keys()) != steps for record in selected):
        raise ValueError("counterfactual seed records have different steps")
    aggregated: dict[str, object] = {}
    for step in steps:
        step_record: dict[str, object] = {}
        for metric in _METRIC_NAMES:
            values = np.asarray(
                [record[step][metric] for record in selected],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "counterfactual seed metrics must be finite"
                )
            step_record[metric] = {
                "mean": float(np.mean(values)),
                "sample_std": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
            }
        aggregated[step] = step_record
    return {
        "seeds": [int(seed) for seed in seed_values],
        "steps": aggregated,
    }


def _build_counterfactual_comparison(
    *,
    source: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    source_steps = source.get("steps")
    candidate_steps = candidate.get("steps")
    if not isinstance(source_steps, Mapping) or not isinstance(
        candidate_steps,
        Mapping,
    ) or tuple(source_steps.keys()) != tuple(candidate_steps.keys()):
        raise ValueError("counterfactual model steps do not match")
    comparison: dict[str, object] = {}
    for step in source_steps:
        source_record = source_steps[step]
        candidate_record = candidate_steps[step]
        if not isinstance(source_record, Mapping) or not isinstance(
            candidate_record,
            Mapping,
        ):
            raise ValueError("counterfactual model records are malformed")
        comparison[step] = {}
        step_comparison = comparison[step]
        assert isinstance(step_comparison, dict)
        for metric in _METRIC_NAMES:
            source_metric = source_record.get(metric)
            candidate_metric = candidate_record.get(metric)
            if not isinstance(source_metric, Mapping) or not isinstance(
                candidate_metric,
                Mapping,
            ):
                raise ValueError("counterfactual metric record is missing")
            source_value = float(source_metric["mean"])
            candidate_value = float(candidate_metric["mean"])
            if not (
                np.isfinite(source_value)
                and np.isfinite(candidate_value)
            ):
                raise ValueError("counterfactual comparison must be finite")
            step_comparison[metric] = {
                "absolute": candidate_value - source_value,
                "relative_percent": _safe_relative_percent(
                    candidate_value,
                    source_value,
                ),
            }
    return comparison


def _metric_mean(
    model: Mapping[str, object],
    *,
    horizon: int,
    metric: str,
) -> float:
    steps = model.get("steps")
    if not isinstance(steps, Mapping):
        raise ValueError("counterfactual model steps are missing")
    record = steps.get(str(horizon))
    if not isinstance(record, Mapping):
        raise ValueError(
            f"counterfactual horizon {horizon} is missing"
        )
    metric_record = record.get(metric)
    if not isinstance(metric_record, Mapping):
        raise ValueError(
            f"counterfactual metric {metric} is missing"
        )
    value = float(metric_record["mean"])
    if not np.isfinite(value):
        raise ValueError("counterfactual metric must be finite")
    return value


def _build_preregistered_decision(
    *,
    source: Mapping[str, object],
    candidate: Mapping[str, object],
    decision_horizon: int,
) -> dict[str, object]:
    if (
        isinstance(decision_horizon, (bool, np.bool_))
        or not isinstance(decision_horizon, (int, np.integer))
        or int(decision_horizon) <= 1
    ):
        raise ValueError(
            "decision_horizon must be an integer greater than one"
        )
    horizon = int(decision_horizon)
    gate_specs = (
        (
            "horizon_direct_latent_improvement",
            horizon,
            "normalized_latent_mse",
            "strict_improvement",
            1.0,
        ),
        (
            "horizon_cumulative_changed_pixel_improvement",
            horizon,
            "cumulative_changed_pixel_mae",
            "strict_improvement",
            1.0,
        ),
        (
            "horizon_latent_action_effect_improvement",
            horizon,
            "normalized_latent_effect_mse",
            "strict_improvement",
            1.0,
        ),
        (
            "horizon_direct_position_improvement",
            horizon,
            "normalized_position_mse",
            "strict_improvement",
            1.0,
        ),
        (
            "h1_direct_latent_stability",
            1,
            "normalized_latent_mse",
            "upper_bound",
            1.10,
        ),
        (
            "h1_cumulative_changed_pixel_stability",
            1,
            "cumulative_changed_pixel_mae",
            "upper_bound",
            1.05,
        ),
    )
    gates: list[dict[str, object]] = []
    for name, gate_horizon, metric, kind, multiplier in gate_specs:
        source_value = _metric_mean(
            source,
            horizon=gate_horizon,
            metric=metric,
        )
        candidate_value = _metric_mean(
            candidate,
            horizon=gate_horizon,
            metric=metric,
        )
        limit = source_value * multiplier
        passed = (
            candidate_value < limit
            if kind == "strict_improvement"
            else candidate_value <= limit
        )
        gates.append(
            {
                "name": name,
                "horizon": gate_horizon,
                "metric": metric,
                "operator": "<" if kind == "strict_improvement" else "<=",
                "source": source_value,
                "limit": limit,
                "candidate": candidate_value,
                "passed": bool(passed),
            }
        )
    return {
        "decision_horizon": horizon,
        "candidate_passes": all(gate["passed"] for gate in gates),
        "gates": gates,
    }


def _encode_counterfactual_targets(
    *,
    model: LoadedVisualLatentModel,
    batch: MatchedCounterfactualBatch,
    batch_size: int,
) -> np.ndarray:
    valid_frames = batch.true_frames[batch.valid_steps]
    encoded = encode_all_frames(
        model.autoencoder,
        valid_frames,
        batch_size=batch_size,
    )
    latent_dim = int(model.autoencoder.latent_dim)
    raw = np.broadcast_to(
        model.latent_normalizer.mean,
        (batch.count, batch.horizon, latent_dim),
    ).copy()
    raw[batch.valid_steps] = encoded
    normalized = model.latent_normalizer.normalize(raw)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("encoded counterfactual targets are non-finite")
    return np.asarray(normalized, dtype=np.float32)


def _channels_first_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim == 4:
        return np.transpose(values, (0, 3, 1, 2))
    if values.ndim == 5:
        return np.transpose(values, (0, 1, 4, 2, 3))
    raise ValueError("visual frames must have four or five dimensions")


def _aggregate_coverage(
    batches: Iterable[MatchedCounterfactualBatch],
    *,
    seeds: Iterable[int],
) -> dict[str, object]:
    selected = tuple(batches)
    seed_values = tuple(seeds)
    if not selected or len(selected) != len(seed_values):
        raise ValueError("counterfactual coverage batches are invalid")
    count = selected[0].count
    horizon = selected[0].horizon
    if any(
        batch.count != count or batch.horizon != horizon
        for batch in selected
    ):
        raise ValueError("counterfactual coverage shapes do not match")
    terminal_reasons: dict[str, int] = {}
    clipped_action_steps = 0
    for batch in selected:
        clipped_action_steps += int(
            np.sum(
                np.any(
                    batch.requested_actions != batch.applied_actions,
                    axis=2,
                )
            )
        )
        for reason in batch.terminal_reasons.tolist():
            if reason:
                terminal_reasons[str(reason)] = (
                    terminal_reasons.get(str(reason), 0) + 1
                )
    possible = len(selected) * count
    return {
        "seeds": [int(seed) for seed in seed_values],
        "branches_per_seed": count,
        "clipped_action_steps": clipped_action_steps,
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "steps": {
            str(step + 1): {
                "valid_branches": int(
                    sum(
                        np.sum(batch.valid_steps[:, step])
                        for batch in selected
                    )
                ),
                "possible_branches": possible,
                "valid_fraction": float(
                    sum(
                        np.sum(batch.valid_steps[:, step])
                        for batch in selected
                    )
                    / possible
                ),
            }
            for step in range(horizon)
        },
    }


def plot_matched_counterfactual_comparison(
    *,
    metrics: Mapping[str, object],
    output_path: Path | str,
) -> Path:
    """Plot direct matched accuracy and action-effect errors."""

    models = metrics.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("counterfactual metrics models are missing")
    source = models.get("source")
    candidate = models.get("candidate")
    if not isinstance(source, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("counterfactual source or candidate is missing")
    source_steps = source.get("steps")
    candidate_steps = candidate.get("steps")
    if not isinstance(source_steps, Mapping) or not isinstance(
        candidate_steps,
        Mapping,
    ):
        raise ValueError("counterfactual plot steps are missing")
    horizons = np.asarray(
        [int(step) for step in source_steps.keys()],
        dtype=np.int64,
    )
    panels = (
        ("normalized_latent_mse", "Matched latent error", "MSE"),
        (
            "cumulative_changed_pixel_mae",
            "Matched cumulative changed-pixel error",
            "MAE",
        ),
        (
            "normalized_latent_effect_mse",
            "Latent action-effect error",
            "MSE",
        ),
        (
            "world_position_error",
            "Matched car-centre error",
            "world units",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        for name, record, color in (
            ("source", source, "tab:blue"),
            ("candidate", candidate, "tab:orange"),
        ):
            steps = record["steps"]
            means = np.asarray(
                [steps[str(step)][metric]["mean"] for step in horizons],
                dtype=np.float64,
            )
            stds = np.asarray(
                [
                    steps[str(step)][metric]["sample_std"]
                    for step in horizons
                ],
                dtype=np.float64,
            )
            axis.plot(horizons, means, label=name, color=color)
            axis.fill_between(
                horizons,
                np.maximum(0.0, means - stds),
                means + stds,
                color=color,
                alpha=0.15,
            )
        if metric == "cumulative_changed_pixel_mae":
            oracle = metrics.get("oracle")
            if isinstance(oracle, Mapping):
                oracle_steps = oracle.get("steps")
                if isinstance(oracle_steps, Mapping):
                    oracle_values = [
                        oracle_steps[str(step)][metric]["mean"]
                        for step in horizons
                    ]
                    axis.plot(
                        horizons,
                        oracle_values,
                        label="oracle reconstruction",
                        color="tab:gray",
                        linestyle="--",
                    )
        axis.set_title(title)
        axis.set_xlabel("rollout step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Matched visual counterfactual diagnostics")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _write_json(path: Path, value: Mapping[str, object]) -> Path:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_visual_counterfactual_diagnostics(
    *,
    data_path: Path | str,
    source_checkpoint_path: Path | str,
    candidate_checkpoint_path: Path | str,
    output_dir: Path | str,
    horizons: Iterable[int] = (1, 5, 10),
    windows_per_episode: int = 8,
    counterfactual_seeds: Iterable[int] = tuple(range(10)),
    decision_horizon: int = 5,
    encode_batch_size: int = 256,
    decode_batch_size: int = 256,
) -> dict[str, object]:
    """Compare compatible models against matched simulator branches."""

    horizon_values, seed_values = _validate_diagnostic_protocol(
        horizons=horizons,
        windows_per_episode=windows_per_episode,
        counterfactual_seeds=counterfactual_seeds,
    )
    if 1 not in horizon_values:
        raise ValueError("horizons must include 1 for stability gates")
    if (
        isinstance(decision_horizon, (bool, np.bool_))
        or not isinstance(decision_horizon, (int, np.integer))
        or int(decision_horizon) not in horizon_values
        or int(decision_horizon) <= 1
    ):
        raise ValueError(
            "decision_horizon must be greater than one and in horizons"
        )
    if encode_batch_size <= 0 or decode_batch_size <= 0:
        raise ValueError("encode and decode batch sizes must be positive")
    data = Path(data_path)
    source_path = Path(source_checkpoint_path)
    candidate_path = Path(candidate_checkpoint_path)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")
    dataset = load_visual_dataset(data)
    dataset_sha = sha256_file(data)
    source = load_visual_latent_checkpoint(source_path)
    candidate = load_visual_latent_checkpoint(candidate_path)
    _validate_visual_checkpoint_pair(
        baseline=source,
        aligned=candidate,
        dataset_sha256=dataset_sha,
    )
    test_ids = source.split_episode_ids["test"]
    missing_ids = sorted(
        set(int(value) for value in test_ids.tolist())
        - set(int(value) for value in dataset["episode_ids"].tolist())
    )
    if missing_ids:
        raise ValueError(
            "checkpoint test episode IDs are missing from the dataset: "
            + ", ".join(map(str, missing_ids))
        )
    latent_frames = encode_all_frames(
        source.autoencoder,
        dataset["frames"],
        batch_size=encode_batch_size,
    )
    normalized_latent_frames = np.asarray(
        source.latent_normalizer.normalize(latent_frames),
        dtype=np.float32,
    )
    normalized_positions = normalize_object_positions(
        np.asarray(dataset["states"]),
        np.asarray(dataset["scene_world_bounds"]),
    )
    train_frame_indices = frame_indices_for_episode_ids(
        dataset,
        source.split_episode_ids["train"],
    )
    validation_frame_indices = frame_indices_for_episode_ids(
        dataset,
        source.split_episode_ids["validation"],
    )
    position_probe = fit_linear_object_position_probe(
        normalized_latent_frames[train_frame_indices],
        normalized_positions[train_frame_indices],
        ridge=_OBJECT_POSITION_PROBE_RIDGE,
    )
    position_probe_metadata: dict[str, object] = {
        "ridge": _OBJECT_POSITION_PROBE_RIDGE,
        "fit_split": "train_frames",
        "target": "normalized_xy",
        "sha256": object_position_probe_sha256(position_probe),
        "train": evaluate_object_position_probe(
            position_probe,
            normalized_latents=normalized_latent_frames[
                train_frame_indices
            ],
            normalized_positions=normalized_positions[
                train_frame_indices
            ],
            world_bounds=np.asarray(dataset["scene_world_bounds"]),
        ),
        "validation": evaluate_object_position_probe(
            position_probe,
            normalized_latents=normalized_latent_frames[
                validation_frame_indices
            ],
            normalized_positions=normalized_positions[
                validation_frame_indices
            ],
            world_bounds=np.asarray(dataset["scene_world_bounds"]),
        ),
    }
    selection = select_visual_rollout_windows(
        dataset=dataset,
        latent_frames=latent_frames,
        selected_episode_ids=test_ids,
        max_horizon=horizon_values[-1],
        windows_per_episode=int(windows_per_episode),
    )
    arrays = _stack_rollout_windows(selection.windows)
    true_factual_normalized = np.asarray(
        source.latent_normalizer.normalize(arrays["target_latents"]),
        dtype=np.float32,
    )
    true_initial_frames = _channels_first_frames(
        dataset["frames"][arrays["initial_frame_indices"]]
    )
    true_factual_frames = _channels_first_frames(
        dataset["frames"][arrays["target_frame_indices"]]
    )
    true_factual_positions = normalized_positions[
        arrays["target_frame_indices"]
    ]
    factual_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, model in (("source", source), ("candidate", candidate)):
        factual_latents = rollout_normalized_latents(
            model.dynamics,
            context_latents=arrays["context_latents"],
            history_actions=arrays["history_actions"],
            future_actions=arrays["future_actions"],
            latent_normalizer=model.latent_normalizer,
            action_normalizer=model.action_normalizer,
        )
        factual_frames = _decode_normalized_rollout_latents(
            model.autoencoder,
            normalized_latents=factual_latents,
            latent_normalizer=model.latent_normalizer,
            batch_size=decode_batch_size,
        )
        factual_predictions[name] = (factual_latents, factual_frames)
    oracle_factual_frames = _decode_normalized_rollout_latents(
        source.autoencoder,
        normalized_latents=true_factual_normalized,
        latent_normalizer=source.latent_normalizer,
        batch_size=decode_batch_size,
    )

    seed_records: dict[str, list[dict[str, object]]] = {
        "source": [],
        "candidate": [],
        "oracle": [],
    }
    batches: list[MatchedCounterfactualBatch] = []
    permutations: list[list[int]] = []
    for seed in seed_values:
        permutation = _sattolo_permutation(
            len(selection.windows),
            seed=seed,
        )
        batch = build_matched_counterfactual_batch(
            dataset,
            selection.windows,
            permutation,
        )
        batches.append(batch)
        permutations.append(
            [int(value) for value in permutation.tolist()]
        )
        true_counterfactual_normalized = (
            _encode_counterfactual_targets(
                model=source,
                batch=batch,
                batch_size=encode_batch_size,
            )
        )
        true_counterfactual_frames = _channels_first_frames(
            batch.true_frames
        )
        true_counterfactual_positions = np.zeros(
            (batch.count, batch.horizon, 2),
            dtype=np.float32,
        )
        true_counterfactual_positions[batch.valid_steps] = (
            normalize_object_positions(
                batch.true_states[batch.valid_steps],
                np.asarray(dataset["scene_world_bounds"]),
            )
        )
        oracle_counterfactual_frames = (
            _decode_normalized_rollout_latents(
                source.autoencoder,
                normalized_latents=true_counterfactual_normalized,
                latent_normalizer=source.latent_normalizer,
                batch_size=decode_batch_size,
            )
        )
        oracle_record = summarize_matched_counterfactual_predictions(
            predicted_counterfactual_normalized_latents=(
                true_counterfactual_normalized
            ),
            true_counterfactual_normalized_latents=(
                true_counterfactual_normalized
            ),
            predicted_factual_normalized_latents=(
                true_factual_normalized
            ),
            true_factual_normalized_latents=true_factual_normalized,
            predicted_counterfactual_frames=(
                oracle_counterfactual_frames
            ),
            true_counterfactual_frames=true_counterfactual_frames,
            predicted_factual_frames=oracle_factual_frames,
            true_factual_frames=true_factual_frames,
            true_initial_frames=true_initial_frames,
            position_probe=position_probe,
            true_counterfactual_normalized_positions=(
                true_counterfactual_positions
            ),
            true_factual_normalized_positions=true_factual_positions,
            world_bounds=np.asarray(dataset["scene_world_bounds"]),
            valid_steps=batch.valid_steps,
            episode_ids=batch.recipient_episode_ids,
        )
        seed_records["oracle"].append(oracle_record)
        for name, model in (
            ("source", source),
            ("candidate", candidate),
        ):
            counterfactual_latents = rollout_normalized_latents(
                model.dynamics,
                context_latents=arrays["context_latents"],
                history_actions=arrays["history_actions"],
                future_actions=batch.applied_actions,
                latent_normalizer=model.latent_normalizer,
                action_normalizer=model.action_normalizer,
            )
            counterfactual_frames = _decode_normalized_rollout_latents(
                model.autoencoder,
                normalized_latents=counterfactual_latents,
                latent_normalizer=model.latent_normalizer,
                batch_size=decode_batch_size,
            )
            factual_latents, factual_frames = factual_predictions[name]
            record = summarize_matched_counterfactual_predictions(
                predicted_counterfactual_normalized_latents=(
                    counterfactual_latents
                ),
                true_counterfactual_normalized_latents=(
                    true_counterfactual_normalized
                ),
                predicted_factual_normalized_latents=factual_latents,
                true_factual_normalized_latents=true_factual_normalized,
                predicted_counterfactual_frames=counterfactual_frames,
                true_counterfactual_frames=true_counterfactual_frames,
                predicted_factual_frames=factual_frames,
                true_factual_frames=true_factual_frames,
                true_initial_frames=true_initial_frames,
                position_probe=position_probe,
                true_counterfactual_normalized_positions=(
                    true_counterfactual_positions
                ),
                true_factual_normalized_positions=(
                    true_factual_positions
                ),
                world_bounds=np.asarray(
                    dataset["scene_world_bounds"]
                ),
                valid_steps=batch.valid_steps,
                episode_ids=batch.recipient_episode_ids,
            )
            seed_records[name].append(record)

    model_records = {
        name: aggregate_counterfactual_seed_records(
            records,
            seeds=seed_values,
        )
        for name, records in (
            ("source", seed_records["source"]),
            ("candidate", seed_records["candidate"]),
        )
    }
    oracle = aggregate_counterfactual_seed_records(
        seed_records["oracle"],
        seeds=seed_values,
    )
    comparison = _build_counterfactual_comparison(
        source=model_records["source"],
        candidate=model_records["candidate"],
    )
    decision = _build_preregistered_decision(
        source=model_records["source"],
        candidate=model_records["candidate"],
        decision_horizon=int(decision_horizon),
    )
    coverage = _aggregate_coverage(batches, seeds=seed_values)
    metrics: dict[str, object] = {
        "schema_version": 1,
        "models": model_records,
        "oracle": oracle,
        "object_position_probe": position_probe_metadata,
        "coverage": coverage,
        "comparison": {"candidate_minus_source": comparison},
        "decision": decision,
        "snapshots": {
            str(horizon): {
                "source": model_records["source"]["steps"][str(horizon)],
                "candidate": model_records["candidate"]["steps"][
                    str(horizon)
                ],
                "oracle": oracle["steps"][str(horizon)],
                "coverage": coverage["steps"][str(horizon)],
                "comparison": comparison[str(horizon)],
            }
            for horizon in horizon_values
        },
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
            "counterfactual_seeds": list(seed_values),
            "action_donation": (
                "complete-current-and-future-row-by-sattolo-permutation"
            ),
            "model_action_input": "simulator-applied-clipped-actions",
            "simulator_reconstruction": (
                "fresh-CarEnv-from-recipient-aligned-anchor-state"
            ),
            "terminal_transition": "valid",
            "post_terminal_steps": "masked",
            "aggregation": (
                "valid-windows-within-episode-then-episodes-equally-"
                "then-seeds"
            ),
            "object_position_target": "normalized_xy",
            "object_position_probe_fit_split": "train_frames",
            "object_position_probe_ridge": (
                _OBJECT_POSITION_PROBE_RIDGE
            ),
            "object_position_probe_sha256": (
                position_probe_metadata["sha256"]
            ),
        },
        "test_episode_ids": [
            int(value) for value in test_ids.tolist()
        ],
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
        "branches": {
            "donor_permutations": permutations,
            "clipped_action_steps": coverage["clipped_action_steps"],
            "terminal_reasons": coverage["terminal_reasons"],
        },
        "decision_gates": [
            {
                "name": gate["name"],
                "horizon": gate["horizon"],
                "metric": gate["metric"],
                "operator": gate["operator"],
            }
            for gate in decision["gates"]
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
        plot_matched_counterfactual_comparison(
            metrics=metrics,
            output_path=(
                staging / "matched_counterfactual_comparison.png"
            ),
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
        "plot": str(
            output / "matched_counterfactual_comparison.png"
        ),
        "horizons": list(horizon_values),
        "windows": len(selection.windows),
        "eligible_episodes": int(selection.eligible_episode_ids.size),
        "candidate_passes": decision["candidate_passes"],
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--candidate-checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 5, 10],
    )
    parser.add_argument(
        "--windows-per-episode",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--counterfactual-seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
    )
    parser.add_argument("--decision-horizon", type=int, default=5)
    return parser


def main() -> None:
    parser = _build_argument_parser()
    arguments = parser.parse_args()
    try:
        summary = run_visual_counterfactual_diagnostics(
            data_path=arguments.data,
            source_checkpoint_path=arguments.source_checkpoint,
            candidate_checkpoint_path=arguments.candidate_checkpoint,
            output_dir=arguments.output_dir,
            horizons=arguments.horizons,
            windows_per_episode=arguments.windows_per_episode,
            counterfactual_seeds=arguments.counterfactual_seeds,
            decision_horizon=arguments.decision_horizon,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
