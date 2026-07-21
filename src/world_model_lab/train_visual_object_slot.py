"""Train a structured object-slot decoder with an exactly local image write."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .diagnose_model import sha256_file
from .train_visual_dynamics_objective import _source_seed
from .train_visual_latent_model import (
    PhaseTrainingResult,
    _preflight_output_paths,
    evaluate_autoencoder,
    load_visual_latent_checkpoint,
    plot_visual_latent_predictions,
    save_visual_latent_checkpoint,
)
from .visual_dataset import IMAGE_SIZE, load_visual_dataset
from .visual_latent_data import (
    build_latent_window_arrays,
    encode_all_frames,
    frame_indices_for_episode_ids,
)
from .visual_latent_model import (
    SpatialConvAutoencoder,
    SpatialLatentDynamicsCNN,
)
from .visual_object_slot import (
    evaluate_object_slot_decoder,
    normalize_object_slot_targets,
    normalized_affine_to_raw,
    train_object_slot_decoder,
)
from .visual_windows import build_visual_window_index


def _require_finite_weight(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _same_module_state(
    left: torch.nn.Module,
    right: torch.nn.Module,
) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name])
        for name in left_state
    )


def _state_metrics(
    predicted_slots: np.ndarray,
    target_slots: np.ndarray,
) -> dict[str, float | int]:
    predicted = np.asarray(predicted_slots, dtype=np.float64)
    target = np.asarray(target_slots, dtype=np.float64)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 4
        or predicted.shape[0] == 0
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("slot arrays must be finite matching [N, 4]")
    centre_errors = np.linalg.norm(
        predicted[:, :2] - target[:, :2],
        axis=1,
    ) * ((IMAGE_SIZE - 1) / 2.0)
    predicted_heading = predicted[:, 2:]
    heading_norms = np.linalg.norm(
        predicted_heading,
        axis=1,
        keepdims=True,
    )
    predicted_heading = predicted_heading / np.maximum(
        heading_norms,
        1e-12,
    )
    heading_cosines = np.clip(
        np.sum(predicted_heading * target[:, 2:], axis=1),
        -1.0,
        1.0,
    )
    heading_errors = np.degrees(np.arccos(heading_cosines))
    return {
        "frames": int(target.shape[0]),
        "mean_centre_error_pixels": float(np.mean(centre_errors)),
        "p95_centre_error_pixels": float(
            np.quantile(centre_errors, 0.95)
        ),
        "mean_heading_error_degrees": float(np.mean(heading_errors)),
        "p95_heading_error_degrees": float(
            np.quantile(heading_errors, 0.95)
        ),
    }


def _fit_source_probe(
    normalized_latents: np.ndarray,
    targets: np.ndarray,
    *,
    train_indices: np.ndarray,
    split_indices: dict[str, np.ndarray],
    ridge: float,
) -> tuple[dict[str, Any], np.ndarray]:
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("source_probe_ridge must be finite and non-negative")
    latents = np.asarray(normalized_latents, dtype=np.float64)
    slot_targets = np.asarray(targets, dtype=np.float64)
    design = np.concatenate(
        (
            latents[train_indices],
            np.ones((len(train_indices), 1), dtype=np.float64),
        ),
        axis=1,
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    try:
        solution = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ slot_targets[train_indices],
        )
    except np.linalg.LinAlgError:
        raise ValueError("source object-slot probe ridge system is singular") from None
    if not np.all(np.isfinite(solution)):
        raise ValueError("source object-slot probe is non-finite")
    metrics: dict[str, Any] = {
        "ridge": ridge,
        "fit_split": "train_frames",
        "target": "image_normalized_cx_cy_sin_cos",
    }
    for name, indices in split_indices.items():
        split_design = np.concatenate(
            (
                latents[indices],
                np.ones((len(indices), 1), dtype=np.float64),
            ),
            axis=1,
        )
        metrics[name] = _state_metrics(
            split_design @ solution,
            slot_targets[indices],
        )
    return metrics, solution


def run_visual_object_slot_training(
    *,
    data_path: Path | str,
    source_checkpoint_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
    patch_size: int = 11,
    hidden_size: int = 64,
    initial_alpha: float = 0.01,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    foreground_loss_weight: float = 1.0,
    mask_loss_weight: float = 0.01,
    centre_loss_weight: float = 1.0,
    heading_loss_weight: float = 0.1,
    source_probe_ridge: float = 1e-3,
    locator: str = "spatial_attention",
) -> dict[str, Any]:
    """Train and publish one frozen-base object-slot candidate."""

    output = Path(output_path)
    preview = Path(preview_path)
    _preflight_output_paths(output, preview)
    foreground_weight = _require_finite_weight(
        "foreground_loss_weight",
        foreground_loss_weight,
    )
    mask_weight = _require_finite_weight(
        "mask_loss_weight",
        mask_loss_weight,
    )
    centre_loss = _require_finite_weight(
        "centre_loss_weight",
        centre_loss_weight,
    )
    heading_weight = _require_finite_weight(
        "heading_loss_weight",
        heading_loss_weight,
    )
    ridge = _require_finite_weight(
        "source_probe_ridge",
        source_probe_ridge,
    )
    if locator not in {"spatial_attention", "global_affine"}:
        raise ValueError(
            "locator must be 'spatial_attention' or 'global_affine'"
        )
    data = Path(data_path)
    source_path = Path(source_checkpoint_path)
    dataset = load_visual_dataset(data)
    source = load_visual_latent_checkpoint(source_path)
    if (
        not isinstance(source.autoencoder, SpatialConvAutoencoder)
        or not isinstance(source.dynamics, SpatialLatentDynamicsCNN)
        or source.autoencoder.object_residual_decoder
        or source.autoencoder.object_slot_decoder
    ):
        raise ValueError(
            "source checkpoint must contain a plain spatial CNN world model"
        )
    dataset_sha256 = sha256_file(data)
    if source.dataset_metadata.get("sha256") != dataset_sha256:
        raise ValueError(
            "source checkpoint dataset SHA-256 does not match data"
        )
    seed = _source_seed(source.training_config)
    source_dynamics_state = {
        name: tensor.detach().clone()
        for name, tensor in source.dynamics.state_dict().items()
    }
    source_latents = encode_all_frames(
        source.autoencoder,
        dataset["frames"],
        batch_size=batch_size,
    )
    normalized_source_latents = np.asarray(
        source.latent_normalizer.normalize(source_latents),
        dtype=np.float32,
    )
    slot_targets = normalize_object_slot_targets(
        np.asarray(dataset["states"]),
        np.asarray(dataset["scene_world_bounds"]),
    )
    split_frame_indices = {
        name: frame_indices_for_episode_ids(
            dataset,
            source.split_episode_ids[name],
        )
        for name in ("train", "validation", "test")
    }
    source_probe, probe_solution = _fit_source_probe(
        normalized_source_latents,
        slot_targets,
        train_indices=split_frame_indices["train"],
        split_indices=split_frame_indices,
        ridge=ridge,
    )
    centre_locator_weight: np.ndarray | None = None
    centre_locator_bias: np.ndarray | None = None
    centre_probe_conversion: dict[str, float] | None = None
    if locator == "global_affine":
        centre_locator_weight, centre_locator_bias = (
            normalized_affine_to_raw(
                probe_solution[:-1, :2].T,
                probe_solution[-1, :2],
                np.asarray(source.latent_normalizer.mean),
                np.asarray(source.latent_normalizer.std),
            )
        )
        test_indices = split_frame_indices["test"]
        normalized_predictions = (
            normalized_source_latents[test_indices]
            @ probe_solution[:-1, :2]
            + probe_solution[-1, :2]
        )
        raw_predictions = (
            source_latents[test_indices] @ centre_locator_weight.T
            + centre_locator_bias
        )
        centre_probe_conversion = {
            "max_prediction_delta": float(
                np.max(
                    np.abs(
                        normalized_predictions.astype(np.float64)
                        - raw_predictions.astype(np.float64)
                    )
                )
            ),
            "raw_weight_abs_max": float(
                np.max(np.abs(centre_locator_weight))
            ),
            "raw_bias_abs_max": float(
                np.max(np.abs(centre_locator_bias))
            ),
        }

    autoencoder_result = train_object_slot_decoder(
        source.autoencoder,
        dataset,
        split_episode_ids=source.split_episode_ids,
        locator=locator,
        centre_weight=centre_locator_weight,
        centre_bias=centre_locator_bias,
        patch_size=patch_size,
        hidden_size=hidden_size,
        initial_alpha=initial_alpha,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        foreground_loss_weight=foreground_weight,
        mask_loss_weight=mask_weight,
        centre_loss_weight=centre_loss,
        heading_loss_weight=heading_weight,
        seed=seed,
    )
    autoencoder = autoencoder_result.model
    if not isinstance(autoencoder, SpatialConvAutoencoder):
        raise RuntimeError("object-slot trainer returned an invalid model")
    if autoencoder.object_slot_locator != locator:
        raise RuntimeError("object-slot trainer changed locator mode")
    if locator == "global_affine":
        assert (
            centre_locator_weight is not None
            and centre_locator_bias is not None
        )
        assert autoencoder.object_center is not None
        if not np.array_equal(
            autoencoder.object_center.weight.detach().cpu().numpy(),
            centre_locator_weight.astype(np.float32),
        ) or not np.array_equal(
            autoencoder.object_center.bias.detach().cpu().numpy(),
            centre_locator_bias.astype(np.float32),
        ):
            raise RuntimeError("frozen affine centre locator changed")
    for module_name in ("encoder_convolutions", "decoder_convolutions"):
        if not _same_module_state(
            getattr(autoencoder, module_name),
            getattr(source.autoencoder, module_name),
        ):
            raise RuntimeError(f"frozen {module_name} changed during training")
    if source.dynamics.state_dict().keys() != source_dynamics_state.keys() or any(
        not torch.equal(tensor, source_dynamics_state[name])
        for name, tensor in source.dynamics.state_dict().items()
    ):
        raise RuntimeError("source dynamics changed during object-slot training")

    reconstruction_metrics = evaluate_autoencoder(
        autoencoder,
        dataset,
        selected_episode_ids=source.split_episode_ids["test"],
        batch_size=batch_size,
    )
    slot_metrics = evaluate_object_slot_decoder(
        autoencoder,
        dataset,
        selected_episode_ids=source.split_episode_ids["test"],
        batch_size=batch_size,
    )
    autoencoder_test_metrics = {
        **reconstruction_metrics,
        **{
            f"slot_{name}": value
            for name, value in slot_metrics.items()
            if name not in {"frames", "object_pixels", "background_pixels"}
        },
    }
    candidate_latents = encode_all_frames(
        autoencoder,
        dataset["frames"],
        batch_size=batch_size,
    )
    if not np.array_equal(candidate_latents, source_latents):
        raise RuntimeError("frozen encoder changed object-slot latent frames")
    test_arrays = build_latent_window_arrays(
        dataset,
        build_visual_window_index(
            dataset,
            source.split_episode_ids["test"],
        ),
        candidate_latents,
    )
    if test_arrays.count == 0:
        raise ValueError("test split has no eligible visual windows")

    dynamics_result = PhaseTrainingResult(
        model=source.dynamics,
        train_losses=list(source.dynamics_history["train_losses"]),
        validation_losses=list(
            source.dynamics_history["validation_losses"]
        ),
        best_epoch=int(source.dynamics_history["best_epoch"]),
    )
    source_sha256 = sha256_file(source_path)
    training_config = dict(source.training_config)
    training_config.update(
        {
            "data_path": str(data.resolve()),
            "device": "cpu",
            "latent_layout": "spatial",
            "latent_dim": autoencoder.latent_dim,
            "spatial_latent_channels": autoencoder.latent_channels,
            "base_channels": autoencoder.base_channels,
            "autoencoder_epochs": epochs,
            "autoencoder_batch_size": batch_size,
            "autoencoder_learning_rate": learning_rate,
            "seed": seed,
            "source_checkpoint": str(source_path.resolve()),
            "source_checkpoint_sha256": source_sha256,
            "autoencoder_encoder_frozen": True,
            "autoencoder_base_decoder_frozen": True,
            "autoencoder_object_slot_decoder": True,
            "object_slot_patch_size": patch_size,
            "object_slot_hidden_size": hidden_size,
            "object_slot_locator": locator,
            "object_initial_alpha": initial_alpha,
            "object_full_frame_loss_weight": 1.0,
            "object_foreground_loss_weight": foreground_weight,
            "object_mask_loss_weight": mask_weight,
            "object_centre_loss_weight": centre_loss,
            "object_heading_loss_weight": heading_weight,
            "object_slot_source_probe_ridge": ridge,
            "object_slot_target": "image_normalized_cx_cy_sin_cos",
            "object_mask_target": (
                "exact_renderer_car_or_heading_colour"
            ),
            "dynamics_reused_unmodified": True,
        }
    )
    dataset_metadata = {
        "path": str(data.resolve()),
        "sha256": dataset_sha256,
        "schema_version": int(dataset["schema_version"].item()),
        "renderer_version": str(dataset["renderer_version"].item()),
    }
    save_visual_latent_checkpoint(
        output,
        autoencoder_result=autoencoder_result,
        dynamics_result=dynamics_result,
        latent_normalizer=source.latent_normalizer,
        action_normalizer=source.action_normalizer,
        split_episode_ids=source.split_episode_ids,
        training_config=training_config,
        dataset_metadata=dataset_metadata,
        autoencoder_test_metrics=autoencoder_test_metrics,
        dynamics_test_metrics=source.dynamics_test_metrics,
    )
    try:
        plot_visual_latent_predictions(
            preview,
            autoencoder=autoencoder,
            dynamics=source.dynamics,
            dataset=dataset,
            arrays=test_arrays,
            latent_normalizer=source.latent_normalizer,
            action_normalizer=source.action_normalizer,
        )
    except Exception:
        try:
            output.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise RuntimeError(
                "preview publication failed and checkpoint rollback failed: "
                f"{output}"
            ) from cleanup_error
        raise

    centre_limit = source_probe["test"]["mean_centre_error_pixels"]
    centre_operator = "<"
    centre_name = "held_out_centre_error_improvement"
    if locator == "global_affine":
        centre_limit *= 1.05
        centre_operator = "<="
        centre_name = "held_out_centre_error_stability"
    centre_candidate = slot_metrics["mean_centre_error_pixels"]
    centre_passed = (
        centre_candidate <= centre_limit
        if centre_operator == "<="
        else centre_candidate < centre_limit
    )
    state_gates = [
        {
            "name": centre_name,
            "metric": "mean_centre_error_pixels",
            "operator": centre_operator,
            "limit": centre_limit,
            "candidate": centre_candidate,
            "passed": centre_passed,
        },
        {
            "name": "held_out_heading_error_improvement",
            "metric": "mean_heading_error_degrees",
            "operator": "<",
            "limit": source_probe["test"]["mean_heading_error_degrees"],
            "candidate": slot_metrics["mean_heading_error_degrees"],
            "passed": (
                slot_metrics["mean_heading_error_degrees"]
                < source_probe["test"]["mean_heading_error_degrees"]
            ),
        },
    ]
    autoencoder_summary: dict[str, Any] = {
        "initial_train_loss": autoencoder_result.train_losses[0],
        "final_train_loss": autoencoder_result.train_losses[-1],
        "best_epoch": autoencoder_result.best_epoch,
        "best_validation_loss": min(
            autoencoder_result.validation_losses
        ),
        "test": reconstruction_metrics,
        "source_probe": source_probe,
        "slot": slot_metrics,
    }
    if centre_probe_conversion is not None:
        autoencoder_summary["centre_probe_conversion"] = (
            centre_probe_conversion
        )
    return {
        "dataset": dataset_metadata,
        "source_checkpoint": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
        },
        "autoencoder": autoencoder_summary,
        "decision": {
            "state_gates": state_gates,
            "state_gates_passed": all(
                bool(gate["passed"]) for gate in state_gates
            ),
            "representation_decision_pending_autoencoder_diagnostic": True,
        },
        "checkpoint": str(output),
        "preview": str(preview),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/visual_episodes.npz"),
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path(
            "artifacts/visual_latent_spatial8_objective_w01.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/visual_latent_spatial8_object_slot.pt"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(
            "artifacts/visual_latent_spatial8_object_slot_predictions.png"
        ),
    )
    parser.add_argument(
        "--locator",
        choices=("spatial_attention", "global_affine"),
        default="spatial_attention",
    )
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--initial-alpha", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--foreground-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--mask-loss-weight", type=float, default=0.01)
    parser.add_argument("--centre-loss-weight", type=float, default=1.0)
    parser.add_argument("--heading-loss-weight", type=float, default=0.1)
    parser.add_argument("--source-probe-ridge", type=float, default=1e-3)
    arguments = parser.parse_args()
    try:
        summary = run_visual_object_slot_training(
            data_path=arguments.data,
            source_checkpoint_path=arguments.source_checkpoint,
            output_path=arguments.output,
            preview_path=arguments.preview,
            patch_size=arguments.patch_size,
            hidden_size=arguments.hidden_size,
            initial_alpha=arguments.initial_alpha,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            foreground_loss_weight=arguments.foreground_loss_weight,
            mask_loss_weight=arguments.mask_loss_weight,
            centre_loss_weight=arguments.centre_loss_weight,
            heading_loss_weight=arguments.heading_loss_weight,
            source_probe_ridge=arguments.source_probe_ridge,
            locator=arguments.locator,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
