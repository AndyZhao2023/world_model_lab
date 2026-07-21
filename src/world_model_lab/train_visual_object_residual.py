"""Train an object-only residual decoder over a frozen spatial world model."""

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
from .visual_dataset import load_visual_dataset
from .visual_latent_data import (
    build_latent_window_arrays,
    encode_all_frames,
)
from .visual_latent_model import (
    SpatialConvAutoencoder,
    SpatialLatentDynamicsCNN,
)
from .visual_object_residual import (
    evaluate_object_residual_mask,
    train_object_residual_decoder,
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


def run_visual_object_residual_training(
    *,
    data_path: Path | str,
    source_checkpoint_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
    head_channels: int = 16,
    initial_alpha: float = 0.01,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    foreground_loss_weight: float = 1.0,
    mask_loss_weight: float = 0.01,
) -> dict[str, Any]:
    """Train and publish one frozen-base object residual candidate."""

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
    data = Path(data_path)
    source_path = Path(source_checkpoint_path)
    dataset = load_visual_dataset(data)
    source = load_visual_latent_checkpoint(source_path)
    if (
        not isinstance(source.autoencoder, SpatialConvAutoencoder)
        or not isinstance(source.dynamics, SpatialLatentDynamicsCNN)
        or source.autoencoder.object_residual_decoder
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
    autoencoder_result = train_object_residual_decoder(
        source.autoencoder,
        dataset,
        split_episode_ids=source.split_episode_ids,
        head_channels=head_channels,
        initial_alpha=initial_alpha,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        foreground_loss_weight=foreground_weight,
        mask_loss_weight=mask_weight,
        seed=seed,
    )
    autoencoder = autoencoder_result.model
    if not isinstance(autoencoder, SpatialConvAutoencoder):
        raise RuntimeError("residual trainer returned an invalid model")
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
        raise RuntimeError("source dynamics changed during residual training")

    reconstruction_metrics = evaluate_autoencoder(
        autoencoder,
        dataset,
        selected_episode_ids=source.split_episode_ids["test"],
        batch_size=batch_size,
    )
    mask_metrics = evaluate_object_residual_mask(
        autoencoder,
        dataset,
        selected_episode_ids=source.split_episode_ids["test"],
        batch_size=batch_size,
    )
    autoencoder_test_metrics = {
        **reconstruction_metrics,
        **{
            f"residual_{name}": value
            for name, value in mask_metrics.items()
            if name not in {"frames", "object_pixels", "background_pixels"}
        },
    }
    latent_frames = encode_all_frames(
        autoencoder,
        dataset["frames"],
        batch_size=batch_size,
    )
    test_arrays = build_latent_window_arrays(
        dataset,
        build_visual_window_index(
            dataset,
            source.split_episode_ids["test"],
        ),
        latent_frames,
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
            "autoencoder_object_residual_decoder": True,
            "object_head_channels": head_channels,
            "object_initial_alpha": initial_alpha,
            "object_full_frame_loss_weight": 1.0,
            "object_foreground_loss_weight": foreground_weight,
            "object_mask_loss_weight": mask_weight,
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
    return {
        "dataset": dataset_metadata,
        "source_checkpoint": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
        },
        "autoencoder": {
            "initial_train_loss": autoencoder_result.train_losses[0],
            "final_train_loss": autoencoder_result.train_losses[-1],
            "best_epoch": autoencoder_result.best_epoch,
            "best_validation_loss": min(
                autoencoder_result.validation_losses
            ),
            "test": reconstruction_metrics,
            "mask": mask_metrics,
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
        default=Path(
            "artifacts/visual_latent_spatial8_object_residual.pt"
        ),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(
            "artifacts/"
            "visual_latent_spatial8_object_residual_predictions.png"
        ),
    )
    parser.add_argument("--head-channels", type=int, default=16)
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
    arguments = parser.parse_args()
    try:
        summary = run_visual_object_residual_training(
            data_path=arguments.data,
            source_checkpoint_path=arguments.source_checkpoint,
            output_path=arguments.output,
            preview_path=arguments.preview,
            head_channels=arguments.head_channels,
            initial_alpha=arguments.initial_alpha,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            foreground_loss_weight=arguments.foreground_loss_weight,
            mask_loss_weight=arguments.mask_loss_weight,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
