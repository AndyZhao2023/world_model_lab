"""Retrain spatial latent dynamics with frozen-decoder image supervision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .diagnose_model import sha256_file
from .train_visual_latent_model import (
    PhaseTrainingResult,
    _preflight_output_paths,
    evaluate_latent_dynamics,
    load_visual_latent_checkpoint,
    plot_visual_latent_predictions,
    save_visual_latent_checkpoint,
    train_latent_dynamics,
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
from .visual_windows import build_visual_window_index


def _positive_source_int(
    training_config: dict[str, Any],
    name: str,
    override: int | None,
) -> int:
    value = training_config.get(name) if override is None else override
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer") from None
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_source_float(
    training_config: dict[str, Any],
    name: str,
) -> float:
    try:
        value = float(training_config.get(name))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite and positive") from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _source_seed(training_config: dict[str, Any]) -> int:
    value = training_config.get("seed")
    if isinstance(value, bool):
        raise ValueError("source seed must be a non-negative integer")
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "source seed must be a non-negative integer"
        ) from None
    if seed < 0 or seed != value:
        raise ValueError("source seed must be a non-negative integer")
    return seed


def run_frozen_decoder_dynamics_training(
    *,
    data_path: Path | str,
    source_checkpoint_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
    changed_pixel_loss_weight: float = 0.1,
    dynamics_epochs: int | None = None,
    dynamics_batch_size: int | None = None,
) -> dict[str, Any]:
    """Train fresh spatial CNN dynamics while freezing source representation."""

    output = Path(output_path)
    preview = Path(preview_path)
    _preflight_output_paths(output, preview)
    if (
        not math.isfinite(changed_pixel_loss_weight)
        or changed_pixel_loss_weight < 0.0
    ):
        raise ValueError(
            "changed_pixel_loss_weight must be finite and non-negative"
        )

    data = Path(data_path)
    source_path = Path(source_checkpoint_path)
    dataset = load_visual_dataset(data)
    source = load_visual_latent_checkpoint(source_path)
    if not isinstance(source.autoencoder, SpatialConvAutoencoder) or not (
        isinstance(source.dynamics, SpatialLatentDynamicsCNN)
    ):
        raise ValueError(
            "source checkpoint must contain a spatial CNN world model"
        )

    dataset_sha256 = sha256_file(data)
    if source.dataset_metadata.get("sha256") != dataset_sha256:
        raise ValueError(
            "source checkpoint dataset SHA-256 does not match data"
        )

    config = source.training_config
    autoencoder_batch_size = _positive_source_int(
        config,
        "autoencoder_batch_size",
        None,
    )
    selected_dynamics_epochs = _positive_source_int(
        config,
        "dynamics_epochs",
        dynamics_epochs,
    )
    selected_dynamics_batch_size = _positive_source_int(
        config,
        "dynamics_batch_size",
        dynamics_batch_size,
    )
    dynamics_learning_rate = _positive_source_float(
        config,
        "dynamics_learning_rate",
    )
    seed = _source_seed(config)

    autoencoder = source.autoencoder
    latent_frames = encode_all_frames(
        autoencoder,
        dataset["frames"],
        batch_size=autoencoder_batch_size,
    )
    window_arrays = {
        name: build_latent_window_arrays(
            dataset,
            build_visual_window_index(
                dataset,
                source.split_episode_ids[name],
            ),
            latent_frames,
        )
        for name in ("train", "validation", "test")
    }
    for name, arrays in window_arrays.items():
        if arrays.count == 0:
            raise ValueError(f"{name} split has no eligible visual windows")

    dynamics_result = train_latent_dynamics(
        window_arrays["train"],
        window_arrays["validation"],
        latent_normalizer=source.latent_normalizer,
        action_normalizer=source.action_normalizer,
        latent_layout="spatial",
        spatial_latent_channels=autoencoder.latent_channels,
        spatial_dynamics_architecture="cnn",
        hidden_size=source.dynamics.hidden_size,
        epochs=selected_dynamics_epochs,
        batch_size=selected_dynamics_batch_size,
        learning_rate=dynamics_learning_rate,
        seed=seed,
        decoder=autoencoder,
        visual_dataset=dataset,
        changed_pixel_loss_weight=changed_pixel_loss_weight,
    )
    dynamics = dynamics_result.model
    assert isinstance(dynamics, SpatialLatentDynamicsCNN)
    dynamics_test_metrics = evaluate_latent_dynamics(
        dynamics,
        autoencoder,
        dataset,
        window_arrays["test"],
        latent_normalizer=source.latent_normalizer,
        action_normalizer=source.action_normalizer,
        batch_size=selected_dynamics_batch_size,
    )
    autoencoder_result = PhaseTrainingResult(
        model=autoencoder,
        train_losses=list(source.autoencoder_history["train_losses"]),
        validation_losses=list(
            source.autoencoder_history["validation_losses"]
        ),
        best_epoch=int(source.autoencoder_history["best_epoch"]),
    )
    source_sha256 = sha256_file(source_path)
    training_config = dict(config)
    training_config.update(
        {
            "data_path": str(data.resolve()),
            "device": "cpu",
            "latent_layout": "spatial",
            "latent_dim": autoencoder.latent_dim,
            "spatial_latent_channels": autoencoder.latent_channels,
            "spatial_dynamics_architecture": "cnn",
            "dynamics_hidden_size": dynamics.hidden_size,
            "dynamics_epochs": selected_dynamics_epochs,
            "dynamics_batch_size": selected_dynamics_batch_size,
            "dynamics_learning_rate": dynamics_learning_rate,
            "seed": seed,
            "source_checkpoint": str(source_path.resolve()),
            "source_checkpoint_sha256": source_sha256,
            "autoencoder_frozen": True,
            "dynamics_loss": (
                "normalized_latent_mse_plus_changed_pixel_mae"
                if changed_pixel_loss_weight > 0.0
                else "normalized_latent_mse"
            ),
            "dynamics_changed_pixel_loss_weight": (
                changed_pixel_loss_weight
            ),
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
        autoencoder_test_metrics=source.autoencoder_test_metrics,
        dynamics_test_metrics=dynamics_test_metrics,
    )
    try:
        plot_visual_latent_predictions(
            preview,
            autoencoder=autoencoder,
            dynamics=dynamics,
            dataset=dataset,
            arrays=window_arrays["test"],
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
        "split_windows": {
            name: arrays.count for name, arrays in window_arrays.items()
        },
        "dynamics": {
            "loss": training_config["dynamics_loss"],
            "changed_pixel_loss_weight": changed_pixel_loss_weight,
            "initial_train_loss": dynamics_result.train_losses[0],
            "final_train_loss": dynamics_result.train_losses[-1],
            "best_epoch": dynamics_result.best_epoch,
            "best_validation_loss": min(
                dynamics_result.validation_losses
            ),
            "test": dynamics_test_metrics,
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
        default=Path("artifacts/visual_latent_spatial8.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/visual_latent_spatial8_objective_w01.pt"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(
            "artifacts/visual_latent_spatial8_objective_w01_predictions.png"
        ),
    )
    parser.add_argument(
        "--changed-pixel-loss-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument("--dynamics-epochs", type=int)
    parser.add_argument("--dynamics-batch-size", type=int)
    args = parser.parse_args()
    try:
        summary = run_frozen_decoder_dynamics_training(
            data_path=args.data,
            source_checkpoint_path=args.source_checkpoint,
            output_path=args.output,
            preview_path=args.preview,
            changed_pixel_loss_weight=args.changed_pixel_loss_weight,
            dynamics_epochs=args.dynamics_epochs,
            dynamics_batch_size=args.dynamics_batch_size,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
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
