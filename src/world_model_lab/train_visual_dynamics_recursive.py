"""Train visual latent dynamics with an H-step recursive rollout loss."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .diagnose_model import sha256_file
from .train_visual_dynamics_objective import (
    _positive_source_float,
    _positive_source_int,
    _source_seed,
)
from .train_visual_latent_model import (
    PhaseTrainingResult,
    _preflight_output_paths,
    evaluate_latent_dynamics,
    load_visual_latent_checkpoint,
    plot_visual_latent_predictions,
    save_visual_latent_checkpoint,
)
from .visual_dataset import load_visual_dataset
from .visual_latent_data import (
    build_latent_rollout_arrays,
    build_latent_window_arrays,
    encode_all_frames,
)
from .visual_latent_model import (
    SpatialConvAutoencoder,
    SpatialLatentDynamicsCNN,
)
from .visual_recursive_training import train_recursive_latent_dynamics
from .visual_windows import build_visual_window_index


def _validate_rollout_protocol(
    *,
    rollout_horizon: int,
    rollout_loss_weight: float,
    changed_pixel_loss_weight: float,
) -> tuple[int, float, float]:
    if (
        isinstance(rollout_horizon, bool)
        or not isinstance(rollout_horizon, int)
        or rollout_horizon <= 1
    ):
        raise ValueError("rollout_horizon must be an integer greater than one")
    if (
        not math.isfinite(rollout_loss_weight)
        or rollout_loss_weight < 0.0
    ):
        raise ValueError(
            "rollout_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(changed_pixel_loss_weight)
        or changed_pixel_loss_weight < 0.0
    ):
        raise ValueError(
            "changed_pixel_loss_weight must be finite and non-negative"
        )
    return (
        int(rollout_horizon),
        float(rollout_loss_weight),
        float(changed_pixel_loss_weight),
    )


def run_recursive_dynamics_training(
    *,
    data_path: Path | str,
    source_checkpoint_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
    changed_pixel_loss_weight: float = 0.1,
    rollout_horizon: int = 5,
    rollout_loss_weight: float = 1.0,
    dynamics_epochs: int | None = None,
    dynamics_batch_size: int | None = None,
) -> dict[str, Any]:
    """Train fresh CNN dynamics with one-step and recursive objectives."""

    output = Path(output_path)
    preview = Path(preview_path)
    _preflight_output_paths(output, preview)
    horizon, rollout_weight, changed_weight = _validate_rollout_protocol(
        rollout_horizon=rollout_horizon,
        rollout_loss_weight=rollout_loss_weight,
        changed_pixel_loss_weight=changed_pixel_loss_weight,
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
    selected_epochs = _positive_source_int(
        config,
        "dynamics_epochs",
        dynamics_epochs,
    )
    selected_batch_size = _positive_source_int(
        config,
        "dynamics_batch_size",
        dynamics_batch_size,
    )
    learning_rate = _positive_source_float(
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
    one_step_arrays = {
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
    for name, arrays in one_step_arrays.items():
        if arrays.count == 0:
            raise ValueError(f"{name} split has no eligible visual windows")
    rollout_arrays = {}
    for name in ("train", "validation", "test"):
        try:
            rollout_arrays[name] = build_latent_rollout_arrays(
                dataset,
                source.split_episode_ids[name],
                latent_frames,
                horizon=horizon,
            )
        except ValueError as error:
            if "not long enough" not in str(error):
                raise
            raise ValueError(
                f"{name} split has no eligible H{horizon} rollout windows"
            ) from error

    dynamics_result = train_recursive_latent_dynamics(
        one_step_arrays["train"],
        one_step_arrays["validation"],
        rollout_arrays["train"],
        rollout_arrays["validation"],
        latent_normalizer=source.latent_normalizer,
        action_normalizer=source.action_normalizer,
        spatial_latent_channels=autoencoder.latent_channels,
        hidden_size=source.dynamics.hidden_size,
        epochs=selected_epochs,
        batch_size=selected_batch_size,
        learning_rate=learning_rate,
        seed=seed,
        decoder=autoencoder,
        visual_dataset=dataset,
        changed_pixel_loss_weight=changed_weight,
        rollout_loss_weight=rollout_weight,
    )
    dynamics = dynamics_result.model
    assert isinstance(dynamics, SpatialLatentDynamicsCNN)
    test_metrics = evaluate_latent_dynamics(
        dynamics,
        autoencoder,
        dataset,
        one_step_arrays["test"],
        latent_normalizer=source.latent_normalizer,
        action_normalizer=source.action_normalizer,
        batch_size=selected_batch_size,
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
            "dynamics_epochs": selected_epochs,
            "dynamics_batch_size": selected_batch_size,
            "dynamics_learning_rate": learning_rate,
            "seed": seed,
            "source_checkpoint": str(source_path.resolve()),
            "source_checkpoint_sha256": source_sha256,
            "autoencoder_frozen": True,
            "dynamics_reinitialized": True,
            "dynamics_loss": "one_step_plus_recursive_rollout",
            "dynamics_changed_pixel_loss_weight": changed_weight,
            "dynamics_rollout_horizon": horizon,
            "dynamics_rollout_loss_weight": rollout_weight,
            "dynamics_train_one_step_losses": list(
                dynamics_result.train_one_step_losses
            ),
            "dynamics_train_rollout_losses": list(
                dynamics_result.train_rollout_losses
            ),
            "dynamics_validation_one_step_losses": list(
                dynamics_result.validation_one_step_losses
            ),
            "dynamics_validation_rollout_losses": list(
                dynamics_result.validation_rollout_losses
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
        dynamics_test_metrics=test_metrics,
    )
    try:
        plot_visual_latent_predictions(
            preview,
            autoencoder=autoencoder,
            dynamics=dynamics,
            dataset=dataset,
            arrays=one_step_arrays["test"],
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

    best_index = dynamics_result.best_epoch - 1
    return {
        "dataset": dataset_metadata,
        "source_checkpoint": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
        },
        "one_step_windows": {
            name: arrays.count for name, arrays in one_step_arrays.items()
        },
        "rollout_windows": {
            name: arrays.count for name, arrays in rollout_arrays.items()
        },
        "dynamics": {
            "loss": training_config["dynamics_loss"],
            "changed_pixel_loss_weight": changed_weight,
            "rollout_horizon": horizon,
            "rollout_loss_weight": rollout_weight,
            "initial_train_loss": dynamics_result.train_losses[0],
            "final_train_loss": dynamics_result.train_losses[-1],
            "best_epoch": dynamics_result.best_epoch,
            "best_validation_loss": (
                dynamics_result.validation_losses[best_index]
            ),
            "best_validation_one_step_loss": (
                dynamics_result.validation_one_step_losses[best_index]
            ),
            "best_validation_rollout_loss": (
                dynamics_result.validation_rollout_losses[best_index]
            ),
            "test": test_metrics,
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
            "artifacts/visual_latent_spatial8_objective_w01_h5.pt"
        ),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(
            "artifacts/"
            "visual_latent_spatial8_objective_w01_h5_predictions.png"
        ),
    )
    parser.add_argument(
        "--changed-pixel-loss-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument("--dynamics-epochs", type=int)
    parser.add_argument("--dynamics-batch-size", type=int)
    arguments = parser.parse_args()
    try:
        summary = run_recursive_dynamics_training(
            data_path=arguments.data,
            source_checkpoint_path=arguments.source_checkpoint,
            output_path=arguments.output,
            preview_path=arguments.preview,
            changed_pixel_loss_weight=(
                arguments.changed_pixel_loss_weight
            ),
            rollout_horizon=arguments.rollout_horizon,
            rollout_loss_weight=arguments.rollout_loss_weight,
            dynamics_epochs=arguments.dynamics_epochs,
            dynamics_batch_size=arguments.dynamics_batch_size,
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
