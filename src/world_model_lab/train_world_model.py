"""Train and evaluate a one-step state-delta world model."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import (
    Normalizer,
    build_model_arrays,
    fit_normalizer,
    split_episode_ids,
    wrap_angle,
)
from .model import WorldModelMLP


@dataclass
class TrainingResult:
    model: WorldModelMLP
    input_normalizer: Normalizer
    target_normalizer: Normalizer
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int

    @property
    def losses(self) -> list[float]:
        """Backward-compatible alias for the training-loss history."""

        return self.train_losses


@dataclass
class LoadedWorldModel:
    model: WorldModelMLP
    input_normalizer: Normalizer
    target_normalizer: Normalizer
    split_episode_ids: dict[str, np.ndarray]
    training_config: dict[str, Any]
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    test_metrics: dict[str, float]

    @property
    def losses(self) -> list[float]:
        """Backward-compatible alias for the training-loss history."""

        return self.train_losses


def _as_float_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def train_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> TrainingResult:
    """Fit an MLP on already constructed model inputs and delta targets."""

    inputs = np.asarray(inputs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    validation_inputs = np.asarray(validation_inputs, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != WorldModelMLP.input_size:
        raise ValueError("inputs must have shape [N, 7]")
    if targets.ndim != 2 or targets.shape != (inputs.shape[0], 4):
        raise ValueError("targets must have shape [N, 4]")
    if (
        validation_inputs.ndim != 2
        or validation_inputs.shape[1] != WorldModelMLP.input_size
    ):
        raise ValueError("validation_inputs must have shape [N, 7]")
    if validation_targets.shape != (validation_inputs.shape[0], 4):
        raise ValueError("validation_targets must have shape [N, 4]")
    if validation_inputs.shape[0] == 0:
        raise ValueError("validation data must not be empty")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    input_normalizer = fit_normalizer(inputs)
    target_normalizer = fit_normalizer(targets)
    normalized_inputs = _as_float_tensor(input_normalizer.normalize(inputs))
    normalized_targets = _as_float_tensor(target_normalizer.normalize(targets))
    normalized_validation_inputs = _as_float_tensor(
        input_normalizer.normalize(validation_inputs)
    )
    normalized_validation_targets = _as_float_tensor(
        target_normalizer.normalize(validation_targets)
    )

    torch.manual_seed(seed)
    model = WorldModelMLP(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch_index in range(epochs):
        permutation = torch.randperm(inputs.shape[0], generator=generator)
        epoch_loss = 0.0
        model.train()
        for start in range(0, inputs.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            batch_inputs = normalized_inputs[indices]
            batch_targets = normalized_targets[indices]

            optimizer.zero_grad()
            prediction = model(batch_inputs)
            loss = loss_function(prediction, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * indices.numel()
        train_losses.append(epoch_loss / inputs.shape[0])

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(
                    model(normalized_validation_inputs),
                    normalized_validation_targets,
                )
            )
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch_index + 1
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is None:
        raise RuntimeError("training did not produce a model checkpoint")
    model.load_state_dict(best_state_dict)
    model.eval()
    return TrainingResult(
        model=model,
        input_normalizer=input_normalizer,
        target_normalizer=target_normalizer,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
    )


def evaluate_model(
    result: TrainingResult | LoadedWorldModel,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Return one-step errors after converting deltas back to physical units."""

    inputs = np.asarray(inputs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    normalized_inputs = _as_float_tensor(result.input_normalizer.normalize(inputs))
    with torch.no_grad():
        normalized_predictions = result.model(normalized_inputs).cpu().numpy()
    predictions = result.target_normalizer.denormalize(normalized_predictions)
    errors = predictions - targets
    errors[:, 2] = wrap_angle(errors[:, 2])
    absolute_errors = np.abs(errors)
    normalized_targets = result.target_normalizer.normalize(targets)
    normalized_error = normalized_predictions - normalized_targets
    heading_mae = float(absolute_errors[:, 2].mean())
    return {
        "normalized_mse": float(np.mean(np.square(normalized_error))),
        "mae_x": float(absolute_errors[:, 0].mean()),
        "mae_y": float(absolute_errors[:, 1].mean()),
        "mae_heading_radians": heading_mae,
        "mae_heading_degrees": math.degrees(heading_mae),
        "mae_velocity": float(absolute_errors[:, 3].mean()),
    }


def save_checkpoint(
    path: Path | str,
    result: TrainingResult,
    *,
    split_episode_ids: dict[str, np.ndarray],
    training_config: dict[str, Any],
    test_metrics: dict[str, float],
) -> Path:
    """Save weights and all metadata required for reproducible inference."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_config": {"hidden_size": result.model.hidden_size},
        "model_state_dict": result.model.state_dict(),
        "input_mean": _as_float_tensor(result.input_normalizer.mean),
        "input_std": _as_float_tensor(result.input_normalizer.std),
        "target_mean": _as_float_tensor(result.target_normalizer.mean),
        "target_std": _as_float_tensor(result.target_normalizer.std),
        "split_episode_ids": {
            name: torch.as_tensor(ids, dtype=torch.int64)
            for name, ids in split_episode_ids.items()
        },
        "training_config": dict(training_config),
        "train_losses": list(result.train_losses),
        "validation_losses": list(result.validation_losses),
        "best_epoch": result.best_epoch,
        "test_metrics": dict(test_metrics),
    }
    torch.save(payload, output)
    return output


def load_checkpoint(path: Path | str) -> LoadedWorldModel:
    """Load a checkpoint without permitting arbitrary pickled objects."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    format_version = payload.get("format_version")
    if format_version not in (1, 2):
        raise ValueError("unsupported checkpoint format")
    model = WorldModelMLP(hidden_size=int(payload["model_config"]["hidden_size"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    if format_version == 1:
        train_losses = [float(loss) for loss in payload["losses"]]
        validation_losses: list[float] = []
        best_epoch = len(train_losses)
        test_metrics: dict[str, float] = {}
    else:
        train_losses = [float(loss) for loss in payload["train_losses"]]
        validation_losses = [
            float(loss) for loss in payload["validation_losses"]
        ]
        best_epoch = int(payload["best_epoch"])
        test_metrics = {
            name: float(value) for name, value in payload["test_metrics"].items()
        }

    return LoadedWorldModel(
        model=model,
        input_normalizer=Normalizer(
            mean=payload["input_mean"].numpy().astype(np.float64),
            std=payload["input_std"].numpy().astype(np.float64),
        ),
        target_normalizer=Normalizer(
            mean=payload["target_mean"].numpy().astype(np.float64),
            std=payload["target_std"].numpy().astype(np.float64),
        ),
        split_episode_ids={
            name: ids.numpy().astype(np.int64)
            for name, ids in payload["split_episode_ids"].items()
        },
        training_config=dict(payload["training_config"]),
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
    )


def run_training(
    *,
    data_path: Path | str,
    output_path: Path | str,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Load an NPZ dataset, train by episode split, and save a checkpoint."""

    data_path = Path(data_path)
    required_arrays = {"states", "actions", "next_states", "episode_ids"}
    with np.load(data_path, allow_pickle=False) as loaded:
        missing = required_arrays - set(loaded.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {', '.join(sorted(missing))}")
        states = loaded["states"]
        actions = loaded["actions"]
        next_states = loaded["next_states"]
        episode_ids = loaded["episode_ids"]

    inputs, targets = build_model_arrays(states, actions, next_states)
    if episode_ids.ndim != 1 or episode_ids.shape[0] != inputs.shape[0]:
        raise ValueError("episode_ids must have shape [N]")
    splits = split_episode_ids(episode_ids, seed=seed)
    masks = {name: np.isin(episode_ids, ids) for name, ids in splits.items()}

    result = train_model(
        inputs[masks["train"]],
        targets[masks["train"]],
        validation_inputs=inputs[masks["validation"]],
        validation_targets=targets[masks["validation"]],
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    training_config = {
        "data_path": str(data_path),
        "hidden_size": hidden_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    validation_metrics = evaluate_model(
        result, inputs[masks["validation"]], targets[masks["validation"]]
    )
    test_metrics = evaluate_model(
        result, inputs[masks["test"]], targets[masks["test"]]
    )
    save_checkpoint(
        output_path,
        result,
        split_episode_ids=splits,
        training_config=training_config,
        test_metrics=test_metrics,
    )
    return {
        "transitions": int(inputs.shape[0]),
        "split_episodes": {name: int(ids.size) for name, ids in splits.items()},
        "split_transitions": {
            name: int(np.count_nonzero(mask)) for name, mask in masks.items()
        },
        "initial_train_loss": result.train_losses[0],
        "final_train_loss": result.train_losses[-1],
        "best_epoch": result.best_epoch,
        "best_validation_loss": min(result.validation_losses),
        "validation": validation_metrics,
        "test": test_metrics,
        "checkpoint": str(Path(output_path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/world_model.pt")
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        summary = run_training(
            data_path=args.data,
            output_path=args.output,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
