"""Train and evaluate a one-step state-delta world model."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import (
    Normalizer,
    SequenceWindows,
    build_model_arrays,
    build_sequence_windows,
    fit_normalizer,
    split_episode_ids,
    wrap_angle,
)
from .model import WorldModelMLP
from .rollout_training import rollout_state_loss


@dataclass
class TrainingResult:
    model: WorldModelMLP
    input_normalizer: Normalizer
    target_normalizer: Normalizer
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    train_one_step_losses: list[float] = field(default_factory=list)
    train_rollout_losses: list[float] = field(default_factory=list)
    validation_one_step_losses: list[float] = field(default_factory=list)
    validation_rollout_losses: list[float] = field(default_factory=list)

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
    train_one_step_losses: list[float] = field(default_factory=list)
    train_rollout_losses: list[float] = field(default_factory=list)
    validation_one_step_losses: list[float] = field(default_factory=list)
    validation_rollout_losses: list[float] = field(default_factory=list)

    @property
    def losses(self) -> list[float]:
        """Backward-compatible alias for the training-loss history."""

        return self.train_losses


def _as_float_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def _mean_rollout_loss(
    model: WorldModelMLP,
    *,
    sequence_states: torch.Tensor,
    sequence_actions: torch.Tensor,
    sequence_next_states: torch.Tensor,
    batch_size: int,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    """Evaluate rollout loss across every supplied sequence window."""

    total_loss = 0.0
    count = sequence_states.shape[0]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        loss = rollout_state_loss(
            model,
            initial_states=sequence_states[start:stop, 0],
            actions=sequence_actions[start:stop],
            true_next_states=sequence_next_states[start:stop],
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        total_loss += float(loss) * (stop - start)
    return total_loss / count


def _validated_normalizer(
    normalizer: Normalizer,
    *,
    size: int,
    name: str,
) -> Normalizer:
    mean = np.asarray(normalizer.mean, dtype=np.float64)
    std = np.asarray(normalizer.std, dtype=np.float64)
    if (
        mean.shape != (size,)
        or std.shape != (size,)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0.0)
    ):
        raise ValueError(
            f"{name} must have finite mean/std shape [{size}] "
            "and positive std"
        )
    return Normalizer(mean.copy(), std.copy())


def train_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    train_sequences: SequenceWindows | None = None,
    validation_sequences: SequenceWindows | None = None,
    rollout_loss_weight: float = 1.0,
    input_normalizer: Normalizer | None = None,
    target_normalizer: Normalizer | None = None,
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
    if (train_sequences is None) != (validation_sequences is None):
        raise ValueError("train and validation sequences must be provided together")
    if not math.isfinite(rollout_loss_weight) or rollout_loss_weight < 0.0:
        raise ValueError("rollout_loss_weight must be finite and non-negative")
    multi_step = train_sequences is not None
    if multi_step:
        assert validation_sequences is not None
        if train_sequences.count == 0 or validation_sequences.count == 0:
            raise ValueError("multi-step training requires non-empty sequence windows")
        if train_sequences.horizon != validation_sequences.horizon:
            raise ValueError("train and validation horizons must match")
        if train_sequences.horizon <= 1:
            raise ValueError("multi-step training requires a horizon greater than one")

    if (input_normalizer is None) != (target_normalizer is None):
        raise ValueError("input and target normalizers must be provided together")
    if input_normalizer is None:
        input_normalizer = fit_normalizer(inputs)
        target_normalizer = fit_normalizer(targets)
    else:
        input_normalizer = _validated_normalizer(
            input_normalizer,
            size=WorldModelMLP.input_size,
            name="input_normalizer",
        )
        assert target_normalizer is not None
        target_normalizer = _validated_normalizer(
            target_normalizer,
            size=4,
            name="target_normalizer",
        )
    normalized_inputs = _as_float_tensor(input_normalizer.normalize(inputs))
    normalized_targets = _as_float_tensor(target_normalizer.normalize(targets))
    normalized_validation_inputs = _as_float_tensor(
        input_normalizer.normalize(validation_inputs)
    )
    normalized_validation_targets = _as_float_tensor(
        target_normalizer.normalize(validation_targets)
    )
    input_mean = _as_float_tensor(input_normalizer.mean)
    input_std = _as_float_tensor(input_normalizer.std)
    target_mean = _as_float_tensor(target_normalizer.mean)
    target_std = _as_float_tensor(target_normalizer.std)
    if multi_step:
        train_sequence_states = _as_float_tensor(train_sequences.states)
        train_sequence_actions = _as_float_tensor(train_sequences.actions)
        train_sequence_next_states = _as_float_tensor(train_sequences.next_states)
        validation_sequence_states = _as_float_tensor(validation_sequences.states)
        validation_sequence_actions = _as_float_tensor(
            validation_sequences.actions
        )
        validation_sequence_next_states = _as_float_tensor(
            validation_sequences.next_states
        )

    torch.manual_seed(seed)
    model = WorldModelMLP(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    sequence_generator = torch.Generator().manual_seed(seed + 1)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    train_one_step_losses: list[float] = []
    train_rollout_losses: list[float] = []
    validation_one_step_losses: list[float] = []
    validation_rollout_losses: list[float] = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch_index in range(epochs):
        permutation = torch.randperm(inputs.shape[0], generator=generator)
        sequence_permutation = (
            torch.randperm(
                train_sequences.count,
                generator=sequence_generator,
            )
            if multi_step
            else None
        )
        epoch_one_step_loss = 0.0
        epoch_rollout_loss = 0.0
        model.train()
        for batch_index, start in enumerate(
            range(0, inputs.shape[0], batch_size)
        ):
            indices = permutation[start : start + batch_size]
            batch_inputs = normalized_inputs[indices]
            batch_targets = normalized_targets[indices]

            optimizer.zero_grad()
            prediction = model(batch_inputs)
            one_step_loss = loss_function(prediction, batch_targets)
            if multi_step:
                sequence_offsets = (
                    torch.arange(batch_size) + batch_index * batch_size
                ) % train_sequences.count
                sequence_indices = sequence_permutation[sequence_offsets]
                rollout_loss = rollout_state_loss(
                    model,
                    initial_states=train_sequence_states[sequence_indices, 0],
                    actions=train_sequence_actions[sequence_indices],
                    true_next_states=train_sequence_next_states[sequence_indices],
                    input_mean=input_mean,
                    input_std=input_std,
                    target_mean=target_mean,
                    target_std=target_std,
                )
            else:
                rollout_loss = torch.zeros((), dtype=torch.float32)
            total_loss = one_step_loss + rollout_loss_weight * rollout_loss
            total_loss.backward()
            optimizer.step()
            weight = indices.numel()
            epoch_one_step_loss += float(one_step_loss.detach()) * weight
            epoch_rollout_loss += float(rollout_loss.detach()) * weight
        train_one_step = epoch_one_step_loss / inputs.shape[0]
        train_rollout = epoch_rollout_loss / inputs.shape[0]
        train_one_step_losses.append(train_one_step)
        train_rollout_losses.append(train_rollout)
        train_losses.append(
            train_one_step + rollout_loss_weight * train_rollout
        )

        model.eval()
        with torch.no_grad():
            validation_one_step = float(
                loss_function(
                    model(normalized_validation_inputs),
                    normalized_validation_targets,
                )
            )
            validation_rollout = (
                _mean_rollout_loss(
                    model,
                    sequence_states=validation_sequence_states,
                    sequence_actions=validation_sequence_actions,
                    sequence_next_states=validation_sequence_next_states,
                    batch_size=batch_size,
                    input_mean=input_mean,
                    input_std=input_std,
                    target_mean=target_mean,
                    target_std=target_std,
                )
                if multi_step
                else 0.0
            )
        validation_total = (
            validation_one_step + rollout_loss_weight * validation_rollout
        )
        validation_losses.append(validation_total)
        validation_one_step_losses.append(validation_one_step)
        validation_rollout_losses.append(validation_rollout)
        if validation_total < best_validation_loss:
            best_validation_loss = validation_total
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
        train_one_step_losses=train_one_step_losses,
        train_rollout_losses=train_rollout_losses,
        validation_one_step_losses=validation_one_step_losses,
        validation_rollout_losses=validation_rollout_losses,
    )


def predict_deltas(
    result: TrainingResult | LoadedWorldModel,
    inputs: np.ndarray,
) -> np.ndarray:
    """Predict denormalized state deltas for a batch of model inputs."""

    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != WorldModelMLP.input_size:
        raise ValueError("inputs must have shape [N, 7]")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("inputs must contain only finite values")
    normalized_inputs = _as_float_tensor(result.input_normalizer.normalize(inputs))
    result.model.eval()
    with torch.no_grad():
        normalized_predictions = result.model(normalized_inputs).cpu().numpy()
    return result.target_normalizer.denormalize(normalized_predictions)


def evaluate_model(
    result: TrainingResult | LoadedWorldModel,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Return one-step errors after converting deltas back to physical units."""

    inputs = np.asarray(inputs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    predictions = predict_deltas(result, inputs)
    normalized_predictions = result.target_normalizer.normalize(predictions)
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
    train_one_step_losses = result.train_one_step_losses or result.train_losses
    train_rollout_losses = result.train_rollout_losses or [
        0.0
    ] * len(result.train_losses)
    validation_one_step_losses = (
        result.validation_one_step_losses or result.validation_losses
    )
    validation_rollout_losses = result.validation_rollout_losses or [
        0.0
    ] * len(result.validation_losses)
    payload = {
        "format_version": 3,
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
        "train_one_step_losses": list(train_one_step_losses),
        "train_rollout_losses": list(train_rollout_losses),
        "validation_one_step_losses": list(validation_one_step_losses),
        "validation_rollout_losses": list(validation_rollout_losses),
        "best_epoch": result.best_epoch,
        "test_metrics": dict(test_metrics),
    }
    torch.save(payload, output)
    return output


def load_checkpoint(path: Path | str) -> LoadedWorldModel:
    """Load a checkpoint without permitting arbitrary pickled objects."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    format_version = payload.get("format_version")
    if format_version not in (1, 2, 3):
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
    if format_version in (1, 2):
        train_one_step_losses = list(train_losses)
        train_rollout_losses = [0.0] * len(train_losses)
        validation_one_step_losses = list(validation_losses)
        validation_rollout_losses = [0.0] * len(validation_losses)
    else:
        train_one_step_losses = [
            float(loss) for loss in payload["train_one_step_losses"]
        ]
        train_rollout_losses = [
            float(loss) for loss in payload["train_rollout_losses"]
        ]
        validation_one_step_losses = [
            float(loss) for loss in payload["validation_one_step_losses"]
        ]
        validation_rollout_losses = [
            float(loss) for loss in payload["validation_rollout_losses"]
        ]

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
        train_one_step_losses=train_one_step_losses,
        train_rollout_losses=train_rollout_losses,
        validation_one_step_losses=validation_one_step_losses,
        validation_rollout_losses=validation_rollout_losses,
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
    split_seed: int | None = None,
    rollout_horizon: int = 1,
    rollout_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Load an NPZ dataset, train by episode split, and save a checkpoint."""

    if rollout_horizon <= 0:
        raise ValueError("rollout_horizon must be positive")
    if not math.isfinite(rollout_loss_weight) or rollout_loss_weight < 0.0:
        raise ValueError("rollout_loss_weight must be finite and non-negative")
    effective_rollout_weight = (
        rollout_loss_weight if rollout_horizon > 1 else 0.0
    )
    data_path = Path(data_path)
    required_arrays = {"states", "actions", "next_states", "episode_ids"}
    if rollout_horizon > 1:
        required_arrays.add("step_ids")
    with np.load(data_path, allow_pickle=False) as loaded:
        missing = required_arrays - set(loaded.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {', '.join(sorted(missing))}")
        states = loaded["states"]
        actions = loaded["actions"]
        next_states = loaded["next_states"]
        episode_ids = loaded["episode_ids"]
        step_ids = loaded["step_ids"] if rollout_horizon > 1 else None

    inputs, targets = build_model_arrays(states, actions, next_states)
    if episode_ids.ndim != 1 or episode_ids.shape[0] != inputs.shape[0]:
        raise ValueError("episode_ids must have shape [N]")
    effective_split_seed = seed if split_seed is None else split_seed
    splits = split_episode_ids(episode_ids, seed=effective_split_seed)
    masks = {name: np.isin(episode_ids, ids) for name, ids in splits.items()}
    train_sequences = validation_sequences = None
    if rollout_horizon > 1:
        assert step_ids is not None
        train_sequences = build_sequence_windows(
            states,
            actions,
            next_states,
            episode_ids=episode_ids,
            step_ids=step_ids,
            selected_episode_ids=splits["train"],
            horizon=rollout_horizon,
        )
        validation_sequences = build_sequence_windows(
            states,
            actions,
            next_states,
            episode_ids=episode_ids,
            step_ids=step_ids,
            selected_episode_ids=splits["validation"],
            horizon=rollout_horizon,
        )
        if train_sequences.count == 0 or validation_sequences.count == 0:
            raise ValueError(
                f"rollout horizon {rollout_horizon} has no eligible "
                "training or validation windows"
            )

    result = train_model(
        inputs[masks["train"]],
        targets[masks["train"]],
        validation_inputs=inputs[masks["validation"]],
        validation_targets=targets[masks["validation"]],
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
        rollout_loss_weight=effective_rollout_weight,
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
        "split_seed": effective_split_seed,
        "rollout_horizon": rollout_horizon,
        "rollout_loss_weight": effective_rollout_weight,
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
    best_index = result.best_epoch - 1
    return {
        "transitions": int(inputs.shape[0]),
        "split_seed": effective_split_seed,
        "rollout_horizon": rollout_horizon,
        "rollout_loss_weight": effective_rollout_weight,
        "train_sequence_windows": (
            train_sequences.count if train_sequences is not None else 0
        ),
        "validation_sequence_windows": (
            validation_sequences.count
            if validation_sequences is not None
            else 0
        ),
        "split_episodes": {name: int(ids.size) for name, ids in splits.items()},
        "split_transitions": {
            name: int(np.count_nonzero(mask)) for name, mask in masks.items()
        },
        "initial_train_loss": result.train_losses[0],
        "final_train_loss": result.train_losses[-1],
        "initial_train_one_step_loss": result.train_one_step_losses[0],
        "final_train_one_step_loss": result.train_one_step_losses[-1],
        "initial_train_rollout_loss": result.train_rollout_losses[0],
        "final_train_rollout_loss": result.train_rollout_losses[-1],
        "best_epoch": result.best_epoch,
        "best_validation_loss": min(result.validation_losses),
        "best_validation_one_step_loss": (
            result.validation_one_step_losses[best_index]
        ),
        "best_validation_rollout_loss": (
            result.validation_rollout_losses[best_index]
        ),
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
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--rollout-horizon", type=int, default=1)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
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
            split_seed=args.split_seed,
            rollout_horizon=args.rollout_horizon,
            rollout_loss_weight=args.rollout_loss_weight,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
