"""Evaluate recursive multi-step predictions on held-out episodes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import wrap_angle
from .train_world_model import LoadedWorldModel, load_checkpoint


@dataclass(frozen=True)
class EpisodeRollout:
    episode_id: int
    true_states: np.ndarray
    predicted_states: np.ndarray
    actions: np.ndarray


def _model_input(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            state[0],
            state[1],
            math.sin(state[2]),
            math.cos(state[2]),
            state[3],
            action[0],
            action[1],
        ],
        dtype=np.float64,
    )


def rollout_episode(
    world_model: LoadedWorldModel,
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Recursively predict a state sequence from one state and recorded actions."""

    initial_state = np.asarray(initial_state, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if initial_state.shape != (4,):
        raise ValueError("initial_state must have shape [4]")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [T, 2]")
    if not np.all(np.isfinite(initial_state)) or not np.all(np.isfinite(actions)):
        raise ValueError("rollout state and actions must contain only finite values")

    predicted_states = [initial_state.copy()]
    world_model.model.eval()
    for action in actions:
        current_state = predicted_states[-1]
        raw_input = _model_input(current_state, action)[None, :]
        normalized_input = world_model.input_normalizer.normalize(raw_input)
        input_tensor = torch.as_tensor(normalized_input, dtype=torch.float32)
        with torch.no_grad():
            normalized_delta = world_model.model(input_tensor).cpu().numpy()
        delta = world_model.target_normalizer.denormalize(normalized_delta)[0]
        next_state = current_state + delta
        next_state[2] = wrap_angle(next_state[2])
        predicted_states.append(next_state)
    return np.asarray(predicted_states, dtype=np.float64)


def build_episode_rollouts(
    world_model: LoadedWorldModel,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    episode_ids: np.ndarray,
    step_ids: np.ndarray,
    selected_episode_ids: np.ndarray,
) -> dict[int, EpisodeRollout]:
    """Build ordered recursive rollouts for selected held-out episodes."""

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    step_ids = np.asarray(step_ids)
    count = states.shape[0]
    if states.shape != (count, 4) or next_states.shape != (count, 4):
        raise ValueError("states and next_states must have shape [N, 4]")
    if actions.shape != (count, 2):
        raise ValueError("actions must have shape [N, 2]")
    if episode_ids.shape != (count,) or step_ids.shape != (count,):
        raise ValueError("episode_ids and step_ids must have shape [N]")

    rollouts: dict[int, EpisodeRollout] = {}
    for episode_id_value in np.asarray(selected_episode_ids).tolist():
        episode_id = int(episode_id_value)
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size == 0:
            raise ValueError(f"episode {episode_id} is missing from the dataset")
        indices = indices[np.argsort(step_ids[indices], kind="stable")]
        ordered_steps = step_ids[indices]
        if not np.array_equal(ordered_steps, np.arange(indices.size)):
            raise ValueError(f"episode {episode_id} step_ids must be contiguous from zero")

        episode_states = states[indices]
        episode_next_states = next_states[indices]
        if indices.size > 1 and not np.allclose(
            episode_states[1:], episode_next_states[:-1], atol=1e-10
        ):
            raise ValueError(f"episode {episode_id} transitions are not contiguous")
        episode_actions = actions[indices]
        true_states = np.vstack((episode_states[0], episode_next_states))
        predicted_states = rollout_episode(
            world_model, episode_states[0], episode_actions
        )
        rollouts[episode_id] = EpisodeRollout(
            episode_id=episode_id,
            true_states=true_states,
            predicted_states=predicted_states,
            actions=episode_actions,
        )
    return rollouts


def summarize_horizons(
    rollouts: dict[int, EpisodeRollout],
    *,
    horizons: Iterable[int],
) -> dict[str, Any]:
    """Aggregate exact-horizon errors across all sufficiently long episodes."""

    horizon_metrics: dict[str, dict[str, float | int]] = {}
    for horizon_value in horizons:
        horizon = int(horizon_value)
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        eligible = [
            rollout
            for rollout in rollouts.values()
            if rollout.actions.shape[0] >= horizon
        ]
        if not eligible:
            continue

        position_errors = []
        heading_errors = []
        velocity_errors = []
        for rollout in eligible:
            error = rollout.predicted_states[horizon] - rollout.true_states[horizon]
            position_errors.append(float(np.linalg.norm(error[:2])))
            heading_errors.append(abs(float(wrap_angle(error[2]))))
            velocity_errors.append(abs(float(error[3])))
        horizon_metrics[str(horizon)] = {
            "episodes": len(eligible),
            "mean_position_error": float(np.mean(position_errors)),
            "mean_heading_error_radians": float(np.mean(heading_errors)),
            "mean_heading_error_degrees": math.degrees(float(np.mean(heading_errors))),
            "mean_velocity_error": float(np.mean(velocity_errors)),
        }

    return {
        "episodes": len(rollouts),
        "horizons": horizon_metrics,
    }


def plot_episode_rollout(
    rollout: EpisodeRollout,
    output_path: Path | str,
) -> Path:
    """Plot a representative trajectory and its accumulated state errors."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = rollout.predicted_states[1:] - rollout.true_states[1:]
    position_errors = np.linalg.norm(errors[:, :2], axis=1)
    heading_errors = np.degrees(np.abs(wrap_angle(errors[:, 2])))
    velocity_errors = np.abs(errors[:, 3])
    steps = np.arange(1, rollout.predicted_states.shape[0])

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    trajectory_axis = axes[0, 0]
    trajectory_axis.plot(
        rollout.true_states[:, 0],
        rollout.true_states[:, 1],
        label="True trajectory",
        linewidth=2,
    )
    trajectory_axis.plot(
        rollout.predicted_states[:, 0],
        rollout.predicted_states[:, 1],
        label="Predicted rollout",
        linewidth=2,
    )
    trajectory_axis.scatter(
        rollout.true_states[0, 0],
        rollout.true_states[0, 1],
        label="Start",
        color="#2a9d8f",
        zorder=3,
    )
    trajectory_axis.set(
        title="XY trajectory",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    trajectory_axis.axis("equal")
    trajectory_axis.legend()

    plots = (
        (axes[0, 1], position_errors, "Position error", "error (m)"),
        (axes[1, 0], heading_errors, "Heading error", "error (degrees)"),
        (axes[1, 1], velocity_errors, "Velocity error", "error (m/s)"),
    )
    for axis, values, title, ylabel in plots:
        axis.plot(steps, values, linewidth=2)
        axis.set(title=title, xlabel="Rollout step", ylabel=ylabel)
        axis.grid(True, alpha=0.3)

    figure.suptitle(f"World Model Rollout — Test Episode {rollout.episode_id}")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def run_rollout_evaluation(
    *,
    data_path: Path | str,
    checkpoint_path: Path | str,
    plot_path: Path | str,
    horizons: Iterable[int] = (1, 5, 10, 20, 50),
    episode_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate all held-out episodes and plot one representative rollout."""

    required_arrays = {
        "states",
        "actions",
        "next_states",
        "episode_ids",
        "step_ids",
    }
    with np.load(Path(data_path), allow_pickle=False) as loaded:
        missing = required_arrays - set(loaded.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {', '.join(sorted(missing))}")
        arrays = {name: loaded[name] for name in required_arrays}

    world_model = load_checkpoint(checkpoint_path)
    test_episode_ids = world_model.split_episode_ids.get("test")
    if test_episode_ids is None or test_episode_ids.size == 0:
        raise ValueError("checkpoint does not contain test episode IDs")
    rollouts = build_episode_rollouts(
        world_model,
        states=arrays["states"],
        actions=arrays["actions"],
        next_states=arrays["next_states"],
        episode_ids=arrays["episode_ids"],
        step_ids=arrays["step_ids"],
        selected_episode_ids=test_episode_ids,
    )

    if episode_id is None:
        representative_episode = max(
            rollouts,
            key=lambda current_id: rollouts[current_id].actions.shape[0],
        )
    else:
        representative_episode = int(episode_id)
        if representative_episode not in rollouts:
            raise ValueError(
                f"episode {representative_episode} is not in the checkpoint test split"
            )

    output = plot_episode_rollout(
        rollouts[representative_episode],
        plot_path,
    )
    summary = summarize_horizons(rollouts, horizons=horizons)
    summary.update(
        {
            "representative_episode": representative_episode,
            "plot": str(output),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/world_model.pt"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("artifacts/rollout_evaluation.png"),
    )
    parser.add_argument("--episode-id", type=int)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50],
    )
    args = parser.parse_args()

    try:
        summary = run_rollout_evaluation(
            data_path=args.data,
            checkpoint_path=args.checkpoint,
            plot_path=args.plot,
            horizons=args.horizons,
            episode_id=args.episode_id,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
