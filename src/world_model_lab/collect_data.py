"""Collect reproducible state-transition data from :class:`CarEnv`."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .car_env import CarEnv


Dataset = dict[str, np.ndarray]


def _sample_initial_state(rng: np.random.Generator, env: CarEnv) -> np.ndarray:
    min_x, max_x, min_y, max_y = env.world_bounds
    for _ in range(10_000):
        position = np.asarray(
            [
                rng.uniform(min_x + env.car_radius, max_x - env.car_radius),
                rng.uniform(min_y + env.car_radius, max_y - env.car_radius),
            ],
            dtype=np.float64,
        )
        outside_obstacle = (
            np.linalg.norm(position - env.obstacle)
            > env.car_radius + env.obstacle_radius
        )
        outside_goal = np.linalg.norm(position - env.goal) > env.goal_radius
        if outside_obstacle and outside_goal:
            return np.asarray(
                [
                    position[0],
                    position[1],
                    rng.uniform(-math.pi, math.pi),
                    rng.uniform(0.0, env.max_speed),
                ],
                dtype=np.float64,
            )
    raise RuntimeError("could not sample a valid initial state")


def _terminal_reason(info: dict[str, float | int | bool]) -> str:
    if info["reached_goal"]:
        return "goal"
    if info["collision"]:
        return "collision"
    if info["out_of_bounds"]:
        return "out_of_bounds"
    if info["time_limit"]:
        return "time_limit"
    return ""


def collect_transitions(
    *,
    episodes: int = 250,
    max_steps: int = 200,
    action_hold_steps: int = 5,
    seed: int = 7,
) -> Dataset:
    """Run randomized episodes and return aligned transition arrays."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if action_hold_steps <= 0:
        raise ValueError("action_hold_steps must be positive")

    rng = np.random.default_rng(seed)
    template = CarEnv(max_steps=max_steps)
    records: dict[str, list] = {
        "states": [],
        "actions": [],
        "next_states": [],
        "rewards": [],
        "dones": [],
        "episode_ids": [],
        "step_ids": [],
        "terminal_reasons": [],
    }

    for episode_id in range(episodes):
        initial_state = _sample_initial_state(rng, template)
        env = CarEnv(initial_state=initial_state, max_steps=max_steps)
        steering = 0.0
        acceleration = 0.0

        for step_id in range(max_steps):
            if step_id % action_hold_steps == 0:
                steering = rng.uniform(-env.max_steering, env.max_steering)
                acceleration = rng.uniform(
                    -env.max_acceleration, env.max_acceleration
                )

            state = env.state
            next_state, reward, done, info = env.step(steering, acceleration)
            applied_action = np.asarray(
                [info["applied_steering"], info["applied_acceleration"]],
                dtype=np.float64,
            )

            records["states"].append(state)
            records["actions"].append(applied_action)
            records["next_states"].append(next_state)
            records["rewards"].append(reward)
            records["dones"].append(done)
            records["episode_ids"].append(episode_id)
            records["step_ids"].append(step_id)
            records["terminal_reasons"].append(_terminal_reason(info))

            if done:
                break

    return {
        "states": np.asarray(records["states"], dtype=np.float64),
        "actions": np.asarray(records["actions"], dtype=np.float64),
        "next_states": np.asarray(records["next_states"], dtype=np.float64),
        "rewards": np.asarray(records["rewards"], dtype=np.float64),
        "dones": np.asarray(records["dones"], dtype=np.bool_),
        "episode_ids": np.asarray(records["episode_ids"], dtype=np.int64),
        "step_ids": np.asarray(records["step_ids"], dtype=np.int64),
        "terminal_reasons": np.asarray(records["terminal_reasons"], dtype=np.str_),
    }


def save_dataset(dataset: Dataset, output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    return path


def summarize_dataset(dataset: Dataset) -> dict[str, int | float]:
    reasons = dataset["terminal_reasons"]
    episode_ids = dataset["episode_ids"]
    episodes = int(np.unique(episode_ids).size)
    transitions = int(dataset["states"].shape[0])
    return {
        "transitions": transitions,
        "episodes": episodes,
        "average_episode_length": transitions / episodes,
        "terminal_total": int(np.count_nonzero(dataset["dones"])),
        "goals": int(np.count_nonzero(reasons == "goal")),
        "collisions": int(np.count_nonzero(reasons == "collision")),
        "out_of_bounds": int(np.count_nonzero(reasons == "out_of_bounds")),
        "time_limits": int(np.count_nonzero(reasons == "time_limit")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-hold-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/transitions.npz"))
    args = parser.parse_args()

    try:
        dataset = collect_transitions(
            episodes=args.episodes,
            max_steps=args.max_steps,
            action_hold_steps=args.action_hold_steps,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    output = save_dataset(dataset, args.output)
    print(json.dumps(summarize_dataset(dataset), indent=2, sort_keys=True))
    print(f"saved dataset to {output}")


if __name__ == "__main__":
    main()
