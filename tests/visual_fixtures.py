from __future__ import annotations

import numpy as np

from world_model_lab.car_env import CarEnv


def make_transition_source() -> dict[str, np.ndarray]:
    """Return source-order episodes 7/T=4 then 3/T=2."""

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
    terminal_flags = (
        ("reached_goal", "goal"),
        ("collision", "collision"),
        ("out_of_bounds", "out_of_bounds"),
        ("time_limit", "time_limit"),
    )
    episode_specs = (
        (7, 4, (1.0, 1.0, 0.0, 2.0), 0.05, 0.1),
        (3, 2, (2.0, 1.5, 0.2, 1.5), -0.04, 0.2),
    )
    for episode_id, length, initial_state, steering, acceleration in episode_specs:
        env = CarEnv(initial_state=initial_state, max_steps=length)
        for step_id in range(length):
            state = env.state
            next_state, reward, done, info = env.step(steering, acceleration)
            reason = next(
                (
                    label
                    for flag, label in terminal_flags
                    if bool(info[flag])
                ),
                "",
            )
            records["states"].append(state)
            records["actions"].append(
                np.asarray(
                    [
                        info["applied_steering"],
                        info["applied_acceleration"],
                    ],
                    dtype=np.float64,
                )
            )
            records["next_states"].append(next_state)
            records["rewards"].append(reward)
            records["dones"].append(done)
            records["episode_ids"].append(episode_id)
            records["step_ids"].append(step_id)
            records["terminal_reasons"].append(reason)

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


def clone_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {name: values.copy() for name, values in arrays.items()}
