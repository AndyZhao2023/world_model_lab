"""Run a small reproducible trajectory through the car environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .car_env import CarEnv
from .visualize import draw_environment


def run_demo(steps: int, seed: int) -> CarEnv:
    rng = np.random.default_rng(seed)
    env = CarEnv(initial_state=(1.0, 1.0, 0.55, 0.8), max_steps=steps)

    for index in range(steps):
        # A gentle S-curve makes heading changes visible without acting as a planner.
        steering = 0.22 * np.sin(index / 8.0) + rng.normal(0.0, 0.025)
        acceleration = 0.35 if index < 12 else 0.0
        _, _, done, _ = env.step(float(steering), float(acceleration))
        if done:
            break
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    env = run_demo(args.steps, args.seed)
    figure, _ = draw_environment(env)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"saved trajectory to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
