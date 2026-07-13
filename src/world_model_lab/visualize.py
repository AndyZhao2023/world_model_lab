"""Passive Matplotlib visualization for :mod:`world_model_lab.car_env`."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrow, Rectangle

from .car_env import CarEnv


def draw_environment(env: CarEnv, ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Draw an environment snapshot without mutating the environment."""

    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 7))
    else:
        figure = ax.figure
        ax.clear()

    min_x, max_x, min_y, max_y = env.world_bounds
    ax.add_patch(
        Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            fill=False,
            edgecolor="#263238",
            linewidth=2.0,
        )
    )
    ax.add_patch(
        Circle(
            env.obstacle,
            env.obstacle_radius,
            facecolor="#d95f59",
            edgecolor="#8f2f2a",
            alpha=0.85,
            label="Obstacle",
        )
    )
    ax.add_patch(
        Circle(
            env.goal,
            env.goal_radius,
            facecolor="#5abf75",
            edgecolor="#287d3c",
            alpha=0.65,
            label="Goal",
        )
    )

    trajectory = env.trajectory
    xs = [state[0] for state in trajectory]
    ys = [state[1] for state in trajectory]
    ax.plot(xs, ys, color="#2979b8", linewidth=2.0, label="Trajectory")

    x, y, heading, velocity = env.state
    ax.add_patch(
        Circle(
            (x, y),
            env.car_radius,
            facecolor="#174a73",
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
            label="Car",
        )
    )
    arrow_length = max(0.45, env.car_radius * 2.5)
    ax.add_patch(
        FancyArrow(
            x,
            y,
            arrow_length * math.cos(heading),
            arrow_length * math.sin(heading),
            width=0.035,
            head_width=0.18,
            head_length=0.18,
            color="#ffb000",
            length_includes_head=True,
            zorder=6,
        )
    )

    margin = 0.45
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(f"2D Car Environment · step={env.steps} · speed={velocity:.2f}")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper left", frameon=False)
    figure.tight_layout()
    return figure, ax
