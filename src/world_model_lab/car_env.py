"""Deterministic two-dimensional car environment.

This module is the "real environment" for the learning lab.  It owns the
transition function that a learned world model will approximate later.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class CarEnv:
    """A minimal kinematic-bicycle environment with circular geometry."""

    def __init__(
        self,
        *,
        initial_state: Iterable[float] = (1.0, 1.0, 0.0, 0.0),
        world_bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 8.0),
        obstacle: tuple[float, float] = (5.0, 4.0),
        obstacle_radius: float = 1.0,
        goal: tuple[float, float] = (9.0, 7.0),
        goal_radius: float = 0.5,
        car_radius: float = 0.2,
        dt: float = 0.1,
        wheelbase: float = 0.5,
        max_speed: float = 3.0,
        max_steering: float = 0.5,
        max_acceleration: float = 1.0,
        max_steps: int = 500,
    ) -> None:
        self.initial_state = np.asarray(tuple(initial_state), dtype=np.float64)
        if self.initial_state.shape != (4,):
            raise ValueError("initial_state must contain x, y, heading, velocity")

        self.world_bounds = tuple(float(value) for value in world_bounds)
        if len(self.world_bounds) != 4:
            raise ValueError("world_bounds must be (min_x, max_x, min_y, max_y)")
        min_x, max_x, min_y, max_y = self.world_bounds
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("world_bounds minima must be smaller than maxima")

        self.obstacle = np.asarray(obstacle, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        if self.obstacle.shape != (2,) or self.goal.shape != (2,):
            raise ValueError("obstacle and goal must be two-dimensional points")

        positive_values = {
            "obstacle_radius": obstacle_radius,
            "goal_radius": goal_radius,
            "car_radius": car_radius,
            "dt": dt,
            "wheelbase": wheelbase,
            "max_speed": max_speed,
            "max_steering": max_steering,
            "max_acceleration": max_acceleration,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.obstacle_radius = float(obstacle_radius)
        self.goal_radius = float(goal_radius)
        self.car_radius = float(car_radius)
        self.dt = float(dt)
        self.wheelbase = float(wheelbase)
        self.max_speed = float(max_speed)
        self.max_steering = float(max_steering)
        self.max_acceleration = float(max_acceleration)
        self.max_steps = int(max_steps)

        self._state = self.initial_state.copy()
        self._trajectory: list[np.ndarray] = []
        self._steps = 0
        self._done = False
        self.reset()

    @property
    def state(self) -> np.ndarray:
        """Return a copy so callers cannot mutate environment state."""

        return self._state.copy()

    @property
    def trajectory(self) -> list[np.ndarray]:
        """Return independent copies of every recorded state."""

        return [state.copy() for state in self._trajectory]

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def done(self) -> bool:
        return self._done

    def reset(self) -> np.ndarray:
        self._state = self.initial_state.copy()
        self._state[2] = self._wrap_angle(self._state[2])
        self._state[3] = np.clip(self._state[3], 0.0, self.max_speed)
        self._steps = 0
        self._done = False
        self._trajectory = [self._state.copy()]
        return self.state

    def step(
        self, steering: float, acceleration: float
    ) -> tuple[np.ndarray, float, bool, dict[str, float | int | bool]]:
        if self._done:
            raise RuntimeError("episode has terminated; call reset() before step()")

        applied_steering = float(np.clip(steering, -self.max_steering, self.max_steering))
        applied_acceleration = float(
            np.clip(acceleration, -self.max_acceleration, self.max_acceleration)
        )

        x, y, heading, velocity = self._state
        previous_distance = self._distance_to_goal(x, y)

        next_x = x + velocity * math.cos(heading) * self.dt
        next_y = y + velocity * math.sin(heading) * self.dt
        next_heading = self._wrap_angle(
            heading
            + velocity / self.wheelbase * math.tan(applied_steering) * self.dt
        )
        next_velocity = float(
            np.clip(
                velocity + applied_acceleration * self.dt,
                0.0,
                self.max_speed,
            )
        )

        self._state = np.asarray(
            [next_x, next_y, next_heading, next_velocity], dtype=np.float64
        )
        self._steps += 1
        self._trajectory.append(self._state.copy())

        distance_to_goal = self._distance_to_goal(next_x, next_y)
        reached_goal = distance_to_goal <= self.goal_radius
        collision = (
            float(np.linalg.norm(self._state[:2] - self.obstacle))
            <= self.car_radius + self.obstacle_radius
        )
        out_of_bounds = self._is_out_of_bounds(next_x, next_y)
        time_limit = self._steps >= self.max_steps
        self._done = reached_goal or collision or out_of_bounds or time_limit

        reward = previous_distance - distance_to_goal
        if reached_goal:
            reward += 10.0
        if collision:
            reward -= 10.0
        if out_of_bounds:
            reward -= 10.0

        info: dict[str, float | int | bool] = {
            "reached_goal": reached_goal,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "time_limit": time_limit,
            "distance_to_goal": distance_to_goal,
            "steps": self._steps,
            "applied_steering": applied_steering,
            "applied_acceleration": applied_acceleration,
        }
        return self.state, float(reward), self._done, info

    def _distance_to_goal(self, x: float, y: float) -> float:
        return float(np.linalg.norm(np.asarray([x, y]) - self.goal))

    def _is_out_of_bounds(self, x: float, y: float) -> bool:
        min_x, max_x, min_y, max_y = self.world_bounds
        return (
            x - self.car_radius < min_x
            or x + self.car_radius > max_x
            or y - self.car_radius < min_y
            or y + self.car_radius > max_y
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
