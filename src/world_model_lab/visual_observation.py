"""Deterministic Pillow observations for arbitrary car states."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import PIL

from .car_env import CarEnv


IMAGE_SIZE = 64
RENDERER_VERSION = "pillow-raster-v1"
PILLOW_VERSION = PIL.__version__

LETTERBOX_COLOR = (24, 31, 42)
WORLD_COLOR = (239, 242, 245)
BOUNDARY_COLOR = (38, 50, 56)
OBSTACLE_COLOR = (217, 95, 89)
GOAL_COLOR = (90, 191, 117)
CAR_COLOR = (23, 74, 115)
HEADING_COLOR = (255, 176, 0)

BOUNDARY_WIDTH_PIXELS = 1
MIN_CAR_RADIUS_PIXELS = 2
MIN_HEADING_LENGTH_PIXELS = 4
HEADING_WIDTH_PIXELS = 1


@dataclass(frozen=True)
class SceneGeometry:
    """Immutable scene values read from a `CarEnv`."""

    world_bounds: tuple[float, float, float, float]
    obstacle: tuple[float, float]
    obstacle_radius: float
    goal: tuple[float, float]
    goal_radius: float
    car_radius: float
    dt: float

    def __post_init__(self) -> None:
        values = (
            *self.world_bounds,
            *self.obstacle,
            self.obstacle_radius,
            *self.goal,
            self.goal_radius,
            self.car_radius,
            self.dt,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("scene geometry must contain only finite values")
        min_x, max_x, min_y, max_y = self.world_bounds
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("scene world bounds must have positive extent")
        if min(self.obstacle_radius, self.goal_radius, self.car_radius, self.dt) <= 0:
            raise ValueError("scene radii and dt must be positive")


@dataclass(frozen=True)
class PixelTransform:
    """Aspect-preserving world-to-image transform."""

    world_bounds: tuple[float, float, float, float]
    image_size: int
    scale: float
    offset_x: float
    offset_y: float

    def point_float(self, x: float, y: float) -> tuple[float, float]:
        min_x, _, _, max_y = self.world_bounds
        with np.errstate(over="ignore", invalid="ignore"):
            u = self.offset_x + (float(x) - min_x) * self.scale
            v = self.offset_y + (max_y - float(y)) * self.scale
        return float(u), float(v)

    def point(self, x: float, y: float) -> tuple[int, int]:
        u, v = self.point_float(x, y)
        if not math.isfinite(u) or not math.isfinite(v):
            raise OverflowError("world position maps outside finite pixel coordinates")
        return int(np.rint(u)), int(np.rint(v))

    def radius(self, radius: float, *, minimum: int = 1) -> int:
        value = float(radius)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("world radius must be a positive finite value")
        return max(minimum, int(np.rint(value * self.scale)))


def scene_from_env(env: CarEnv) -> SceneGeometry:
    """Copy public scene values without retaining mutable environment arrays."""

    return SceneGeometry(
        world_bounds=tuple(float(value) for value in env.world_bounds),
        obstacle=tuple(float(value) for value in env.obstacle),
        obstacle_radius=float(env.obstacle_radius),
        goal=tuple(float(value) for value in env.goal),
        goal_radius=float(env.goal_radius),
        car_radius=float(env.car_radius),
        dt=float(env.dt),
    )


def _require_image_size(image_size: int) -> int:
    value = int(image_size)
    if value != IMAGE_SIZE:
        raise ValueError(f"schema version 1 requires image_size={IMAGE_SIZE}")
    return value


def build_pixel_transform(
    scene: SceneGeometry,
    *,
    image_size: int = IMAGE_SIZE,
) -> PixelTransform:
    """Build the common-scale, centred letterbox transform."""

    size = _require_image_size(image_size)
    min_x, max_x, min_y, max_y = scene.world_bounds
    scale = min(
        (size - 1) / (max_x - min_x),
        (size - 1) / (max_y - min_y),
    )
    used_width = (max_x - min_x) * scale
    used_height = (max_y - min_y) * scale
    return PixelTransform(
        world_bounds=scene.world_bounds,
        image_size=size,
        scale=float(scale),
        offset_x=float(((size - 1) - used_width) / 2.0),
        offset_y=float(((size - 1) - used_height) / 2.0),
    )


def world_to_pixel(
    position: Sequence[float],
    *,
    scene: SceneGeometry,
    image_size: int = IMAGE_SIZE,
) -> tuple[int, int]:
    """Map one finite world point to its nearest image pixel."""

    point = np.asarray(position, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("position must contain two finite values")
    return build_pixel_transform(scene, image_size=image_size).point(
        float(point[0]),
        float(point[1]),
    )


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    *,
    centre_float: tuple[float, float],
    radius_pixels: int,
    image_size: int,
    color: tuple[int, int, int],
) -> bool:
    u_float, v_float = centre_float
    if not math.isfinite(u_float) or not math.isfinite(v_float):
        return False
    if (
        u_float + radius_pixels < 0
        or v_float + radius_pixels < 0
        or u_float - radius_pixels > image_size - 1
        or v_float - radius_pixels > image_size - 1
    ):
        return False
    u = int(np.rint(u_float))
    v = int(np.rint(v_float))
    draw.ellipse(
        (
            u - radius_pixels,
            v - radius_pixels,
            u + radius_pixels,
            v + radius_pixels,
        ),
        fill=color,
    )
    return True


def render_observation(
    state: Sequence[float],
    *,
    scene: SceneGeometry,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Render one state without evolving or mutating the environment."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("state must have shape [4]")
    if not np.all(np.isfinite(values)):
        raise ValueError("state must contain only finite values")

    transform = build_pixel_transform(scene, image_size=image_size)
    image = Image.new("RGB", (image_size, image_size), LETTERBOX_COLOR)
    draw = ImageDraw.Draw(image)

    min_x, max_x, min_y, max_y = scene.world_bounds
    world_top_left = transform.point(min_x, max_y)
    world_bottom_right = transform.point(max_x, min_y)
    draw.rectangle(
        (*world_top_left, *world_bottom_right),
        fill=WORLD_COLOR,
        outline=BOUNDARY_COLOR,
        width=BOUNDARY_WIDTH_PIXELS,
    )

    _draw_circle(
        draw,
        centre_float=transform.point_float(*scene.obstacle),
        radius_pixels=transform.radius(scene.obstacle_radius),
        image_size=image_size,
        color=OBSTACLE_COLOR,
    )
    _draw_circle(
        draw,
        centre_float=transform.point_float(*scene.goal),
        radius_pixels=transform.radius(scene.goal_radius),
        image_size=image_size,
        color=GOAL_COLOR,
    )

    x, y, heading, _ = values
    car_centre = transform.point_float(float(x), float(y))
    car_radius = transform.radius(
        scene.car_radius,
        minimum=MIN_CAR_RADIUS_PIXELS,
    )
    car_visible = _draw_circle(
        draw,
        centre_float=car_centre,
        radius_pixels=car_radius,
        image_size=image_size,
        color=CAR_COLOR,
    )
    if car_visible:
        u_float, v_float = car_centre
        u = int(np.rint(u_float))
        v = int(np.rint(v_float))
        marker_length = max(
            MIN_HEADING_LENGTH_PIXELS,
            int(np.rint(car_radius * 2.5)),
        )
        end_u = int(np.rint(u_float + marker_length * math.cos(float(heading))))
        end_v = int(np.rint(v_float - marker_length * math.sin(float(heading))))
        draw.line(
            (u, v, end_u, end_v),
            fill=HEADING_COLOR,
            width=HEADING_WIDTH_PIXELS,
        )

    return np.asarray(image, dtype=np.uint8).copy()
