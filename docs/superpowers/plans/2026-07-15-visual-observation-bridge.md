# Visual Observation Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing transition NPZ into a deterministic, validated, episode-oriented 64×64 RGB artifact with causally aligned actions and a GIF inspection path.

**Architecture:** Add a pure Pillow renderer over immutable scene geometry, then reconstruct canonical episodes from the transition source and materialize each physical frame exactly once. Keep schema construction and validation in `visual_dataset.py`, while `build_visual_data.py` owns only paths, preview encoding, JSON output, and CLI behavior.

**Tech Stack:** Python 3.10+, NumPy 2.0+, Pillow 12.0+, argparse, compressed NPZ, GIF, unittest.

## Global Constraints

- `data/transitions.npz` remains the source of truth and is never modified in place.
- Schema version 1 fixes `image_size=64`, `context_frames=4`, and `renderer_version="pillow-raster-v1"`.
- Schema version 1 supports only the current default `CarEnv` scene geometry.
- Canonical output sorts episode IDs numerically and then sorts each episode by step ID.
- Source continuity uses `np.allclose(..., rtol=0.0, atol=1e-10)`.
- Every episode stores `T + 1` frames beside `T` actions; no context window crosses an episode boundary.
- `frames[k] --actions[k]--> frames[k + 1]` is the causal alignment invariant.
- Velocity is not drawn into an individual frame.
- Rendering must not call `CarEnv.step()`, reset the environment, or mutate state, trajectory, step count, or termination state.
- Scale and letterbox offsets remain floating point; final pixel coordinates and radii use `np.rint`.
- Source and visual NPZ files are always loaded with `allow_pickle=False`.
- Input, NPZ output, and GIF preview paths resolve to three different paths, and existing outputs are never overwritten.
- The artifact stores `renderer_version` and the runtime `pillow_version`; byte determinism is scoped to the same renderer and Pillow versions.
- This plan does not add an encoder, VAE, latent dynamics, visual training, reward model, policy, CUDA, MPS, WorldGym, or World-Gymnast.

## File Map

- Create `src/world_model_lab/visual_observation.py`: immutable scene snapshot, aspect-preserving transform, and RGB renderer.
- Create `src/world_model_lab/visual_dataset.py`: source validation, canonical episode reconstruction, schema construction, schema validation, NPZ I/O, and counts.
- Create `src/world_model_lab/build_visual_data.py`: path checks, preview selection, GIF writing, JSON summary, and CLI.
- Create `tests/visual_fixtures.py`: reusable two-episode deterministic transition fixture.
- Create `tests/test_visual_observation.py`: transform, raster, visibility, clipping, and non-mutation coverage.
- Create `tests/test_visual_dataset.py`: source invariants, canonical reconstruction, schema, offsets, determinism, and loader coverage.
- Create `tests/test_build_visual_data.py`: entry point, paths, preview, end-to-end conversion, and JSON coverage.
- Modify `pyproject.toml`: declare Pillow and register `world-model-build-visual-data`.
- Modify `README.md`: document the visual artifact and the future frame/action-history contract.
- Leave `car_env.py`, `collect_data.py`, `dataset.py`, `visualize.py`, `__init__.py`, and `.gitignore` unchanged.

## Locked Public Interfaces

- `visual_observation.py` exports constants `IMAGE_SIZE: int` and
  `RENDERER_VERSION: str`.
- `SceneGeometry` is a frozen dataclass containing immutable tuple/float scene
  fields; `PixelTransform` is a frozen dataclass containing bounds, image size,
  floating scale, and floating offsets.
- Renderer functions are `scene_from_env(env: CarEnv) -> SceneGeometry`,
  `build_pixel_transform(scene: SceneGeometry, *, image_size: int = IMAGE_SIZE)
  -> PixelTransform`, `world_to_pixel(position: Sequence[float], *, scene:
  SceneGeometry, image_size: int = IMAGE_SIZE) -> tuple[int, int]`, and
  `render_observation(state: Sequence[float], *, scene: SceneGeometry,
  image_size: int = IMAGE_SIZE) -> np.ndarray`.
- `visual_dataset.py` exports aliases `TransitionDataset` and `VisualDataset`,
  plus the frozen `OrderedEpisode` dataclass.
- Dataset functions are `load_transition_dataset(path)`,
  `reconstruct_episodes(source)`, `build_visual_dataset(source)`,
  `validate_visual_dataset(dataset)`, `save_visual_dataset(dataset, output)`,
  `load_visual_dataset(path)`, and `summarize_visual_dataset(dataset)` with the
  return types shown in their implementation tasks.
- CLI functions are `write_preview_gif(dataset, output_path, *,
  episode_id=None) -> int`, `run_visual_data_build(*, data_path, output_path,
  preview_path, preview_episode_id=None) -> dict[str, object]`, and
  `main() -> None`.

---

### Task 1: Add the deterministic arbitrary-state RGB renderer

**Files:**

- Create: `src/world_model_lab/visual_observation.py`
- Create: `tests/test_visual_observation.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: public geometry fields from `CarEnv` and an arbitrary finite `[x, y, heading, velocity]` state.
- Produces: `SceneGeometry`, `PixelTransform`, `scene_from_env()`, `build_pixel_transform()`, `world_to_pixel()`, and `render_observation() -> uint8[64,64,3]`.

- [ ] **Step 1: Write failing transform and dependency tests**

Create `tests/test_visual_observation.py` with the first test group:

```python
import unittest
from pathlib import Path

import numpy as np

from world_model_lab.car_env import CarEnv
from world_model_lab.visual_observation import (
    IMAGE_SIZE,
    build_pixel_transform,
    scene_from_env,
    world_to_pixel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisualObservationTest(unittest.TestCase):
    def setUp(self):
        self.env = CarEnv()
        self.scene = scene_from_env(self.env)

    def test_pillow_is_an_explicit_dependency(self):
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"Pillow>=12.0"', pyproject)

    def test_default_transform_preserves_aspect_ratio_and_centres_letterbox(self):
        transform = build_pixel_transform(self.scene)

        self.assertEqual(IMAGE_SIZE, 64)
        self.assertAlmostEqual(transform.scale, 6.3)
        self.assertAlmostEqual(transform.offset_x, 0.0)
        self.assertAlmostEqual(transform.offset_y, 6.3)
        self.assertEqual(
            world_to_pixel((0.0, 8.0), scene=self.scene),
            (0, 6),
        )
        self.assertEqual(
            world_to_pixel((10.0, 0.0), scene=self.scene),
            (63, 57),
        )
        self.assertEqual(
            world_to_pixel((5.0, 4.0), scene=self.scene),
            (32, 32),
        )
        self.assertEqual(
            world_to_pixel((9.0, 7.0), scene=self.scene),
            (57, 13),
        )

    def test_world_axes_map_rightward_and_upward(self):
        left = world_to_pixel((2.0, 2.0), scene=self.scene)
        right = world_to_pixel((3.0, 2.0), scene=self.scene)
        low = world_to_pixel((2.0, 2.0), scene=self.scene)
        high = world_to_pixel((2.0, 3.0), scene=self.scene)

        self.assertGreater(right[0], left[0])
        self.assertLess(high[1], low[1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the transform tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_observation -v
```

Expected: FAIL because `world_model_lab.visual_observation` does not exist and `Pillow` is not explicit in `pyproject.toml`.

- [ ] **Step 3: Declare Pillow and implement immutable geometry plus transform**

Add this dependency to `pyproject.toml`:

```toml
dependencies = [
  "numpy>=2.0",
  "matplotlib>=3.8",
  "torch>=2.6",
  "Pillow>=12.0",
]
```

Create `src/world_model_lab/visual_observation.py` with:

```python
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
```

- [ ] **Step 4: Run the transform tests and confirm GREEN**

Run the command from Step 2.

Expected: 3 tests PASS.

- [ ] **Step 5: Add failing raster, hidden-velocity, clipping, and mutation tests**

Extend the import list in `tests/test_visual_observation.py` with `math` and:

```python
from world_model_lab.visual_observation import (
    BOUNDARY_COLOR,
    CAR_COLOR,
    GOAL_COLOR,
    HEADING_COLOR,
    IMAGE_SIZE,
    OBSTACLE_COLOR,
    build_pixel_transform,
    render_observation,
    scene_from_env,
    world_to_pixel,
)
```

Add these methods to `VisualObservationTest`:

```python
    def test_render_returns_exact_rgb_shape_and_is_repeatable(self):
        state = np.asarray([2.0, 2.0, 0.25, 1.5], dtype=np.float64)

        first = render_observation(state, scene=self.scene)
        second = render_observation(state, scene=self.scene)

        self.assertEqual(first.shape, (64, 64, 3))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)

    def test_default_scene_layers_occupy_expected_pixels(self):
        state = np.asarray([2.0, 2.0, 0.0, 1.0], dtype=np.float64)
        frame = render_observation(state, scene=self.scene)
        obstacle = world_to_pixel(self.scene.obstacle, scene=self.scene)
        goal = world_to_pixel(self.scene.goal, scene=self.scene)
        car = world_to_pixel(state[:2], scene=self.scene)

        self.assertEqual(tuple(frame[6, 0]), BOUNDARY_COLOR)
        self.assertEqual(tuple(frame[obstacle[1], obstacle[0]]), OBSTACLE_COLOR)
        self.assertEqual(tuple(frame[goal[1], goal[0]]), GOAL_COLOR)
        self.assertEqual(tuple(frame[car[1], car[0] - 1]), CAR_COLOR)
        self.assertEqual(tuple(frame[car[1], car[0] + 5]), HEADING_COLOR)

    def test_heading_rotates_marker_without_moving_car(self):
        east = render_observation((2.0, 2.0, 0.0, 1.0), scene=self.scene)
        north = render_observation(
            (2.0, 2.0, math.pi / 2.0, 1.0),
            scene=self.scene,
        )
        car = world_to_pixel((2.0, 2.0), scene=self.scene)

        self.assertEqual(tuple(east[car[1], car[0] + 5]), HEADING_COLOR)
        self.assertEqual(tuple(north[car[1] - 5, car[0]]), HEADING_COLOR)
        self.assertEqual(tuple(east[car[1], car[0] - 1]), CAR_COLOR)
        self.assertEqual(tuple(north[car[1], car[0] - 1]), CAR_COLOR)

    def test_velocity_only_change_is_pixel_identical(self):
        stopped = render_observation((2.0, 2.0, 0.3, 0.0), scene=self.scene)
        moving = render_observation((2.0, 2.0, 0.3, 3.0), scene=self.scene)

        np.testing.assert_array_equal(stopped, moving)

    def test_world_space_obstacle_is_circular_in_pixels(self):
        frame = render_observation((1.0, 1.0, 0.0, 0.0), scene=self.scene)
        mask = np.all(frame == np.asarray(OBSTACLE_COLOR), axis=2)
        rows, columns = np.where(mask)

        self.assertLessEqual(
            abs((columns.max() - columns.min()) - (rows.max() - rows.min())),
            1,
        )

    def test_near_boundary_car_is_clipped_without_state_clamping(self):
        state = np.asarray([-0.1, 1.0, 0.0, 1.0], dtype=np.float64)
        original = state.copy()

        frame = render_observation(state, scene=self.scene)

        np.testing.assert_array_equal(state, original)
        self.assertTrue(np.any(np.all(frame == np.asarray(CAR_COLOR), axis=2)))

    def test_large_finite_offscreen_state_does_not_overflow(self):
        frame = render_observation(
            (np.finfo(np.float64).max, 1.0, 0.0, 1.0),
            scene=self.scene,
        )

        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertFalse(np.any(np.all(frame == np.asarray(CAR_COLOR), axis=2)))
        self.assertFalse(np.any(np.all(frame == np.asarray(HEADING_COLOR), axis=2)))

    def test_invalid_state_shape_and_non_finite_state_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"state must have shape \[4\]"):
            render_observation((1.0, 2.0, 0.0), scene=self.scene)
        with self.assertRaisesRegex(ValueError, "state.*finite"):
            render_observation(
                (1.0, 2.0, np.nan, 0.0),
                scene=self.scene,
            )

    def test_rendering_and_scene_snapshot_do_not_mutate_environment(self):
        self.env.step(0.0, 0.0)
        before_state = self.env.state
        before_trajectory = self.env.trajectory
        before_steps = self.env.steps
        before_done = self.env.done

        scene = scene_from_env(self.env)
        render_observation((3.0, 2.0, 0.4, 2.0), scene=scene)

        np.testing.assert_array_equal(self.env.state, before_state)
        self.assertEqual(self.env.steps, before_steps)
        self.assertEqual(self.env.done, before_done)
        self.assertEqual(len(self.env.trajectory), len(before_trajectory))
        for actual, expected in zip(self.env.trajectory, before_trajectory, strict=True):
            np.testing.assert_array_equal(actual, expected)
```

- [ ] **Step 6: Run the raster tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_observation -v
```

Expected: the new tests fail because `render_observation()` is not defined.

- [ ] **Step 7: Implement fixed raster layers with safe offscreen clipping**

Append these helpers and the public renderer to `visual_observation.py`:

```python
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
```

- [ ] **Step 8: Run renderer and existing visualization tests**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_observation tests.test_visualize tests.test_car_env -v
```

Expected: all tests PASS, proving the new renderer did not alter the existing Matplotlib or physics paths.

- [ ] **Step 9: Commit the renderer**

```bash
git add pyproject.toml src/world_model_lab/visual_observation.py tests/test_visual_observation.py
git commit -m "feat: add deterministic visual observations"
```

### Task 2: Validate transition sources and reconstruct canonical episodes

**Files:**

- Create: `src/world_model_lab/visual_dataset.py`
- Create: `tests/visual_fixtures.py`
- Create: `tests/test_visual_dataset.py`

**Interfaces:**

- Consumes: the eight arrays from `collect_transitions()`.
- Produces: `OrderedEpisode`, `load_transition_dataset()`, and `reconstruct_episodes()` with episode-ID/step-ID canonical ordering.

- [ ] **Step 1: Create an exact reusable transition fixture**

Create `tests/visual_fixtures.py`:

```python
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
```

- [ ] **Step 2: Write canonical-order and load-safety tests**

Create `tests/test_visual_dataset.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.visual_fixtures import clone_arrays, make_transition_source
from world_model_lab.visual_dataset import (
    load_transition_dataset,
    reconstruct_episodes,
)


class VisualSourceValidationTest(unittest.TestCase):
    def setUp(self):
        self.source = make_transition_source()

    def test_reconstructs_episodes_in_canonical_id_and_step_order(self):
        episodes = reconstruct_episodes(self.source)

        self.assertEqual([episode.episode_id for episode in episodes], [3, 7])
        np.testing.assert_array_equal(episodes[0].step_ids, [0, 1])
        np.testing.assert_array_equal(episodes[1].step_ids, [0, 1, 2, 3])
        self.assertEqual(episodes[0].states.shape, (2, 4))
        self.assertEqual(episodes[1].actions.shape, (4, 2))

    def test_shuffled_rows_reconstruct_identically(self):
        permutation = np.asarray([5, 1, 4, 0, 3, 2])
        shuffled = {
            name: values[permutation]
            for name, values in self.source.items()
        }

        expected = reconstruct_episodes(self.source)
        actual = reconstruct_episodes(shuffled)

        for expected_episode, actual_episode in zip(
            expected,
            actual,
            strict=True,
        ):
            self.assertEqual(actual_episode.episode_id, expected_episode.episode_id)
            for field in (
                "states",
                "actions",
                "next_states",
                "rewards",
                "dones",
                "step_ids",
                "terminal_reasons",
            ):
                np.testing.assert_array_equal(
                    getattr(actual_episode, field),
                    getattr(expected_episode, field),
                )

    def test_loader_uses_allow_pickle_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object-source.npz"
            unsafe = clone_arrays(self.source)
            unsafe["states"] = np.asarray([object()], dtype=object)
            np.savez_compressed(path, **unsafe)

            with self.assertRaisesRegex(ValueError, "Object arrays"):
                load_transition_dataset(path)

    def test_loader_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.npz"
            with self.assertRaisesRegex(FileNotFoundError, "regular file"):
                load_transition_dataset(path)
```

- [ ] **Step 3: Run the source tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_dataset.VisualSourceValidationTest -v
```

Expected: FAIL because `world_model_lab.visual_dataset` does not exist.

- [ ] **Step 4: Implement exact source shapes, dtypes, ordering, and continuity**

Create `src/world_model_lab/visual_dataset.py`:

```python
"""Episode-oriented visual artifacts built from transition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


TransitionDataset = dict[str, np.ndarray]
VisualDataset = dict[str, np.ndarray]

REQUIRED_SOURCE_ARRAYS = (
    "states",
    "actions",
    "next_states",
    "rewards",
    "dones",
    "episode_ids",
    "step_ids",
    "terminal_reasons",
)
VALID_TERMINAL_REASONS = frozenset(
    {"goal", "collision", "out_of_bounds", "time_limit"}
)


@dataclass(frozen=True)
class OrderedEpisode:
    episode_id: int
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    step_ids: np.ndarray
    terminal_reasons: np.ndarray

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])


def load_transition_dataset(path: Path | str) -> TransitionDataset:
    """Load required source arrays without enabling pickle."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"transition dataset is not a regular file: {source_path}"
        )
    with np.load(source_path, allow_pickle=False) as loaded:
        missing = set(REQUIRED_SOURCE_ARRAYS) - set(loaded.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"transition dataset is missing arrays: {names}")
        source = {
            name: np.asarray(loaded[name])
            for name in REQUIRED_SOURCE_ARRAYS
        }
    reconstruct_episodes(source)
    return source


def _require_numeric_shape(
    source: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(source[name])
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if values.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric")
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def reconstruct_episodes(
    source: Mapping[str, np.ndarray],
) -> tuple[OrderedEpisode, ...]:
    """Validate and return episodes sorted by ID and step."""

    missing = set(REQUIRED_SOURCE_ARRAYS) - set(source)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"transition dataset is missing arrays: {names}")

    raw_states = np.asarray(source["states"])
    if raw_states.ndim != 2:
        raise ValueError("states must have shape [N, 4]")
    count = int(raw_states.shape[0])
    if count == 0:
        raise ValueError("transition dataset must contain at least one row")

    states = _require_numeric_shape(source, "states", (count, 4))
    actions = _require_numeric_shape(source, "actions", (count, 2))
    next_states = _require_numeric_shape(source, "next_states", (count, 4))
    rewards = _require_numeric_shape(source, "rewards", (count,))

    dones = np.asarray(source["dones"])
    if dones.shape != (count,) or dones.dtype.kind != "b":
        raise ValueError("dones must be a boolean array with shape [N]")

    episode_ids = np.asarray(source["episode_ids"])
    step_ids = np.asarray(source["step_ids"])
    for name, values in (("episode_ids", episode_ids), ("step_ids", step_ids)):
        if values.shape != (count,) or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer array with shape [N]")

    terminal_reasons = np.asarray(source["terminal_reasons"])
    if terminal_reasons.shape != (count,) or terminal_reasons.dtype.kind != "U":
        raise ValueError(
            "terminal_reasons must be a Unicode array with shape [N]"
        )

    episodes: list[OrderedEpisode] = []
    for raw_episode_id in np.unique(episode_ids):
        episode_id = int(raw_episode_id)
        row_indices = np.flatnonzero(episode_ids == raw_episode_id)
        order = np.argsort(step_ids[row_indices], kind="stable")
        indices = row_indices[order]
        ordered_steps = np.asarray(step_ids[indices], dtype=np.int64)
        expected_steps = np.arange(indices.size, dtype=np.int64)
        if not np.array_equal(ordered_steps, expected_steps):
            raise ValueError(
                f"episode {episode_id} step_ids must be unique, non-negative, "
                "and contiguous from zero"
            )

        episode_states = np.asarray(states[indices], dtype=np.float64)
        episode_next_states = np.asarray(next_states[indices], dtype=np.float64)
        if indices.size > 1 and not np.allclose(
            episode_states[1:],
            episode_next_states[:-1],
            rtol=0.0,
            atol=1e-10,
        ):
            matching = np.all(
                np.isclose(
                    episode_states[1:],
                    episode_next_states[:-1],
                    rtol=0.0,
                    atol=1e-10,
                ),
                axis=1,
            )
            failing_step = int(np.flatnonzero(~matching)[0])
            raise ValueError(
                f"episode {episode_id} is discontinuous after step {failing_step}"
            )

        episode_dones = np.asarray(dones[indices], dtype=np.bool_)
        episode_reasons = np.asarray(terminal_reasons[indices], dtype=np.str_)
        if np.any(episode_dones[:-1]):
            failing_step = int(np.flatnonzero(episode_dones[:-1])[0])
            raise ValueError(
                f"episode {episode_id} terminates before its final row "
                f"at step {failing_step}"
            )
        if not bool(episode_dones[-1]):
            raise ValueError(f"episode {episode_id} final row is not terminal")
        if np.any(episode_reasons[:-1] != ""):
            failing_step = int(np.flatnonzero(episode_reasons[:-1] != "")[0])
            raise ValueError(
                f"episode {episode_id} has terminal reason before final row "
                f"at step {failing_step}"
            )
        if str(episode_reasons[-1]) not in VALID_TERMINAL_REASONS:
            raise ValueError(
                f"episode {episode_id} final terminal reason is invalid: "
                f"{episode_reasons[-1]!s}"
            )

        episodes.append(
            OrderedEpisode(
                episode_id=episode_id,
                states=episode_states.copy(),
                actions=np.asarray(actions[indices], dtype=np.float64).copy(),
                next_states=episode_next_states.copy(),
                rewards=np.asarray(rewards[indices], dtype=np.float64).copy(),
                dones=episode_dones.copy(),
                step_ids=ordered_steps.copy(),
                terminal_reasons=episode_reasons.copy(),
            )
        )
    return tuple(episodes)
```

- [ ] **Step 5: Run the canonical-order tests and confirm GREEN**

Run the command from Step 3.

Expected: 4 tests PASS.

- [ ] **Step 6: Add focused malformed-source regression tests**

Add these methods to `VisualSourceValidationTest`:

```python
    def test_missing_required_array_names_the_array(self):
        source = clone_arrays(self.source)
        source.pop("actions")

        with self.assertRaisesRegex(ValueError, "missing arrays: actions"):
            reconstruct_episodes(source)

    def test_non_finite_numeric_value_names_the_array(self):
        source = clone_arrays(self.source)
        source["states"][0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "states.*finite"):
            reconstruct_episodes(source)

    def test_invalid_action_shape_names_the_array(self):
        source = clone_arrays(self.source)
        source["actions"] = source["actions"][:, :1]

        with self.assertRaisesRegex(ValueError, r"actions.*\[6, 2\]"):
            reconstruct_episodes(source)

    def test_float_step_ids_are_rejected(self):
        source = clone_arrays(self.source)
        source["step_ids"] = source["step_ids"].astype(np.float64)

        with self.assertRaisesRegex(ValueError, "step_ids.*integer"):
            reconstruct_episodes(source)

    def test_duplicate_missing_or_negative_steps_are_rejected(self):
        for replacement in (
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([-1, 0], dtype=np.int64),
        ):
            with self.subTest(replacement=replacement):
                source = clone_arrays(self.source)
                mask = source["episode_ids"] == 3
                source["step_ids"][mask] = replacement
                with self.assertRaisesRegex(
                    ValueError,
                    "episode 3 step_ids.*contiguous",
                ):
                    reconstruct_episodes(source)

    def test_transition_discontinuity_names_episode_and_step(self):
        source = clone_arrays(self.source)
        first_row = np.flatnonzero(
            (source["episode_ids"] == 7) & (source["step_ids"] == 0)
        )[0]
        source["next_states"][first_row, 0] += 1e-3

        with self.assertRaisesRegex(
            ValueError,
            "episode 7 is discontinuous after step 0",
        ):
            reconstruct_episodes(source)

    def test_early_terminal_names_episode_and_step(self):
        source = clone_arrays(self.source)
        first_row = np.flatnonzero(
            (source["episode_ids"] == 7) & (source["step_ids"] == 0)
        )[0]
        source["dones"][first_row] = True

        with self.assertRaisesRegex(
            ValueError,
            "episode 7 terminates.*step 0",
        ):
            reconstruct_episodes(source)

    def test_non_terminal_final_row_names_episode(self):
        source = clone_arrays(self.source)
        final_row = np.flatnonzero(
            (source["episode_ids"] == 3) & (source["step_ids"] == 1)
        )[0]
        source["dones"][final_row] = False
        source["terminal_reasons"][final_row] = ""

        with self.assertRaisesRegex(ValueError, "episode 3 final row"):
            reconstruct_episodes(source)

    def test_invalid_terminal_reason_names_episode(self):
        source = clone_arrays(self.source)
        final_row = np.flatnonzero(
            (source["episode_ids"] == 3) & (source["step_ids"] == 1)
        )[0]
        source["terminal_reasons"][final_row] = "unknown"

        with self.assertRaisesRegex(
            ValueError,
            "episode 3 final terminal reason is invalid",
        ):
            reconstruct_episodes(source)
```

- [ ] **Step 7: Run the complete source-validation test class**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_dataset.VisualSourceValidationTest -v
```

Expected: all source-validation tests PASS with array-, episode-, and step-specific messages.

- [ ] **Step 8: Commit canonical episode reconstruction**

```bash
git add src/world_model_lab/visual_dataset.py tests/visual_fixtures.py tests/test_visual_dataset.py
git commit -m "feat: validate visual dataset sources"
```

### Task 3: Build, validate, save, and load schema-version-1 visual episodes

**Files:**

- Modify: `src/world_model_lab/visual_dataset.py`
- Modify: `tests/test_visual_dataset.py`

**Interfaces:**

- Consumes: the canonical `OrderedEpisode` tuple and `render_observation()` from Task 1.
- Produces: `build_visual_dataset()`, `validate_visual_dataset()`, `save_visual_dataset()`, `load_visual_dataset()`, and `summarize_visual_dataset()`.

- [ ] **Step 1: Write failing artifact construction and alignment tests**

Extend the imports in `tests/test_visual_dataset.py`:

```python
from world_model_lab.car_env import CarEnv
from world_model_lab.visual_dataset import (
    build_visual_dataset,
    load_transition_dataset,
    reconstruct_episodes,
    summarize_visual_dataset,
)
from world_model_lab.visual_observation import (
    PILLOW_VERSION,
    RENDERER_VERSION,
    render_observation,
    scene_from_env,
)
```

Add this test class:

```python
class VisualArtifactTest(unittest.TestCase):
    def setUp(self):
        self.source = make_transition_source()
        self.dataset = build_visual_dataset(self.source)

    def test_builds_n_plus_e_frames_and_exact_offsets(self):
        self.assertEqual(self.dataset["episode_ids"].tolist(), [3, 7])
        np.testing.assert_array_equal(
            self.dataset["frame_offsets"],
            [0, 3, 8],
        )
        np.testing.assert_array_equal(
            self.dataset["transition_offsets"],
            [0, 2, 6],
        )
        self.assertEqual(self.dataset["frames"].shape, (8, 64, 64, 3))
        self.assertEqual(self.dataset["states"].shape, (8, 4))
        self.assertEqual(self.dataset["actions"].shape, (6, 2))

    def test_frames_actions_and_states_preserve_causal_alignment(self):
        scene = scene_from_env(CarEnv())
        episodes = reconstruct_episodes(self.source)

        for episode_index, episode in enumerate(episodes):
            frame_start, frame_stop = self.dataset["frame_offsets"][
                episode_index : episode_index + 2
            ]
            transition_start, transition_stop = self.dataset[
                "transition_offsets"
            ][episode_index : episode_index + 2]
            aligned_states = np.concatenate(
                (episode.states[:1], episode.next_states),
                axis=0,
            )

            np.testing.assert_array_equal(
                self.dataset["states"][frame_start:frame_stop],
                aligned_states,
            )
            np.testing.assert_array_equal(
                self.dataset["actions"][transition_start:transition_stop],
                episode.actions,
            )
            for local_index, state in enumerate(aligned_states):
                expected_frame = render_observation(state, scene=scene)
                np.testing.assert_array_equal(
                    self.dataset["frames"][frame_start + local_index],
                    expected_frame,
                )

    def test_terminal_metadata_stays_on_transitions(self):
        episodes = reconstruct_episodes(self.source)

        for episode_index, episode in enumerate(episodes):
            start, stop = self.dataset["transition_offsets"][
                episode_index : episode_index + 2
            ]
            np.testing.assert_array_equal(
                self.dataset["rewards"][start:stop],
                episode.rewards,
            )
            np.testing.assert_array_equal(
                self.dataset["dones"][start:stop],
                episode.dones,
            )
            np.testing.assert_array_equal(
                self.dataset["terminal_reasons"][start:stop],
                episode.terminal_reasons,
            )

    def test_schema_and_scene_metadata_are_exact(self):
        self.assertEqual(self.dataset["schema_version"].item(), 1)
        self.assertEqual(self.dataset["image_size"].item(), 64)
        self.assertEqual(self.dataset["context_frames"].item(), 4)
        self.assertEqual(
            self.dataset["renderer_version"].item(),
            RENDERER_VERSION,
        )
        self.assertEqual(
            self.dataset["pillow_version"].item(),
            PILLOW_VERSION,
        )
        np.testing.assert_array_equal(
            self.dataset["scene_world_bounds"],
            [0.0, 10.0, 0.0, 8.0],
        )
        self.assertEqual(self.dataset["scene_dt"].item(), 0.1)

    def test_short_episodes_are_retained_but_not_eligible(self):
        summary = summarize_visual_dataset(self.dataset)

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["transitions"], 6)
        self.assertEqual(summary["frames"], 8)
        self.assertEqual(summary["four_frame_eligible_episodes"], 1)
        self.assertEqual(summary["one_step_visual_samples"], 1)

    def test_same_source_produces_identical_arrays_and_metadata(self):
        second = build_visual_dataset(self.source)

        self.assertEqual(set(self.dataset), set(second))
        for name in self.dataset:
            np.testing.assert_array_equal(self.dataset[name], second[name])

    def test_shuffled_source_rows_produce_the_same_complete_artifact(self):
        permutation = np.asarray([5, 1, 4, 0, 3, 2])
        shuffled = {
            name: values[permutation]
            for name, values in self.source.items()
        }

        actual = build_visual_dataset(shuffled)

        self.assertEqual(set(self.dataset), set(actual))
        for name in self.dataset:
            np.testing.assert_array_equal(self.dataset[name], actual[name])
```

- [ ] **Step 2: Run artifact construction tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_dataset.VisualArtifactTest -v
```

Expected: FAIL because visual schema construction and summary functions do not exist.

- [ ] **Step 3: Implement preallocated frame construction and exact metadata**

Add these imports and constants near the top of `visual_dataset.py`:

```python
from .car_env import CarEnv
from .visual_observation import (
    IMAGE_SIZE,
    PILLOW_VERSION,
    RENDERER_VERSION,
    render_observation,
    scene_from_env,
)


VISUAL_SCHEMA_VERSION = 1
CONTEXT_FRAMES = 4
TERMINAL_REASON_DTYPE = np.dtype("<U13")
REQUIRED_VISUAL_ARRAYS = (
    "schema_version",
    "image_size",
    "context_frames",
    "renderer_version",
    "pillow_version",
    "frames",
    "states",
    "actions",
    "rewards",
    "dones",
    "terminal_reasons",
    "episode_ids",
    "frame_offsets",
    "transition_offsets",
    "scene_world_bounds",
    "scene_obstacle",
    "scene_obstacle_radius",
    "scene_goal",
    "scene_goal_radius",
    "scene_car_radius",
    "scene_dt",
)
```

Append these functions:

```python
def build_visual_dataset(
    source: Mapping[str, np.ndarray],
) -> VisualDataset:
    """Render canonical episodes into one flattened schema-v1 artifact."""

    episodes = reconstruct_episodes(source)
    scene = scene_from_env(CarEnv())
    episode_count = len(episodes)
    transition_count = sum(
        episode.transition_count for episode in episodes
    )
    frame_count = transition_count + episode_count

    frames = np.empty(
        (frame_count, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )
    aligned_states = np.empty((frame_count, 4), dtype=np.float64)
    actions = np.empty((transition_count, 2), dtype=np.float64)
    rewards = np.empty((transition_count,), dtype=np.float64)
    dones = np.empty((transition_count,), dtype=np.bool_)
    terminal_reasons = np.empty(
        (transition_count,),
        dtype=TERMINAL_REASON_DTYPE,
    )
    episode_ids = np.empty((episode_count,), dtype=np.int64)
    frame_offsets = np.zeros((episode_count + 1,), dtype=np.int64)
    transition_offsets = np.zeros((episode_count + 1,), dtype=np.int64)

    frame_cursor = 0
    transition_cursor = 0
    for episode_index, episode in enumerate(episodes):
        physical_states = np.concatenate(
            (episode.states[:1], episode.next_states),
            axis=0,
        )
        next_frame_cursor = frame_cursor + physical_states.shape[0]
        next_transition_cursor = (
            transition_cursor + episode.transition_count
        )
        aligned_states[frame_cursor:next_frame_cursor] = physical_states
        for local_index, state in enumerate(physical_states):
            frames[frame_cursor + local_index] = render_observation(
                state,
                scene=scene,
            )
        actions[transition_cursor:next_transition_cursor] = episode.actions
        rewards[transition_cursor:next_transition_cursor] = episode.rewards
        dones[transition_cursor:next_transition_cursor] = episode.dones
        terminal_reasons[
            transition_cursor:next_transition_cursor
        ] = episode.terminal_reasons
        episode_ids[episode_index] = episode.episode_id

        frame_cursor = next_frame_cursor
        transition_cursor = next_transition_cursor
        frame_offsets[episode_index + 1] = frame_cursor
        transition_offsets[episode_index + 1] = transition_cursor

    dataset: VisualDataset = {
        "schema_version": np.asarray(
            VISUAL_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "image_size": np.asarray(IMAGE_SIZE, dtype=np.int64),
        "context_frames": np.asarray(CONTEXT_FRAMES, dtype=np.int64),
        "renderer_version": np.asarray(RENDERER_VERSION, dtype=np.str_),
        "pillow_version": np.asarray(PILLOW_VERSION, dtype=np.str_),
        "frames": frames,
        "states": aligned_states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "terminal_reasons": terminal_reasons,
        "episode_ids": episode_ids,
        "frame_offsets": frame_offsets,
        "transition_offsets": transition_offsets,
        "scene_world_bounds": np.asarray(
            scene.world_bounds,
            dtype=np.float64,
        ),
        "scene_obstacle": np.asarray(scene.obstacle, dtype=np.float64),
        "scene_obstacle_radius": np.asarray(
            scene.obstacle_radius,
            dtype=np.float64,
        ),
        "scene_goal": np.asarray(scene.goal, dtype=np.float64),
        "scene_goal_radius": np.asarray(
            scene.goal_radius,
            dtype=np.float64,
        ),
        "scene_car_radius": np.asarray(
            scene.car_radius,
            dtype=np.float64,
        ),
        "scene_dt": np.asarray(scene.dt, dtype=np.float64),
    }
    validate_visual_dataset(dataset)
    return dataset


def summarize_visual_dataset(
    dataset: Mapping[str, np.ndarray],
) -> dict[str, int | str]:
    """Return JSON-safe schema and temporal-window counts."""

    validate_visual_dataset(dataset)
    transition_lengths = np.diff(dataset["transition_offsets"])
    return {
        "schema_version": int(dataset["schema_version"].item()),
        "renderer_version": str(dataset["renderer_version"].item()),
        "image_size": int(dataset["image_size"].item()),
        "context_frames": int(dataset["context_frames"].item()),
        "episodes": int(dataset["episode_ids"].size),
        "transitions": int(dataset["actions"].shape[0]),
        "frames": int(dataset["frames"].shape[0]),
        "four_frame_eligible_episodes": int(
            np.count_nonzero(transition_lengths >= CONTEXT_FRAMES)
        ),
        "one_step_visual_samples": int(
            np.maximum(0, transition_lengths - (CONTEXT_FRAMES - 1)).sum()
        ),
    }
```

Temporarily add this minimal validator immediately before `build_visual_dataset()` so the happy-path tests can drive construction:

```python
def validate_visual_dataset(dataset: Mapping[str, np.ndarray]) -> None:
    missing = set(REQUIRED_VISUAL_ARRAYS) - set(dataset)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"visual dataset is missing arrays: {names}")
```

- [ ] **Step 4: Run artifact construction tests and confirm GREEN**

Run the command from Step 2.

Expected: 7 tests PASS with `N=6`, `E=2`, `F=8`, one eligible episode, and one visual training sample.

- [ ] **Step 5: Write failing schema validation and round-trip tests**

Extend the `world_model_lab.visual_dataset` imports in `tests/test_visual_dataset.py` with:

```python
    load_visual_dataset,
    save_visual_dataset,
    validate_visual_dataset,
```

Add these methods to `VisualArtifactTest`:

```python
    def test_visual_artifact_round_trips_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-episodes.npz"
            save_visual_dataset(self.dataset, path)
            loaded = load_visual_dataset(path)

        self.assertEqual(set(loaded), set(self.dataset))
        for name in self.dataset:
            np.testing.assert_array_equal(loaded[name], self.dataset[name])

    def test_save_refuses_to_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-episodes.npz"
            save_visual_dataset(self.dataset, path)
            before = path.read_bytes()

            with self.assertRaises(FileExistsError):
                save_visual_dataset(self.dataset, path)

            self.assertEqual(path.read_bytes(), before)

    def test_loader_uses_allow_pickle_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe-visual.npz"
            unsafe = clone_arrays(self.dataset)
            unsafe["renderer_version"] = np.asarray(object(), dtype=object)
            np.savez_compressed(path, **unsafe)

            with self.assertRaisesRegex(ValueError, "Object arrays"):
                load_visual_dataset(path)

    def test_missing_visual_field_is_rejected(self):
        broken = clone_arrays(self.dataset)
        broken.pop("actions")

        with self.assertRaisesRegex(
            ValueError,
            "visual dataset is missing arrays: actions",
        ):
            validate_visual_dataset(broken)

    def test_unsupported_schema_version_is_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["schema_version"] = np.asarray(2, dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "unsupported schema_version: 2"):
            validate_visual_dataset(broken)

    def test_frame_dtype_and_shape_are_strict(self):
        wrong_dtype = clone_arrays(self.dataset)
        wrong_dtype["frames"] = wrong_dtype["frames"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "frames must have dtype uint8"):
            validate_visual_dataset(wrong_dtype)

        wrong_shape = clone_arrays(self.dataset)
        wrong_shape["frames"] = wrong_shape["frames"][:, :, :, :2]
        with self.assertRaisesRegex(
            ValueError,
            r"frames must have shape \[F, 64, 64, 3\]",
        ):
            validate_visual_dataset(wrong_shape)

    def test_empty_renderer_or_pillow_version_is_rejected(self):
        for field in ("renderer_version", "pillow_version"):
            with self.subTest(field=field):
                broken = clone_arrays(self.dataset)
                broken[field] = np.asarray("", dtype=np.str_)
                with self.assertRaisesRegex(ValueError, f"{field}.*non-empty"):
                    validate_visual_dataset(broken)

    def test_invalid_offsets_are_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["frame_offsets"] = np.asarray([0, 4, 8], dtype=np.int64)

        with self.assertRaisesRegex(
            ValueError,
            "episode 3 must own T plus one frames",
        ):
            validate_visual_dataset(broken)

    def test_noncanonical_episode_ids_are_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["episode_ids"] = np.asarray([7, 3], dtype=np.int64)

        with self.assertRaisesRegex(
            ValueError,
            "episode_ids must be strictly increasing",
        ):
            validate_visual_dataset(broken)

    def test_nondefault_scene_metadata_is_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["scene_goal"] = np.asarray([8.0, 7.0], dtype=np.float64)

        with self.assertRaisesRegex(
            ValueError,
            "scene_goal does not match the schema-v1 default scene",
        ):
            validate_visual_dataset(broken)
```

- [ ] **Step 6: Run validation tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_dataset.VisualArtifactTest -v
```

Expected: the round-trip imports fail and malformed artifacts pass the temporary validator.

- [ ] **Step 7: Replace the temporary validator with the complete schema invariant**

Replace `validate_visual_dataset()` with these helpers and function:

```python
def _scalar(
    dataset: Mapping[str, np.ndarray],
    name: str,
) -> object:
    values = np.asarray(dataset[name])
    if values.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return values.item()


def _require_dtype(
    dataset: Mapping[str, np.ndarray],
    name: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    values = np.asarray(dataset[name])
    expected = np.dtype(dtype)
    if values.dtype != expected:
        raise ValueError(f"{name} must have dtype {expected.name}")
    return values


def validate_visual_dataset(dataset: Mapping[str, np.ndarray]) -> None:
    """Reject unsupported, malformed, or misaligned visual artifacts."""

    missing = set(REQUIRED_VISUAL_ARRAYS) - set(dataset)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"visual dataset is missing arrays: {names}")

    schema_values = _require_dtype(dataset, "schema_version", np.int64)
    image_size_values = _require_dtype(dataset, "image_size", np.int64)
    context_values = _require_dtype(dataset, "context_frames", np.int64)
    schema_version = int(_scalar(dataset, "schema_version"))
    image_size = int(_scalar(dataset, "image_size"))
    context_frames = int(_scalar(dataset, "context_frames"))
    if schema_values.shape != () or schema_version != VISUAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    if image_size_values.shape != () or image_size != IMAGE_SIZE:
        raise ValueError(f"schema version 1 requires image_size={IMAGE_SIZE}")
    if context_values.shape != () or context_frames != CONTEXT_FRAMES:
        raise ValueError(
            f"schema version 1 requires context_frames={CONTEXT_FRAMES}"
        )

    for name in ("renderer_version", "pillow_version"):
        values = np.asarray(dataset[name])
        if values.shape != () or values.dtype.kind != "U":
            raise ValueError(f"{name} must be a Unicode scalar")
        if not str(values.item()).strip():
            raise ValueError(f"{name} must be a non-empty string")

    frames = _require_dtype(dataset, "frames", np.uint8)
    states = _require_dtype(dataset, "states", np.float64)
    actions = _require_dtype(dataset, "actions", np.float64)
    rewards = _require_dtype(dataset, "rewards", np.float64)
    dones = _require_dtype(dataset, "dones", np.bool_)
    episode_ids = _require_dtype(dataset, "episode_ids", np.int64)
    frame_offsets = _require_dtype(dataset, "frame_offsets", np.int64)
    transition_offsets = _require_dtype(
        dataset,
        "transition_offsets",
        np.int64,
    )
    terminal_reasons = np.asarray(dataset["terminal_reasons"])
    if terminal_reasons.dtype.kind != "U":
        raise ValueError("terminal_reasons must have a Unicode dtype")

    episode_count = int(episode_ids.size)
    transition_count = int(actions.shape[0]) if actions.ndim == 2 else -1
    frame_count = int(frames.shape[0]) if frames.ndim == 4 else -1
    if frames.shape != (frame_count, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("frames must have shape [F, 64, 64, 3]")
    if states.shape != (frame_count, 4):
        raise ValueError("states must have shape [F, 4]")
    if actions.shape != (transition_count, 2):
        raise ValueError("actions must have shape [N, 2]")
    for name, values in (
        ("rewards", rewards),
        ("dones", dones),
        ("terminal_reasons", terminal_reasons),
    ):
        if values.shape != (transition_count,):
            raise ValueError(f"{name} must have shape [N]")
    if episode_ids.shape != (episode_count,):
        raise ValueError("episode_ids must have shape [E]")
    if frame_offsets.shape != (episode_count + 1,):
        raise ValueError("frame_offsets must have shape [E + 1]")
    if transition_offsets.shape != (episode_count + 1,):
        raise ValueError("transition_offsets must have shape [E + 1]")
    if episode_count == 0:
        raise ValueError("visual dataset must contain at least one episode")
    if not np.array_equal(episode_ids, np.unique(episode_ids)):
        raise ValueError("episode_ids must be strictly increasing")

    for name, values in (
        ("states", states),
        ("actions", actions),
        ("rewards", rewards),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")

    if (
        frame_offsets[0] != 0
        or frame_offsets[-1] != frame_count
        or np.any(np.diff(frame_offsets) <= 0)
    ):
        raise ValueError("frame_offsets must cover every frame exactly once")
    if (
        transition_offsets[0] != 0
        or transition_offsets[-1] != transition_count
        or np.any(np.diff(transition_offsets) <= 0)
    ):
        raise ValueError(
            "transition_offsets must cover every transition exactly once"
        )

    frame_lengths = np.diff(frame_offsets)
    transition_lengths = np.diff(transition_offsets)
    for index, episode_id in enumerate(episode_ids):
        if frame_lengths[index] != transition_lengths[index] + 1:
            raise ValueError(
                f"episode {int(episode_id)} must own T plus one frames"
            )
        start = int(transition_offsets[index])
        stop = int(transition_offsets[index + 1])
        if np.any(dones[start : stop - 1]) or not bool(dones[stop - 1]):
            raise ValueError(
                f"episode {int(episode_id)} has invalid terminal placement"
            )
        if (
            np.any(terminal_reasons[start : stop - 1] != "")
            or str(terminal_reasons[stop - 1]) not in VALID_TERMINAL_REASONS
        ):
            raise ValueError(
                f"episode {int(episode_id)} has invalid terminal reasons"
            )

    if frame_count != transition_count + episode_count:
        raise ValueError("visual dataset requires F = N + E")

    expected_scene = scene_from_env(CarEnv())
    scene_arrays = {
        "scene_world_bounds": (
            np.asarray(expected_scene.world_bounds, dtype=np.float64),
            (4,),
        ),
        "scene_obstacle": (
            np.asarray(expected_scene.obstacle, dtype=np.float64),
            (2,),
        ),
        "scene_goal": (
            np.asarray(expected_scene.goal, dtype=np.float64),
            (2,),
        ),
    }
    for name, (expected, shape) in scene_arrays.items():
        values = _require_dtype(dataset, name, np.float64)
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")
        if not np.array_equal(values, expected):
            raise ValueError(
                f"{name} does not match the schema-v1 default scene"
            )

    scene_scalars = {
        "scene_obstacle_radius": expected_scene.obstacle_radius,
        "scene_goal_radius": expected_scene.goal_radius,
        "scene_car_radius": expected_scene.car_radius,
        "scene_dt": expected_scene.dt,
    }
    for name, expected in scene_scalars.items():
        values = _require_dtype(dataset, name, np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a scalar")
        if float(values.item()) != expected:
            raise ValueError(
                f"{name} does not match the schema-v1 default scene"
            )
```

The local variables `schema_values`, `image_size_values`, and `context_values` intentionally enforce exact scalar `int64` storage before their values are interpreted.

- [ ] **Step 8: Implement no-overwrite NPZ save/load with pickle disabled**

Append:

```python
def save_visual_dataset(
    dataset: Mapping[str, np.ndarray],
    output: Path | str,
) -> Path:
    """Validate and exclusively create one compressed visual NPZ."""

    validate_visual_dataset(dataset)
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"visual dataset already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise NotADirectoryError(
            f"visual dataset parent is not a directory: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(values)
                for name, values in dataset.items()
            },
        )
    return path


def load_visual_dataset(path: Path | str) -> VisualDataset:
    """Load and validate one schema-versioned visual artifact."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"visual dataset is not a regular file: {input_path}"
        )
    with np.load(input_path, allow_pickle=False) as loaded:
        missing = set(REQUIRED_VISUAL_ARRAYS) - set(loaded.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"visual dataset is missing arrays: {names}")
        dataset = {
            name: np.asarray(loaded[name])
            for name in loaded.files
        }
    validate_visual_dataset(dataset)
    return dataset
```

- [ ] **Step 9: Run visual dataset tests and existing source-data tests**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_dataset tests.test_collect_data tests.test_dataset -v
```

Expected: all tests PASS. The existing transition schema and state-model preparation remain unchanged.

- [ ] **Step 10: Commit the visual schema**

```bash
git add src/world_model_lab/visual_dataset.py tests/test_visual_dataset.py
git commit -m "feat: build versioned visual episode data"
```

### Task 4: Add deterministic GIF preview and the conversion CLI

**Files:**

- Create: `src/world_model_lab/build_visual_data.py`
- Create: `tests/test_build_visual_data.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: `load_transition_dataset()`, `build_visual_dataset()`, `save_visual_dataset()`, `load_visual_dataset()`, and `summarize_visual_dataset()`.
- Produces: `write_preview_gif()`, `run_visual_data_build()`, `main()`, and the `world-model-build-visual-data` console script.
- Default preview: the longest canonical episode; ties resolve to the lowest episode ID.

- [ ] **Step 1: Write failing entry-point, help, JSON, and path tests**

Create `tests/test_build_visual_data.py`:

```python
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageSequence

from tests.visual_fixtures import make_transition_source
from world_model_lab import build_visual_data
from world_model_lab.build_visual_data import run_visual_data_build
from world_model_lab.visual_dataset import load_visual_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_source(path: Path) -> None:
    np.savez_compressed(path, **make_transition_source())


class BuildVisualDataTest(unittest.TestCase):
    def test_pyproject_registers_visual_data_command(self):
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'world-model-build-visual-data = '
            '"world_model_lab.build_visual_data:main"',
            pyproject,
        )

    def test_cli_help_lists_all_visual_data_parameters(self):
        standard_output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-build-visual-data", "--help"],
        ):
            with redirect_stdout(standard_output):
                with self.assertRaises(SystemExit) as context:
                    build_visual_data.main()

        self.assertEqual(context.exception.code, 0)
        help_text = standard_output.getvalue()
        for flag in (
            "--data",
            "--output",
            "--preview",
            "--preview-episode-id",
        ):
            self.assertIn(flag, help_text)

    def test_cli_prints_sorted_indented_json_without_nan(self):
        standard_output = io.StringIO()
        returned = {"z": 1, "a": 2}
        with patch.object(
            build_visual_data,
            "run_visual_data_build",
            return_value=returned,
        ):
            with patch.object(sys, "argv", ["world-model-build-visual-data"]):
                with redirect_stdout(standard_output):
                    build_visual_data.main()

        self.assertEqual(
            standard_output.getvalue(),
            json.dumps(
                returned,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )

    def test_missing_input_becomes_argument_error(self):
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-build-visual-data",
                    "--data",
                    str(root / "missing.npz"),
                    "--output",
                    str(root / "visual.npz"),
                    "--preview",
                    str(root / "preview.gif"),
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        build_visual_data.main()

        self.assertEqual(context.exception.code, 2)
        self.assertIn("not a regular file", standard_error.getvalue())

    def test_resolved_paths_must_be_pairwise_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)

            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                run_visual_data_build(
                    data_path=source,
                    output_path=root / "nested" / ".." / "source.npz",
                    preview_path=root / "preview.gif",
                )

            same_artifact = root / "artifact.bin"
            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                run_visual_data_build(
                    data_path=source,
                    output_path=same_artifact,
                    preview_path=same_artifact,
                )

    def test_existing_output_or_preview_fails_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)

            existing_output = root / "visual.npz"
            existing_output.write_bytes(b"keep-output")
            preview = root / "preview.gif"
            with self.assertRaises(FileExistsError):
                run_visual_data_build(
                    data_path=source,
                    output_path=existing_output,
                    preview_path=preview,
                )
            self.assertEqual(existing_output.read_bytes(), b"keep-output")
            self.assertFalse(preview.exists())

            existing_output.unlink()
            preview.write_bytes(b"keep-preview")
            with self.assertRaises(FileExistsError):
                run_visual_data_build(
                    data_path=source,
                    output_path=existing_output,
                    preview_path=preview,
                )
            self.assertFalse(existing_output.exists())
            self.assertEqual(preview.read_bytes(), b"keep-preview")

    def test_output_and_preview_directory_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)
            output_directory = root / "visual.npz"
            output_directory.mkdir()
            with self.assertRaisesRegex(IsADirectoryError, "output path"):
                run_visual_data_build(
                    data_path=source,
                    output_path=output_directory,
                    preview_path=root / "preview.gif",
                )

            output_directory.rmdir()
            preview_directory = root / "preview.gif"
            preview_directory.mkdir()
            with self.assertRaisesRegex(IsADirectoryError, "preview path"):
                run_visual_data_build(
                    data_path=source,
                    output_path=root / "visual.npz",
                    preview_path=preview_directory,
                )
```

- [ ] **Step 2: Run CLI contract tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_build_visual_data.BuildVisualDataTest -v
```

Expected: FAIL because `build_visual_data.py` and the console script do not exist.

- [ ] **Step 3: Register the script and implement path, preview, run, and CLI boundaries**

Add this line under `[project.scripts]` in `pyproject.toml`:

```toml
world-model-build-visual-data = "world_model_lab.build_visual_data:main"
```

Create `src/world_model_lab/build_visual_data.py`:

```python
"""Build deterministic episode-oriented RGB data from transition NPZ input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .visual_dataset import (
    build_visual_dataset,
    load_transition_dataset,
    load_visual_dataset,
    save_visual_dataset,
    summarize_visual_dataset,
    validate_visual_dataset,
)


def _resolved_paths(
    *,
    data_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
) -> tuple[Path, Path, Path]:
    source = Path(data_path).expanduser().resolve(strict=False)
    output = Path(output_path).expanduser().resolve(strict=False)
    preview = Path(preview_path).expanduser().resolve(strict=False)
    if len({source, output, preview}) != 3:
        raise ValueError(
            "data, output, and preview paths must be pairwise distinct"
        )
    if not source.is_file():
        raise FileNotFoundError(
            f"transition dataset is not a regular file: {source}"
        )
    for label, path in (("output", output), ("preview", preview)):
        if path.is_dir():
            raise IsADirectoryError(f"{label} path is a directory: {path}")
        if path.exists():
            raise FileExistsError(f"{label} path already exists: {path}")
        if path.parent.exists() and not path.parent.is_dir():
            raise NotADirectoryError(
                f"{label} parent is not a directory: {path.parent}"
            )
    return source, output, preview


def _preview_episode_index(
    dataset: Mapping[str, np.ndarray],
    requested_episode_id: int | None,
) -> int:
    episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    transition_lengths = np.diff(dataset["transition_offsets"])
    if requested_episode_id is None:
        return int(np.argmax(transition_lengths))
    matches = np.flatnonzero(episode_ids == int(requested_episode_id))
    if matches.size != 1:
        raise ValueError(
            f"preview episode is unavailable: {requested_episode_id}"
        )
    return int(matches[0])


def write_preview_gif(
    dataset: Mapping[str, np.ndarray],
    output_path: Path | str,
    *,
    episode_id: int | None = None,
) -> int:
    """Write all frames from one deterministic episode preview."""

    validate_visual_dataset(dataset)
    episode_index = _preview_episode_index(dataset, episode_id)
    selected_episode_id = int(dataset["episode_ids"][episode_index])
    frame_start = int(dataset["frame_offsets"][episode_index])
    frame_stop = int(dataset["frame_offsets"][episode_index + 1])
    frames = np.asarray(dataset["frames"][frame_start:frame_stop])
    duration_ms = max(
        1,
        int(np.rint(float(dataset["scene_dt"].item()) * 1000.0)),
    )

    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"preview path already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise NotADirectoryError(
            f"preview parent is not a directory: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [
        Image.fromarray(frame).convert(
            "P",
            palette=Image.Palette.ADAPTIVE,
            colors=256,
        )
        for frame in frames
    ]
    with path.open("xb") as handle:
        images[0].save(
            handle,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
    return selected_episode_id


def run_visual_data_build(
    *,
    data_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
    preview_episode_id: int | None = None,
) -> dict[str, object]:
    """Validate source, build both artifacts, and return JSON-safe metadata."""

    source, output, preview = _resolved_paths(
        data_path=data_path,
        output_path=output_path,
        preview_path=preview_path,
    )
    transitions = load_transition_dataset(source)
    visual_dataset = build_visual_dataset(transitions)
    selected_index = _preview_episode_index(
        visual_dataset,
        preview_episode_id,
    )
    selected_episode_id = int(
        visual_dataset["episode_ids"][selected_index]
    )

    save_visual_dataset(visual_dataset, output)
    write_preview_gif(
        visual_dataset,
        preview,
        episode_id=selected_episode_id,
    )
    persisted_dataset = load_visual_dataset(output)
    summary: dict[str, object] = summarize_visual_dataset(
        persisted_dataset
    )
    summary.update(
        {
            "source": str(source),
            "output": str(output),
            "preview": str(preview),
            "output_bytes": int(output.stat().st_size),
            "preview_episode_id": selected_episode_id,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/transitions.npz"),
        help="source transition NPZ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/visual_episodes.npz"),
        help="new schema-v1 visual NPZ",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("artifacts/visual_episode_preview.gif"),
        help="new episode preview GIF",
    )
    parser.add_argument(
        "--preview-episode-id",
        type=int,
        help="explicit episode ID; default is the longest episode",
    )
    args = parser.parse_args()

    try:
        summary = run_visual_data_build(
            data_path=args.data,
            output_path=args.output,
            preview_path=args.preview,
            preview_episode_id=args.preview_episode_id,
        )
    except (OSError, ValueError) as error:
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
```

Because `episode_ids` are strictly increasing, `np.argmax()` selects the lowest ID when multiple episodes share the maximum transition count.

- [ ] **Step 4: Run CLI contract tests and confirm GREEN**

Run the command from Step 2.

Expected: 7 tests PASS.

- [ ] **Step 5: Add end-to-end preview, alignment, determinism, and error tests**

Add these methods to `BuildVisualDataTest`:

```python
    def test_tiny_end_to_end_build_writes_valid_npz_and_full_gif(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "visual.npz"
            preview = root / "preview.gif"
            save_source(source)
            source_before = source.read_bytes()

            summary = run_visual_data_build(
                data_path=source,
                output_path=output,
                preview_path=preview,
            )
            loaded = load_visual_dataset(output)
            with Image.open(preview) as image:
                durations = [
                    int(frame.info["duration"])
                    for frame in ImageSequence.Iterator(image)
                ]
                frame_count = int(image.n_frames)

            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["transitions"], 6)
            self.assertEqual(summary["frames"], 8)
            self.assertEqual(summary["four_frame_eligible_episodes"], 1)
            self.assertEqual(summary["one_step_visual_samples"], 1)
            self.assertEqual(summary["preview_episode_id"], 7)
            self.assertEqual(summary["output_bytes"], output.stat().st_size)
            self.assertEqual(frame_count, 5)
            self.assertEqual(durations, [100, 100, 100, 100, 100])
            self.assertEqual(loaded["frames"].shape, (8, 64, 64, 3))
            self.assertEqual(source.read_bytes(), source_before)

    def test_explicit_preview_episode_uses_its_complete_frame_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "visual.npz"
            preview = root / "preview.gif"
            save_source(source)

            summary = run_visual_data_build(
                data_path=source,
                output_path=output,
                preview_path=preview,
                preview_episode_id=3,
            )
            with Image.open(preview) as image:
                frame_count = int(image.n_frames)

            self.assertEqual(summary["preview_episode_id"], 3)
            self.assertEqual(frame_count, 3)

    def test_same_source_and_versions_produce_identical_gif_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)

            first_preview = root / "first.gif"
            second_preview = root / "second.gif"
            run_visual_data_build(
                data_path=source,
                output_path=root / "first.npz",
                preview_path=first_preview,
            )
            run_visual_data_build(
                data_path=source,
                output_path=root / "second.npz",
                preview_path=second_preview,
            )

            self.assertEqual(
                first_preview.read_bytes(),
                second_preview.read_bytes(),
            )

    def test_unavailable_preview_episode_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "visual.npz"
            preview = root / "preview.gif"
            save_source(source)

            with self.assertRaisesRegex(
                ValueError,
                "preview episode is unavailable: 999",
            ):
                run_visual_data_build(
                    data_path=source,
                    output_path=output,
                    preview_path=preview,
                    preview_episode_id=999,
                )

            self.assertFalse(output.exists())
            self.assertFalse(preview.exists())

    def test_invalid_preview_episode_becomes_argument_error(self):
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-build-visual-data",
                    "--data",
                    str(source),
                    "--output",
                    str(root / "visual.npz"),
                    "--preview",
                    str(root / "preview.gif"),
                    "--preview-episode-id",
                    "999",
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        build_visual_data.main()

        self.assertEqual(context.exception.code, 2)
        self.assertIn(
            "preview episode is unavailable: 999",
            standard_error.getvalue(),
        )
```

- [ ] **Step 6: Run focused CLI tests and confirm GREEN**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_build_visual_data -v
```

Expected: all tests PASS; the fixture produces 8 stored frames and a 5-frame default preview for episode 7.

- [ ] **Step 7: Run all affected renderer, schema, and CLI tests**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_visual_observation \
  tests.test_visual_dataset \
  tests.test_build_visual_data -v
```

Expected: all three visual test modules PASS.

- [ ] **Step 8: Commit the conversion workflow**

```bash
git add pyproject.toml src/world_model_lab/build_visual_data.py tests/test_build_visual_data.py
git commit -m "feat: add visual dataset build command"
```

### Task 5: Document and verify the complete visual-observation bridge

**Files:**

- Modify: `README.md`

**Interfaces:**

- Consumes: the completed renderer, versioned artifact, preview writer, and CLI.
- Produces: a reproducible user workflow and acceptance evidence over both the synthetic fixture and the existing real transition dataset.

- [ ] **Step 1: Add the visual episode workflow to the README**

Insert this section after “采集 World Model 训练数据” and before “训练第一个 Learned World Model”:

````markdown
## 生成视觉 Episode 数据

当前状态数据仍然是 ground truth。下面的命令把每条状态轨迹渲染为
`64 × 64 × 3` 的 RGB episode，同时保留实际执行的动作：

```bash
.venv/bin/python -m world_model_lab.build_visual_data \
  --data data/transitions.npz \
  --output data/visual_episodes.npz \
  --preview artifacts/visual_episode_preview.gif
```

重新执行 `.venv/bin/python -m pip install -e .` 后，也可以使用：

```bash
.venv/bin/world-model-build-visual-data
```

单帧只显示边界、障碍物、目标、小车位置和朝向，不显示速度条或运动轨迹。
因此只改变 `velocity` 不会改变当前图像；后续视觉模型需要从连续画面推断运动。

视觉 NPZ 使用 schema version 1：

| 数组 | 形状 | 含义 |
|---|---|---|
| `frames` | `[F, 64, 64, 3]` | 每个 episode 的连续 `uint8` RGB 帧 |
| `states` | `[F, 4]` | 只用于诊断的帧对应物理状态 |
| `actions` | `[N, 2]` | 相邻帧之间实际执行的 steering 和 acceleration |
| `rewards` / `dones` / `terminal_reasons` | `[N]` | transition 元数据 |
| `episode_ids` | `[E]` | 按数值升序排列的 episode ID |
| `frame_offsets` | `[E + 1]` | 每个 episode 的帧切片边界 |
| `transition_offsets` | `[E + 1]` | 每个 episode 的动作切片边界 |

总帧数满足 `F = N + E`。episode 内的因果关系固定为：

```text
frames[k] --actions[k]--> frames[k + 1]
```

schema 同时记录 `renderer_version`、`pillow_version`、默认场景几何、
`image_size=64` 和 `context_frames=4`。短于 4 个 transition 的 episode
仍然保留，但不会被计为可生成“四帧历史 + 下一帧目标”的 episode。

schema version 1 只适用于由未修改的默认 `CarEnv` 和 `collect_transitions`
生成的数据。源 `transitions.npz` 不保存场景或 `dt` provenance，因此转换器
无法检测自定义 `world_bounds`、障碍物/目标、各类半径或 `dt`；
这类数据不得转换为 schema version 1。

未来视觉动力学样本不会只使用“当前帧 + 当前动作”。因为 acceleration
先改变隐藏速度，下一帧的位置仍使用旧速度，所以模型输入需要最近四帧和
对齐的历史动作：

```text
(frames[t-3:t+1], actions[t-3:t]), action[t] -> frame[t+1]
```

这里使用 episode 内部的 Python 半开切片：`frames[t-3:t+1]` 包含四帧，
`actions[t-3:t]` 包含三条历史动作；切片不可跨越 `frame_offsets` /
`transition_offsets` 定义的 episode 边界。

这一步只生成和验证视觉观测数据，不训练 autoencoder、VAE 或 latent
dynamics。`artifacts/visual_episode_preview.gif` 用于人工检查连续运动，
不会进入训练。
````

- [ ] **Step 2: Verify README command, schema, and hidden-state wording**

Run:

```bash
rg -n \
  "world-model-build-visual-data|frames\[k\]|context_frames=4|acceleration|latent" \
  README.md
```

Expected: matches cover the command, causal alignment, four-frame context, hidden acceleration effect, and future latent stage.

- [ ] **Step 3: Run every focused visual test**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_visual_observation \
  tests.test_visual_dataset \
  tests.test_build_visual_data -v
```

Expected: all visual tests PASS.

- [ ] **Step 4: Run the complete repository suite**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v
```

Expected: all existing and new tests PASS with no state-model regressions.

- [ ] **Step 5: Run a real-data smoke conversion in a fresh temporary directory**

Run:

```bash
SMOKE_DIR="$(mktemp -d /tmp/world-model-visual-smoke.XXXXXX)"
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.build_visual_data \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --output "$SMOKE_DIR/visual_episodes.npz" \
  --preview "$SMOKE_DIR/visual_episode_preview.gif"
```

Expected JSON fields:

```json
{
  "context_frames": 4,
  "episodes": 250,
  "four_frame_eligible_episodes": 225,
  "frames": 9018,
  "image_size": 64,
  "one_step_visual_samples": 8053,
  "preview_episode_id": 37,
  "renderer_version": "pillow-raster-v1",
  "schema_version": 1,
  "transitions": 8768
}
```

The same summary also contains absolute `source`, `output`, and `preview` paths plus a positive `output_bytes` count.

- [ ] **Step 6: Inspect the persisted real-data artifact and GIF metadata**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -c \
  "from pathlib import Path; from PIL import Image; from world_model_lab.visual_dataset import load_visual_dataset; root=max(Path('/tmp').glob('world-model-visual-smoke.*'), key=lambda path: path.stat().st_mtime); data=load_visual_dataset(root/'visual_episodes.npz'); image=Image.open(root/'visual_episode_preview.gif'); print({'root': str(root), 'frames': data['frames'].shape, 'actions': data['actions'].shape, 'episodes': data['episode_ids'].shape, 'gif_frames': image.n_frames, 'gif_bytes': (root/'visual_episode_preview.gif').stat().st_size})"
```

Expected: `frames=(9018, 64, 64, 3)`, `actions=(8768, 2)`, `episodes=(250,)`, and positive GIF frame/byte counts. Open the GIF from the printed `SMOKE_DIR` and visually confirm boundaries, obstacle, goal, circular car, heading marker, and continuous motion without a speed overlay.

- [ ] **Step 7: Run repository hygiene checks**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` exits 0. Status lists only the intended README change before the documentation commit; generated `data/` and `artifacts/` files remain ignored.

- [ ] **Step 8: Commit documentation and acceptance evidence**

```bash
git add README.md
git commit -m "docs: explain visual episode artifacts"
```

## Completion Gate

Before declaring the implementation complete, verify all of these statements against command output:

- The renderer is deterministic for the same state, renderer version, and Pillow version.
- A velocity-only state change leaves an individual frame byte-identical.
- Default scene circles retain physical aspect ratio within one pixel.
- Offscreen finite states do not overflow and do not alter the physical state.
- Canonical reconstruction is invariant to source row order.
- Every episode has `T + 1` frames, `T` actions, and exactly one terminal final transition.
- The visual loader rejects unsupported schema versions, invalid offsets, unsafe object arrays, and non-default scene metadata.
- Short episodes remain stored while eligibility and sample counts stay exact.
- The CLI rejects path aliases and existing outputs before writing.
- The GIF and NPZ are independently valid and the source bytes remain unchanged.
- The complete existing test suite passes.
