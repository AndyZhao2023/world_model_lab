# Visual Window Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-independent, index-backed data layer that splits complete visual episodes and lazily returns correctly aligned four-frame, action-conditioned one-step samples.

**Architecture:** Add a focused `visual_windows.py` module next to the existing visual artifact module. It validates schema-v1 data once at construction, stores only episode/time indexes, and copies one sample on demand; the existing `split_episode_ids` remains the sole authority for deterministic 80/10/10 splitting.

**Tech Stack:** Python 3.12, NumPy, standard-library dataclasses and `unittest`, existing `world_model_lab.visual_dataset` and `world_model_lab.dataset` APIs.

## Global Constraints

- Preserve the existing visual schema version 1 and `data/transitions.npz` source contract.
- Keep frames as `uint8` arrays in `[H, W, C]` order; do not normalize, reorder channels, or create tensors.
- Keep actions as `float64`; model-boundary conversion is outside this increment.
- Use exactly four context frames, three history actions, one current action, and one next-frame target.
- Split complete episode IDs before constructing windows; no window may cross an episode boundary.
- Do not expose physical state as a sample input.
- Do not add a second persisted window artifact or any neural-network training behavior.
- Return owned per-sample array copies without mutating caller-owned arrays.
- Use `/Users/andyzhao/Workspace/world_model_lab/.venv/bin/python` with `PYTHONPATH=src` for worktree verification.

---

## File Structure

- Create `src/world_model_lab/visual_windows.py`: index validation, lazy sample access, and whole-episode split factories.
- Create `tests/test_visual_windows.py`: synthetic schema-v1 fixtures and all window/split contract tests.
- Modify `README.md`: document the callable data-layer workflow, sample keys, shapes, dtypes, and model-boundary responsibilities.
- Use existing `src/world_model_lab/dataset.py`: import `split_episode_ids`; do not duplicate its ratio or shuffle logic.
- Use existing `src/world_model_lab/visual_dataset.py`: import `CONTEXT_FRAMES` and `validate_visual_dataset`; do not relax schema validation.

### Task 1: Validated Visual Window Index

**Files:**
- Create: `src/world_model_lab/visual_windows.py`
- Create: `tests/test_visual_windows.py`

**Interfaces:**
- Consumes: `validate_visual_dataset(dataset: Mapping[str, np.ndarray]) -> None`, `CONTEXT_FRAMES == 4`, schema-v1 `episode_ids` and `transition_offsets`.
- Produces: `VisualWindowIndex` and `build_visual_window_index(dataset, selected_episode_ids) -> VisualWindowIndex` for later lazy extraction.

- [ ] **Step 1: Write the synthetic visual fixture and failing index tests**

Create `tests/test_visual_windows.py` with the fixture and tests below:

```python
import unittest

import numpy as np

from tests.visual_fixtures import make_transition_source
from world_model_lab.visual_dataset import (
    build_visual_dataset,
    validate_visual_dataset,
)
from world_model_lab.visual_windows import build_visual_window_index


def make_visual_dataset(
    transition_lengths: tuple[int, ...],
) -> dict[str, np.ndarray]:
    template = build_visual_dataset(make_transition_source())
    episode_count = len(transition_lengths)
    episode_ids = np.arange(10, 10 + episode_count, dtype=np.int64)
    transition_offsets = np.zeros(episode_count + 1, dtype=np.int64)
    frame_offsets = np.zeros(episode_count + 1, dtype=np.int64)
    for index, length in enumerate(transition_lengths):
        if length < 1:
            raise ValueError("fixture episodes require at least one transition")
        transition_offsets[index + 1] = transition_offsets[index] + length
        frame_offsets[index + 1] = frame_offsets[index] + length + 1

    transition_count = int(transition_offsets[-1])
    frame_count = int(frame_offsets[-1])
    frames = np.empty((frame_count, 64, 64, 3), dtype=np.uint8)
    states = np.zeros((frame_count, 4), dtype=np.float64)
    actions = np.empty((transition_count, 2), dtype=np.float64)
    rewards = np.zeros(transition_count, dtype=np.float64)
    dones = np.zeros(transition_count, dtype=np.bool_)
    terminal_reasons = np.full(
        transition_count,
        "",
        dtype=template["terminal_reasons"].dtype,
    )

    for episode_index, episode_id in enumerate(episode_ids):
        frame_start = int(frame_offsets[episode_index])
        frame_stop = int(frame_offsets[episode_index + 1])
        action_start = int(transition_offsets[episode_index])
        action_stop = int(transition_offsets[episode_index + 1])
        for local_frame, global_frame in enumerate(range(frame_start, frame_stop)):
            frames[global_frame].fill(episode_index * 23 + local_frame)
            states[global_frame] = [episode_id, local_frame, 0.0, 0.0]
        for local_step, global_step in enumerate(range(action_start, action_stop)):
            actions[global_step] = [float(episode_id), float(local_step)]
        dones[action_stop - 1] = True
        terminal_reasons[action_stop - 1] = "time_limit"

    dataset = {name: values.copy() for name, values in template.items()}
    dataset.update(
        {
            "frames": frames,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "terminal_reasons": terminal_reasons,
            "episode_ids": episode_ids,
            "frame_offsets": frame_offsets,
            "transition_offsets": transition_offsets,
        }
    )
    validate_visual_dataset(dataset)
    return dataset


class VisualWindowIndexTest(unittest.TestCase):
    def test_index_follows_selected_episode_and_step_order(self):
        dataset = make_visual_dataset((5, 4, 2))

        index = build_visual_window_index(
            dataset,
            np.asarray([11, 10, 12], dtype=np.int64),
        )

        np.testing.assert_array_equal(index.episode_indices, [1, 0, 0])
        np.testing.assert_array_equal(index.step_ids, [3, 3, 4])
        np.testing.assert_array_equal(index.selected_episode_ids, [11, 10, 12])
        np.testing.assert_array_equal(index.eligible_episode_ids, [11, 10])
        np.testing.assert_array_equal(index.skipped_episode_ids, [12])
        self.assertEqual(index.count, 3)

    def test_all_short_episodes_form_a_valid_empty_index(self):
        dataset = make_visual_dataset((1, 3, 2))

        index = build_visual_window_index(dataset, dataset["episode_ids"])

        self.assertEqual(index.count, 0)
        self.assertEqual(index.episode_indices.dtype, np.dtype(np.int64))
        self.assertEqual(index.step_ids.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(index.eligible_episode_ids, [])
        np.testing.assert_array_equal(index.skipped_episode_ids, [10, 11, 12])

    def test_selected_episode_ids_are_strictly_validated(self):
        dataset = make_visual_dataset((4, 4, 4))
        invalid_cases = (
            (np.asarray([], dtype=np.int64), "must be non-empty"),
            (np.asarray([[10]], dtype=np.int64), "one-dimensional"),
            (np.asarray([True]), "integer array"),
            (np.asarray([10.0]), "integer array"),
            (np.asarray([10, 10]), "must not contain duplicates"),
            (np.asarray([99]), "missing from the visual dataset"),
        )
        for selected, message in invalid_cases:
            with self.subTest(selected=selected, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_visual_window_index(dataset, selected)
```

- [ ] **Step 2: Run the index tests to verify the module is missing**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows.VisualWindowIndexTest -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'world_model_lab.visual_windows'`.

- [ ] **Step 3: Implement the immutable index and validated builder**

Create `src/world_model_lab/visual_windows.py` with:

```python
"""Lazy, episode-isolated training windows for visual world models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .visual_dataset import CONTEXT_FRAMES, validate_visual_dataset


@dataclass(frozen=True)
class VisualWindowIndex:
    """Lightweight locations and episode eligibility for visual samples."""

    episode_indices: np.ndarray
    step_ids: np.ndarray
    selected_episode_ids: np.ndarray
    eligible_episode_ids: np.ndarray
    skipped_episode_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "episode_indices",
            "step_ids",
            "selected_episode_ids",
            "eligible_episode_ids",
            "skipped_episode_ids",
        ):
            values = np.asarray(getattr(self, name))
            if values.ndim != 1 or values.dtype != np.dtype(np.int64):
                raise ValueError(f"{name} must be a one-dimensional int64 array")
            owned = values.copy()
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
            arrays[name] = owned
        if arrays["episode_indices"].size != arrays["step_ids"].size:
            raise ValueError("episode_indices and step_ids must have equal lengths")
        if arrays["selected_episode_ids"].size == 0:
            raise ValueError("selected_episode_ids must be non-empty")
        for name in (
            "selected_episode_ids",
            "eligible_episode_ids",
            "skipped_episode_ids",
        ):
            if np.unique(arrays[name]).size != arrays[name].size:
                raise ValueError(f"{name} must not contain duplicates")
        eligible = set(int(value) for value in arrays["eligible_episode_ids"])
        skipped = set(int(value) for value in arrays["skipped_episode_ids"])
        selected = set(int(value) for value in arrays["selected_episode_ids"])
        if eligible & skipped or eligible | skipped != selected:
            raise ValueError("eligible and skipped IDs must partition selected IDs")
        if np.any(arrays["episode_indices"] < 0):
            raise ValueError("episode_indices must be non-negative")
        if np.any(arrays["step_ids"] < CONTEXT_FRAMES - 1):
            raise ValueError("step_ids do not have enough frame history")

    @property
    def count(self) -> int:
        return int(self.step_ids.size)


def _selected_ids(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(selected_episode_ids)
    if raw.ndim != 1:
        raise ValueError("selected_episode_ids must be one-dimensional")
    if raw.size == 0:
        raise ValueError("selected_episode_ids must be non-empty")
    if raw.dtype.kind not in "iu":
        raise ValueError("selected_episode_ids must be an integer array")
    limits = np.iinfo(np.int64)
    integer_values = [int(value) for value in raw.tolist()]
    if min(integer_values) < limits.min or max(integer_values) > limits.max:
        raise ValueError("selected_episode_ids values must fit in int64")
    selected = np.asarray(integer_values, dtype=np.int64)
    if np.unique(selected).size != selected.size:
        raise ValueError("selected_episode_ids must not contain duplicates")
    available = np.asarray(dataset["episode_ids"], dtype=np.int64)
    missing = selected[~np.isin(selected, available)]
    if missing.size:
        raise ValueError(
            f"episode {int(missing[0])} is missing from the visual dataset"
        )
    return selected


def _build_visual_window_index_validated(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowIndex:
    selected = _selected_ids(dataset, selected_episode_ids)
    all_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    transition_offsets = np.asarray(dataset["transition_offsets"], dtype=np.int64)
    positions = {int(episode_id): index for index, episode_id in enumerate(all_ids)}
    episode_indices: list[int] = []
    step_ids: list[int] = []
    eligible_ids: list[int] = []
    skipped_ids: list[int] = []
    first_step = CONTEXT_FRAMES - 1
    for selected_id in selected.tolist():
        episode_index = positions[int(selected_id)]
        transition_count = int(
            transition_offsets[episode_index + 1]
            - transition_offsets[episode_index]
        )
        if transition_count < CONTEXT_FRAMES:
            skipped_ids.append(int(selected_id))
            continue
        eligible_ids.append(int(selected_id))
        for step_id in range(first_step, transition_count):
            episode_indices.append(episode_index)
            step_ids.append(step_id)
    return VisualWindowIndex(
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        selected_episode_ids=selected,
        eligible_episode_ids=np.asarray(eligible_ids, dtype=np.int64),
        skipped_episode_ids=np.asarray(skipped_ids, dtype=np.int64),
    )


def build_visual_window_index(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowIndex:
    """Validate one artifact and index every legal sample in selected episodes."""

    validate_visual_dataset(dataset)
    return _build_visual_window_index_validated(dataset, selected_episode_ids)
```

- [ ] **Step 4: Run the focused index tests**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows.VisualWindowIndexTest -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit the index invariant**

```bash
git add src/world_model_lab/visual_windows.py tests/test_visual_windows.py
git commit -m "feat: index visual training windows"
```

### Task 2: Lazy Sample Extraction

**Files:**
- Modify: `src/world_model_lab/visual_windows.py`
- Modify: `tests/test_visual_windows.py`

**Interfaces:**
- Consumes: `VisualWindowIndex` and `_build_visual_window_index_validated` from Task 1 plus schema-v1 `frames`, `actions`, `frame_offsets`, and `transition_offsets`.
- Produces: `VisualWindowDataset` and `build_visual_window_dataset(dataset, selected_episode_ids) -> VisualWindowDataset`.

- [ ] **Step 1: Add failing sample-alignment, indexing, and copy-isolation tests**

Append these imports and tests to `tests/test_visual_windows.py`:

```python
from world_model_lab.visual_windows import build_visual_window_dataset


class VisualWindowDatasetTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((5, 4, 2))
        self.windows = build_visual_window_dataset(
            self.visual,
            np.asarray([10, 11, 12], dtype=np.int64),
        )

    def test_first_and_last_samples_have_exact_temporal_alignment(self):
        first = self.windows[0]
        np.testing.assert_array_equal(
            first["context_frames"],
            self.visual["frames"][0:4],
        )
        np.testing.assert_array_equal(
            first["history_actions"],
            self.visual["actions"][0:3],
        )
        np.testing.assert_array_equal(first["current_action"], self.visual["actions"][3])
        np.testing.assert_array_equal(first["target_frame"], self.visual["frames"][4])
        self.assertEqual((first["episode_id"], first["step_id"]), (10, 3))

        last_of_first_episode = self.windows[1]
        np.testing.assert_array_equal(
            last_of_first_episode["target_frame"],
            self.visual["frames"][5],
        )
        self.assertEqual(last_of_first_episode["step_id"], 4)

    def test_next_episode_uses_its_own_offsets(self):
        sample = self.windows[2]
        frame_start = int(self.visual["frame_offsets"][1])
        action_start = int(self.visual["transition_offsets"][1])
        np.testing.assert_array_equal(
            sample["context_frames"],
            self.visual["frames"][frame_start : frame_start + 4],
        )
        np.testing.assert_array_equal(
            sample["history_actions"],
            self.visual["actions"][action_start : action_start + 3],
        )
        self.assertEqual((sample["episode_id"], sample["step_id"]), (11, 3))

    def test_sample_contract_and_copy_isolation(self):
        sample = self.windows[0]
        self.assertEqual(
            set(sample),
            {
                "context_frames",
                "history_actions",
                "current_action",
                "target_frame",
                "episode_id",
                "step_id",
            },
        )
        self.assertEqual(sample["context_frames"].shape, (4, 64, 64, 3))
        self.assertEqual(sample["context_frames"].dtype, np.dtype(np.uint8))
        self.assertEqual(sample["history_actions"].shape, (3, 2))
        self.assertEqual(sample["history_actions"].dtype, np.dtype(np.float64))
        self.assertEqual(sample["current_action"].shape, (2,))
        self.assertEqual(sample["target_frame"].shape, (64, 64, 3))
        self.assertIs(type(sample["episode_id"]), int)
        self.assertIs(type(sample["step_id"]), int)

        original_pixel = int(self.visual["frames"][0, 0, 0, 0])
        original_action = float(self.visual["actions"][0, 0])
        sample["context_frames"][0, 0, 0, 0] = 255
        sample["history_actions"][0, 0] = -999.0
        self.assertEqual(int(self.visual["frames"][0, 0, 0, 0]), original_pixel)
        self.assertEqual(float(self.visual["actions"][0, 0]), original_action)

    def test_scalar_index_rules_match_python_sequences(self):
        self.assertEqual(self.windows[-1]["episode_id"], 11)
        self.assertEqual(self.windows[np.int64(0)]["step_id"], 3)
        for invalid in (slice(None), np.asarray([0]), 0.5, True, np.bool_(False)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    self.windows[invalid]
        for invalid in (len(self.windows), -len(self.windows) - 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(IndexError):
                    self.windows[invalid]
```

- [ ] **Step 2: Run the dataset tests to verify the factory is missing**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows.VisualWindowDatasetTest -v
```

Expected: FAIL during import with `ImportError` naming `build_visual_window_dataset`.

- [ ] **Step 3: Implement lazy extraction and strict scalar indexing**

Add `import operator` near the top of `visual_windows.py`, then append:

```python
VisualWindowSample = dict[str, np.ndarray | int]


class VisualWindowDataset:
    """Map-style lazy access to one-step visual dynamics samples."""

    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        index: VisualWindowIndex,
    ) -> None:
        episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
        transition_offsets = np.asarray(dataset["transition_offsets"], dtype=np.int64)
        if np.any(index.episode_indices >= episode_ids.size):
            raise ValueError("window index refers to an unavailable episode")
        for episode_index, step_id in zip(
            index.episode_indices.tolist(),
            index.step_ids.tolist(),
            strict=True,
        ):
            transition_count = int(
                transition_offsets[episode_index + 1]
                - transition_offsets[episode_index]
            )
            if step_id >= transition_count:
                raise ValueError("window index refers to an unavailable step")
        self.index = index
        self._frames = np.asarray(dataset["frames"])
        self._actions = np.asarray(dataset["actions"])
        self._episode_ids = episode_ids
        self._frame_offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
        self._transition_offsets = transition_offsets

    def __len__(self) -> int:
        return self.index.count

    def __getitem__(self, item: int) -> VisualWindowSample:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("visual window index must be an integer")
        try:
            position = operator.index(item)
        except TypeError:
            raise TypeError("visual window index must be an integer") from None
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("visual window index out of range")

        episode_index = int(self.index.episode_indices[position])
        step_id = int(self.index.step_ids[position])
        history_steps = CONTEXT_FRAMES - 1
        frame_start = int(self._frame_offsets[episode_index]) + step_id - history_steps
        action_start = (
            int(self._transition_offsets[episode_index])
            + step_id
            - history_steps
        )
        current_action_index = action_start + history_steps
        return {
            "context_frames": self._frames[
                frame_start : frame_start + CONTEXT_FRAMES
            ].copy(),
            "history_actions": self._actions[
                action_start:current_action_index
            ].copy(),
            "current_action": self._actions[current_action_index].copy(),
            "target_frame": self._frames[
                frame_start + CONTEXT_FRAMES
            ].copy(),
            "episode_id": int(self._episode_ids[episode_index]),
            "step_id": step_id,
        }


def build_visual_window_dataset(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowDataset:
    """Validate and lazily expose samples from selected complete episodes."""

    validate_visual_dataset(dataset)
    index = _build_visual_window_index_validated(dataset, selected_episode_ids)
    return VisualWindowDataset(dataset, index)
```

- [ ] **Step 4: Run all visual-window tests**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit the lazy sample layer**

```bash
git add src/world_model_lab/visual_windows.py tests/test_visual_windows.py
git commit -m "feat: load visual windows lazily"
```

### Task 3: Whole-Episode Split Factory

**Files:**
- Modify: `src/world_model_lab/visual_windows.py`
- Modify: `tests/test_visual_windows.py`

**Interfaces:**
- Consumes: `split_episode_ids(episode_ids, *, seed, ratios) -> dict[str, np.ndarray]`, `_build_visual_window_index_validated`, and `VisualWindowDataset`.
- Produces: `build_visual_window_splits(dataset, *, seed, ratios=(0.8, 0.1, 0.1)) -> dict[str, VisualWindowDataset]`.

- [ ] **Step 1: Add failing split-integrity and determinism tests**

Append this import and test class to `tests/test_visual_windows.py`:

```python
from world_model_lab.visual_windows import build_visual_window_splits


class VisualWindowSplitTest(unittest.TestCase):
    def test_splits_are_episode_disjoint_exhaustive_and_windowed_afterward(self):
        dataset = make_visual_dataset((4,) * 10)

        splits = build_visual_window_splits(dataset, seed=19)

        self.assertEqual(set(splits), {"train", "validation", "test"})
        selected = {
            name: set(window.index.selected_episode_ids.tolist())
            for name, window in splits.items()
        }
        self.assertEqual(
            selected["train"] | selected["validation"] | selected["test"],
            set(dataset["episode_ids"].tolist()),
        )
        self.assertFalse(selected["train"] & selected["validation"])
        self.assertFalse(selected["train"] & selected["test"])
        self.assertFalse(selected["validation"] & selected["test"])
        self.assertEqual(
            {name: len(window) for name, window in splits.items()},
            {"train": 8, "validation": 1, "test": 1},
        )

    def test_fixed_seed_repeats_split_ids_and_window_order(self):
        dataset = make_visual_dataset((5,) * 10)

        first = build_visual_window_splits(dataset, seed=23)
        second = build_visual_window_splits(dataset, seed=23)

        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                first[name].index.selected_episode_ids,
                second[name].index.selected_episode_ids,
            )
            np.testing.assert_array_equal(
                first[name].index.episode_indices,
                second[name].index.episode_indices,
            )
            np.testing.assert_array_equal(
                first[name].index.step_ids,
                second[name].index.step_ids,
            )
```

- [ ] **Step 2: Run split tests to verify the helper is missing**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows.VisualWindowSplitTest -v
```

Expected: FAIL during import with `ImportError` naming `build_visual_window_splits`.

- [ ] **Step 3: Implement split-before-window construction**

Add this import near the top of `visual_windows.py`:

```python
from .dataset import split_episode_ids
```

Append the split factory:

```python
def build_visual_window_splits(
    dataset: Mapping[str, np.ndarray],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, VisualWindowDataset]:
    """Split complete episodes, then lazily expose each split's windows."""

    validate_visual_dataset(dataset)
    split_ids = split_episode_ids(
        np.asarray(dataset["episode_ids"], dtype=np.int64),
        seed=seed,
        ratios=ratios,
    )
    return {
        name: VisualWindowDataset(
            dataset,
            _build_visual_window_index_validated(dataset, split_ids[name]),
        )
        for name in ("train", "validation", "test")
    }
```

- [ ] **Step 4: Run focused and affected visual tests**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows tests.test_dataset tests.test_visual_dataset -v
```

Expected: all focused window, shared split, and visual schema tests PASS.

- [ ] **Step 5: Commit the split factory**

```bash
git add src/world_model_lab/visual_windows.py tests/test_visual_windows.py
git commit -m "feat: split visual windows by episode"
```

### Task 4: Documentation, Full Verification, and Canonical Artifact

**Files:**
- Modify: `README.md`
- Modify: `tests/test_visual_windows.py`
- Generate locally, do not commit: `/Users/andyzhao/Workspace/world_model_lab/data/visual_episodes.npz`
- Generate locally, do not commit: `/Users/andyzhao/Workspace/world_model_lab/artifacts/visual_episode_preview.gif`

**Interfaces:**
- Consumes: `load_visual_dataset`, `build_visual_window_splits`, and the source `/Users/andyzhao/Workspace/world_model_lab/data/transitions.npz`.
- Produces: documented usage plus a validated local artifact with 8,053 total lazy samples.

- [ ] **Step 1: Add a failing README contract test**

Add `from pathlib import Path` to the test imports, define the root, and append:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisualWindowDocumentationTest(unittest.TestCase):
    def test_readme_documents_lazy_split_before_window_usage(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("build_visual_window_splits", readme)
        self.assertIn('sample["context_frames"]', readme)
        self.assertIn('sample["history_actions"]', readme)
        self.assertIn('sample["current_action"]', readme)
        self.assertIn('sample["target_frame"]', readme)
        self.assertIn("先按完整 episode", readme)
        self.assertIn("uint8 NHWC", readme)
```

- [ ] **Step 2: Run the documentation test to verify the README is incomplete**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows.VisualWindowDocumentationTest -v
```

Expected: FAIL because `build_visual_window_splits` is not yet documented.

- [ ] **Step 3: Document the lazy data-layer workflow**

Insert this section after the existing visual sample contract in `README.md`:

````markdown
### 构造视觉训练窗口

训练前先按完整 episode 划分数据，再在每个 split 内构造窗口：

```python
from world_model_lab.visual_dataset import load_visual_dataset
from world_model_lab.visual_windows import build_visual_window_splits

visual = load_visual_dataset("data/visual_episodes.npz")
splits = build_visual_window_splits(visual, seed=42)
train = splits["train"]
sample = train[0]

print(sample["context_frames"].shape)   # (4, 64, 64, 3)
print(sample["history_actions"].shape)  # (3, 2)
print(sample["current_action"].shape)   # (2,)
print(sample["target_frame"].shape)     # (64, 64, 3)
```

窗口层只保存 episode 和时间索引，读取样本时才复制对应图像，因此不会把
四帧历史预先复制数千次。图像保持 `uint8 NHWC`，动作保持 `float64`；除以
255、转换为 `float32` 和变换到 PyTorch 的 `NCHW` 都属于后续模型适配层。

`train.index.selected_episode_ids` 记录实际训练 episode；validation 和 test
拥有互斥的 ID 集合。短于四个 transition 的 episode 会记录在
`skipped_episode_ids` 中，不会进行填充，也不会跨 episode 拼接窗口。
````

- [ ] **Step 4: Run the documentation test and complete suite**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest tests.test_visual_windows.VisualWindowDocumentationTest -v
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m unittest discover -s tests -v
```

Expected: documentation test PASS and the complete suite PASS with no regressions.

- [ ] **Step 5: Generate the canonical local visual artifact**

Run from `/private/tmp/world_model_lab-visual-windows` with permission to write the canonical ignored data and artifact directories:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m world_model_lab.build_visual_data --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz --output /Users/andyzhao/Workspace/world_model_lab/data/visual_episodes.npz --preview /Users/andyzhao/Workspace/world_model_lab/artifacts/visual_episode_preview.gif
```

Expected JSON fields:

```json
{
  "episodes": 250,
  "four_frame_eligible_episodes": 225,
  "frames": 9018,
  "one_step_visual_samples": 8053,
  "schema_version": 1,
  "transitions": 8768
}
```

- [ ] **Step 6: Verify real-data split isolation and total window count**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -c 'from world_model_lab.visual_dataset import load_visual_dataset; from world_model_lab.visual_windows import build_visual_window_splits; d=load_visual_dataset("/Users/andyzhao/Workspace/world_model_lab/data/visual_episodes.npz"); s=build_visual_window_splits(d, seed=42); ids={k:set(v.index.selected_episode_ids.tolist()) for k,v in s.items()}; assert sum(map(len,s.values())) == 8053; assert not ids["train"] & ids["validation"]; assert not ids["train"] & ids["test"]; assert not ids["validation"] & ids["test"]; print({k:{"episodes":len(v.index.selected_episode_ids),"samples":len(v),"skipped":len(v.index.skipped_episode_ids)} for k,v in s.items()})'
```

Expected: the assertion process exits successfully; the three split sample counts sum to `8053`, and the episode counts are `200`, `25`, and `25`.

- [ ] **Step 7: Commit documentation and perform a final clean-tree review**

```bash
git add README.md tests/test_visual_windows.py
git commit -m "docs: explain visual training windows"
git status --short
git diff main...HEAD --check
```

Expected: commit succeeds, `git status --short` prints nothing, and `git diff main...HEAD --check` exits successfully. The generated `data/` and `artifacts/` files remain ignored and are not part of the commit.
