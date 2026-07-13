# Multi-Step World Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional hybrid one-step plus differentiable free-rollout training mode and compare it fairly with the existing one-step world-model baseline.

**Architecture:** Keep sequence validation and construction in `dataset.py`, isolate differentiable PyTorch rollout math in a new `rollout_training.py`, and let `train_world_model.py` orchestrate transition batches, sequence batches, checkpointing, and CLI behavior. Preserve horizon-1 behavior by making sequence inputs optional and keep diagnostics as the unchanged held-out evaluator.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, `unittest`

## Global Constraints

- Keep `WorldModelMLP` architecture unchanged at seven inputs and four normalized delta outputs.
- Keep the CLI default `rollout_horizon=1`; the controlled multi-step experiment explicitly uses horizon 10.
- Use all training transitions for the one-step loss, including transitions from episodes shorter than the rollout horizon.
- Build rollout windows only inside one episode and one train/validation split.
- Use recorded actions during rollout; do not predict actions, rewards, or termination.
- Keep rollout math in PyTorch so gradients cross every predicted state.
- Fit both normalizers from training transitions only.
- Select the best epoch using total validation loss and never use test data for selection.
- Save checkpoint format version 3 and continue loading versions 1 and 2.
- Keep generated datasets, checkpoints, metrics, and plots ignored.
- Treat a negative real-data hypothesis result as valid evidence; do not change seeds or windows to hide it.

## File Structure

- Modify `src/world_model_lab/dataset.py`: sequence-window value and validated deterministic construction.
- Create `src/world_model_lab/rollout_training.py`: differentiable encoding, recursive rollout, wrapped state error, and rollout loss.
- Modify `src/world_model_lab/train_world_model.py`: component histories, hybrid optimizer loop, checkpoint v3, NPZ orchestration, and CLI flags.
- Modify `src/world_model_lab/plot_training.py`: render total and multi-step component histories.
- Modify `tests/test_dataset.py`: sequence construction and validation tests.
- Create `tests/test_rollout_training.py`: recursive prediction, angle wrapping, and gradient-flow tests.
- Modify `tests/test_train_world_model.py`: hybrid training, checkpoint compatibility, and orchestration tests.
- Modify `tests/test_plot_training.py`: component-history plotting regression.
- Modify `README.md`: commands, losses, compatibility, and comparison protocol.

---

### Task 1: Contiguous Sequence Windows

**Files:**
- Modify: `src/world_model_lab/dataset.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Produces: `SequenceWindows(states, actions, next_states, episode_ids, start_step_ids)`.
- Produces: `build_sequence_windows(..., selected_episode_ids: np.ndarray, horizon: int) -> SequenceWindows`.
- Preserves: `build_model_inputs`, `build_model_arrays`, and `split_episode_ids` behavior.

- [ ] **Step 1: Write failing deterministic-window test**

Add the imports and test below to `tests/test_dataset.py`:

```python
from world_model_lab.dataset import (
    SequenceWindows,
    build_sequence_windows,
)

def test_sequence_windows_are_ordered_and_never_cross_episodes(self):
    states = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [11.0, 0.0, 0.0, 0.0],
        ]
    )
    next_states = states + np.asarray([1.0, 0.0, 0.0, 0.0])
    actions = np.arange(10, dtype=np.float64).reshape(5, 2)

    windows = build_sequence_windows(
        states,
        actions,
        next_states,
        episode_ids=np.asarray([0, 0, 0, 1, 1]),
        step_ids=np.asarray([0, 1, 2, 0, 1]),
        selected_episode_ids=np.asarray([0, 1]),
        horizon=2,
    )

    self.assertIsInstance(windows, SequenceWindows)
    self.assertEqual(windows.states.shape, (3, 2, 4))
    self.assertEqual(windows.actions.shape, (3, 2, 2))
    self.assertEqual(windows.next_states.shape, (3, 2, 4))
    np.testing.assert_array_equal(windows.episode_ids, [0, 0, 1])
    np.testing.assert_array_equal(windows.start_step_ids, [0, 1, 0])
    np.testing.assert_array_equal(windows.states[1, :, 0], [1.0, 2.0])
    np.testing.assert_array_equal(windows.states[2, :, 0], [10.0, 11.0])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_dataset.DatasetTest.test_sequence_windows_are_ordered_and_never_cross_episodes -v
```

Expected: FAIL with an import error because `SequenceWindows` and
`build_sequence_windows` do not exist.

- [ ] **Step 3: Implement the sequence value and builder**

Add to `src/world_model_lab/dataset.py`:

```python
@dataclass(frozen=True)
class SequenceWindows:
    """Validated fixed-horizon transition windows."""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    episode_ids: np.ndarray
    start_step_ids: np.ndarray

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def count(self) -> int:
        return int(self.actions.shape[0])


def build_sequence_windows(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    *,
    episode_ids: np.ndarray,
    step_ids: np.ndarray,
    selected_episode_ids: np.ndarray,
    horizon: int,
) -> SequenceWindows:
    """Return every contiguous fixed-horizon window from selected episodes."""

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    step_ids = np.asarray(step_ids)
    selected_episode_ids = np.asarray(selected_episode_ids)
    count = states.shape[0]
    if horizon <= 0:
        raise ValueError("rollout horizon must be positive")
    if states.shape != (count, 4) or next_states.shape != (count, 4):
        raise ValueError("states and next_states must have shape [N, 4]")
    if actions.shape != (count, 2):
        raise ValueError("actions must have shape [N, 2]")
    if episode_ids.shape != (count,) or step_ids.shape != (count,):
        raise ValueError("episode_ids and step_ids must have shape [N]")
    if not all(np.all(np.isfinite(array)) for array in (states, actions, next_states)):
        raise ValueError("sequence arrays must contain only finite values")

    state_windows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    next_state_windows: list[np.ndarray] = []
    window_episode_ids: list[int] = []
    start_step_ids: list[int] = []
    for episode_id_value in selected_episode_ids.tolist():
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
        for start in range(indices.size - horizon + 1):
            stop = start + horizon
            state_windows.append(episode_states[start:stop])
            action_windows.append(actions[indices[start:stop]])
            next_state_windows.append(episode_next_states[start:stop])
            window_episode_ids.append(episode_id)
            start_step_ids.append(start)

    if state_windows:
        window_states = np.stack(state_windows)
        window_actions = np.stack(action_windows)
        window_next_states = np.stack(next_state_windows)
    else:
        window_states = np.empty((0, horizon, 4), dtype=np.float64)
        window_actions = np.empty((0, horizon, 2), dtype=np.float64)
        window_next_states = np.empty((0, horizon, 4), dtype=np.float64)
    return SequenceWindows(
        states=window_states,
        actions=window_actions,
        next_states=window_next_states,
        episode_ids=np.asarray(window_episode_ids, dtype=np.int64),
        start_step_ids=np.asarray(start_step_ids, dtype=np.int64),
    )
```

- [ ] **Step 4: Add failure and short-episode tests**

Add these focused tests:

```python
def test_sequence_windows_reject_non_contiguous_steps(self):
    states = np.zeros((2, 4))
    next_states = states.copy()
    next_states[0] = states[1]
    with self.assertRaisesRegex(ValueError, "step_ids must be contiguous"):
        build_sequence_windows(
            states,
            np.zeros((2, 2)),
            next_states,
            episode_ids=np.asarray([3, 3]),
            step_ids=np.asarray([0, 2]),
            selected_episode_ids=np.asarray([3]),
            horizon=2,
        )

def test_sequence_windows_return_typed_empty_arrays_for_short_episodes(self):
    windows = build_sequence_windows(
        np.zeros((1, 4)),
        np.zeros((1, 2)),
        np.ones((1, 4)),
        episode_ids=np.asarray([4]),
        step_ids=np.asarray([0]),
        selected_episode_ids=np.asarray([4]),
        horizon=2,
    )
    self.assertEqual(windows.count, 0)
    self.assertEqual(windows.states.shape, (0, 2, 4))
    self.assertEqual(windows.actions.shape, (0, 2, 2))
```

- [ ] **Step 5: Run dataset tests and verify GREEN**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_dataset -v
```

Expected: all dataset tests pass.

- [ ] **Step 6: Commit the sequence boundary**

```bash
git add src/world_model_lab/dataset.py tests/test_dataset.py
git commit -m "feat: build contiguous training sequences"
```

---

### Task 2: Differentiable Free Rollout Loss

**Files:**
- Create: `src/world_model_lab/rollout_training.py`
- Create: `tests/test_rollout_training.py`

**Interfaces:**
- Produces: `rollout_states(model, initial_states, actions, input_mean, input_std, target_mean, target_std) -> torch.Tensor` with shape `[B, H, 4]`.
- Produces: `wrapped_state_errors(predicted_states, true_states) -> torch.Tensor`.
- Produces: `rollout_state_loss(...) -> torch.Tensor`.

- [ ] **Step 1: Write the failing recursive-rollout and gradient tests**

Create `tests/test_rollout_training.py`:

```python
import math
import unittest

import torch
from torch import nn

from world_model_lab.rollout_training import (
    rollout_state_loss,
    rollout_states,
    wrapped_state_errors,
)


class ConstantDeltaModel(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = nn.Parameter(torch.as_tensor(delta, dtype=torch.float32))

    def forward(self, inputs):
        return self.delta.expand(inputs.shape[0], -1)


class RolloutTrainingTest(unittest.TestCase):
    def test_rollout_recursively_updates_predicted_states(self):
        model = ConstantDeltaModel([1.0, 0.0, 0.0, 0.5])
        predictions = rollout_states(
            model,
            torch.zeros((1, 4)),
            torch.zeros((1, 3, 2)),
            input_mean=torch.zeros(7),
            input_std=torch.ones(7),
            target_mean=torch.zeros(4),
            target_std=torch.ones(4),
        )
        torch.testing.assert_close(
            predictions[0],
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [2.0, 0.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 1.5],
                ]
            ),
        )

    def test_later_rollout_errors_backpropagate_to_model_parameters(self):
        model = ConstantDeltaModel([0.5, 0.0, 0.0, 0.0])
        true_states = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]
        )
        loss = rollout_state_loss(
            model,
            initial_states=torch.zeros((1, 4)),
            actions=torch.zeros((1, 2, 2)),
            true_next_states=true_states,
            input_mean=torch.zeros(7),
            input_std=torch.ones(7),
            target_mean=torch.zeros(4),
            target_std=torch.ones(4),
        )
        loss.backward()
        self.assertIsNotNone(model.delta.grad)
        self.assertNotEqual(float(model.delta.grad[0]), 0.0)

    def test_state_error_wraps_heading_at_pi_boundary(self):
        predicted = torch.tensor([[[0.0, 0.0, math.radians(179.0), 0.0]]])
        true = torch.tensor([[[0.0, 0.0, math.radians(-179.0), 0.0]]])
        errors = wrapped_state_errors(predicted, true)
        self.assertAlmostEqual(
            abs(math.degrees(float(errors[0, 0, 2]))), 2.0, places=4
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_rollout_training -v
```

Expected: FAIL because `world_model_lab.rollout_training` does not exist.

- [ ] **Step 3: Implement differentiable rollout math**

Create `src/world_model_lab/rollout_training.py`:

```python
"""Differentiable free-rollout operations used during world-model training."""

from __future__ import annotations

import torch
from torch import nn


def _model_inputs(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            states[:, 0],
            states[:, 1],
            torch.sin(states[:, 2]),
            torch.cos(states[:, 2]),
            states[:, 3],
            actions[:, 0],
            actions[:, 1],
        ),
        dim=1,
    )


def _wrap_angle(values: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


def rollout_states(
    model: nn.Module,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    *,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Recursively predict physical states while preserving autograd."""

    if initial_states.ndim != 2 or initial_states.shape[1] != 4:
        raise ValueError("initial_states must have shape [B, 4]")
    if actions.ndim != 3 or actions.shape[0] != initial_states.shape[0] or actions.shape[2] != 2:
        raise ValueError("actions must have shape [B, H, 2]")
    current_states = initial_states
    predicted_states = []
    for offset in range(actions.shape[1]):
        raw_inputs = _model_inputs(current_states, actions[:, offset])
        normalized_inputs = (raw_inputs - input_mean) / input_std
        normalized_deltas = model(normalized_inputs)
        deltas = normalized_deltas * target_std + target_mean
        next_states = current_states + deltas
        next_states = torch.cat(
            (
                next_states[:, :2],
                _wrap_angle(next_states[:, 2:3]),
                next_states[:, 3:4],
            ),
            dim=1,
        )
        predicted_states.append(next_states)
        current_states = next_states
    return torch.stack(predicted_states, dim=1)


def wrapped_state_errors(
    predicted_states: torch.Tensor,
    true_states: torch.Tensor,
) -> torch.Tensor:
    if predicted_states.shape != true_states.shape or predicted_states.shape[-1] != 4:
        raise ValueError("predicted and true states must have matching shape [..., 4]")
    difference = predicted_states - true_states
    return torch.cat(
        (
            difference[..., :2],
            _wrap_angle(difference[..., 2:3]),
            difference[..., 3:4],
        ),
        dim=-1,
    )


def rollout_state_loss(
    model: nn.Module,
    *,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    true_next_states: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    predictions = rollout_states(
        model,
        initial_states,
        actions,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    errors = wrapped_state_errors(predictions, true_next_states)
    return torch.mean(torch.square(errors / target_std))
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_rollout_training -v
```

Expected: all three rollout-training tests pass.

- [ ] **Step 5: Run the existing rollout evaluator tests**

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_evaluate_rollout tests.test_diagnostics -v
```

Expected: existing no-gradient evaluation behavior remains green.

- [ ] **Step 6: Commit differentiable rollout math**

```bash
git add src/world_model_lab/rollout_training.py tests/test_rollout_training.py
git commit -m "feat: add differentiable world model rollout"
```

---

### Task 3: Hybrid Training Loop and Component Histories

**Files:**
- Modify: `src/world_model_lab/train_world_model.py`
- Modify: `tests/test_train_world_model.py`

**Interfaces:**
- Extends: `train_model(..., train_sequences=None, validation_sequences=None, rollout_loss_weight=1.0)`.
- Extends: `TrainingResult` with four component-history lists.
- Preserves: horizon-1 callers that provide only transition arrays.

- [ ] **Step 1: Write failing horizon-1 compatibility assertions**

Extend `test_training_reduces_normalized_loss_on_deterministic_dynamics`:

```python
self.assertEqual(result.train_one_step_losses, result.train_losses)
self.assertEqual(result.validation_one_step_losses, result.validation_losses)
self.assertEqual(result.train_rollout_losses, [0.0] * 100)
self.assertEqual(result.validation_rollout_losses, [0.0] * 100)
```

Run the focused test and verify it fails because the component fields do not
exist:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model.TrainWorldModelTest.test_training_reduces_normalized_loss_on_deterministic_dynamics -v
```

- [ ] **Step 2: Add component histories without changing one-step math**

Update the result dataclasses so old synthetic constructors remain valid:

```python
from dataclasses import dataclass, field

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
```

Initialize all four lists in `train_model`. In the current horizon-1 loop append
the existing epoch and validation losses to the one-step histories and append
`0.0` to both rollout histories. Return all histories in `TrainingResult`.

- [ ] **Step 3: Verify horizon-1 GREEN before adding multi-step behavior**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model -v
```

Expected: all existing training tests and the new component assertions pass.

- [ ] **Step 4: Write failing hybrid-training integration test**

Add a helper and test to `tests/test_train_world_model.py`:

```python
from world_model_lab.dataset import build_model_arrays, build_sequence_windows

def make_sequence_dynamics(episodes=4, steps=4):
    states = []
    actions = []
    next_states = []
    episode_ids = []
    step_ids = []
    for episode_id in range(episodes):
        state = np.asarray([float(episode_id), 0.0, 0.0, 0.2])
        for step in range(steps):
            action = np.asarray([0.05 * episode_id, 0.1])
            delta = np.asarray([0.1 + 0.02 * state[3], 0.0, 0.01 * action[0], 0.02])
            next_state = state + delta
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            episode_ids.append(episode_id)
            step_ids.append(step)
            state = next_state
    return tuple(
        np.asarray(values)
        for values in (states, actions, next_states, episode_ids, step_ids)
    )

def test_multistep_training_records_finite_component_histories(self):
    states, actions, next_states, episode_ids, step_ids = make_sequence_dynamics()
    inputs, targets = build_model_arrays(states, actions, next_states)
    train_mask = episode_ids < 3
    validation_mask = episode_ids == 3
    train_sequences = build_sequence_windows(
        states, actions, next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        selected_episode_ids=np.asarray([0, 1, 2]),
        horizon=3,
    )
    validation_sequences = build_sequence_windows(
        states, actions, next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
        selected_episode_ids=np.asarray([3]),
        horizon=3,
    )

    result = train_model(
        inputs[train_mask],
        targets[train_mask],
        validation_inputs=inputs[validation_mask],
        validation_targets=targets[validation_mask],
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
        rollout_loss_weight=1.0,
        hidden_size=16,
        epochs=4,
        batch_size=4,
        learning_rate=1e-3,
        seed=9,
    )

    self.assertEqual(len(result.train_losses), 4)
    self.assertTrue(np.all(np.isfinite(result.train_one_step_losses)))
    self.assertTrue(np.all(np.isfinite(result.train_rollout_losses)))
    self.assertTrue(np.all(np.asarray(result.train_rollout_losses) > 0.0))
    np.testing.assert_allclose(
        result.train_losses,
        np.asarray(result.train_one_step_losses)
        + np.asarray(result.train_rollout_losses),
        rtol=1e-6,
    )
```

- [ ] **Step 5: Run the hybrid test and verify RED**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model.TrainWorldModelTest.test_multistep_training_records_finite_component_histories -v
```

Expected: FAIL because `train_model` does not accept sequence arguments.

- [ ] **Step 6: Implement hybrid batching and validation**

Import `SequenceWindows` and `rollout_state_loss`. Extend the signature:

```python
def train_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    train_sequences: SequenceWindows | None = None,
    validation_sequences: SequenceWindows | None = None,
    rollout_loss_weight: float = 1.0,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> TrainingResult:
```

Validate paired sequence arguments and finite non-negative weight:

```python
multi_step = train_sequences is not None or validation_sequences is not None
if (train_sequences is None) != (validation_sequences is None):
    raise ValueError("train and validation sequences must be provided together")
if not math.isfinite(rollout_loss_weight) or rollout_loss_weight < 0.0:
    raise ValueError("rollout_loss_weight must be finite and non-negative")
if multi_step:
    if train_sequences.count == 0 or validation_sequences.count == 0:
        raise ValueError("multi-step training requires non-empty sequence windows")
    if train_sequences.horizon != validation_sequences.horizon:
        raise ValueError("train and validation horizons must match")
```

Convert normalizer statistics and sequence arrays once before the epoch loop:

```python
input_mean = _as_float_tensor(input_normalizer.mean)
input_std = _as_float_tensor(input_normalizer.std)
target_mean = _as_float_tensor(target_normalizer.mean)
target_std = _as_float_tensor(target_normalizer.std)
if multi_step:
    train_sequence_states = _as_float_tensor(train_sequences.states)
    train_sequence_actions = _as_float_tensor(train_sequences.actions)
    train_sequence_next_states = _as_float_tensor(train_sequences.next_states)
    validation_sequence_states = _as_float_tensor(validation_sequences.states)
    validation_sequence_actions = _as_float_tensor(validation_sequences.actions)
    validation_sequence_next_states = _as_float_tensor(
        validation_sequences.next_states
    )
```

Inside each epoch, create a seeded sequence permutation. For transition batch
number `batch_index`, select a cycling sequence batch and compute both losses:

```python
sequence_permutation = (
    torch.randperm(train_sequences.count, generator=generator)
    if multi_step
    else None
)
epoch_one_step_loss = 0.0
epoch_rollout_loss = 0.0
for batch_index, start in enumerate(range(0, inputs.shape[0], batch_size)):
    indices = permutation[start : start + batch_size]
    one_step_loss = loss_function(
        model(normalized_inputs[indices]),
        normalized_targets[indices],
    )
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
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    weight = indices.numel()
    epoch_one_step_loss += float(one_step_loss.detach()) * weight
    epoch_rollout_loss += float(rollout_loss.detach()) * weight
```

Append weighted epoch averages and define total histories from their exact
components:

```python
train_one_step = epoch_one_step_loss / inputs.shape[0]
train_rollout = epoch_rollout_loss / inputs.shape[0]
train_one_step_losses.append(train_one_step)
train_rollout_losses.append(train_rollout)
train_losses.append(train_one_step + rollout_loss_weight * train_rollout)
```

For validation, compute one-step MSE over all validation transitions and
rollout MSE over every validation sequence in `batch_size` chunks under
`torch.no_grad()`. Weight chunk losses by sequence count, then select the best
state using:

```python
validation_total = (
    validation_one_step
    + rollout_loss_weight * validation_rollout
)
```

- [ ] **Step 7: Run hybrid and full training tests**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model tests.test_rollout_training -v
```

Expected: all tests pass with finite component histories.

- [ ] **Step 8: Commit the hybrid optimizer loop**

```bash
git add src/world_model_lab/train_world_model.py tests/test_train_world_model.py
git commit -m "feat: train world model through free rollouts"
```

---

### Task 4: Checkpoint Version 3 and Loss Visualization

**Files:**
- Modify: `src/world_model_lab/train_world_model.py`
- Modify: `src/world_model_lab/plot_training.py`
- Modify: `tests/test_train_world_model.py`
- Modify: `tests/test_plot_training.py`

**Interfaces:**
- Saves: checkpoint format version 3 with component histories.
- Loads: formats 1, 2, and 3 into one complete `LoadedWorldModel`.
- Preserves: total train/validation histories and existing plot CLI.

- [ ] **Step 1: Write failing checkpoint-v3 assertions**

Extend `test_checkpoint_round_trip_preserves_predictions_and_metadata`:

```python
self.assertEqual(loaded.train_one_step_losses, result.train_one_step_losses)
self.assertEqual(loaded.train_rollout_losses, result.train_rollout_losses)
self.assertEqual(
    loaded.validation_one_step_losses,
    result.validation_one_step_losses,
)
self.assertEqual(
    loaded.validation_rollout_losses,
    result.validation_rollout_losses,
)
payload = torch.load(path, map_location="cpu", weights_only=True)
self.assertEqual(payload["format_version"], 3)
```

Load `payload` inside the temporary-directory block before the checkpoint file
is removed.

Add a version-2 compatibility test that saves a normal checkpoint, rewrites its
payload without component keys and with `format_version=2`, then asserts:

```python
self.assertEqual(loaded.train_one_step_losses, loaded.train_losses)
self.assertEqual(loaded.train_rollout_losses, [0.0] * len(loaded.train_losses))
self.assertEqual(
    loaded.validation_rollout_losses,
    [0.0] * len(loaded.validation_losses),
)
```

Add a version-1 compatibility test by converting the saved payload and loading
it through the public loader:

```python
legacy_payload = dict(payload)
legacy_payload["format_version"] = 1
legacy_payload["losses"] = legacy_payload.pop("train_losses")
for key in (
    "validation_losses",
    "train_one_step_losses",
    "train_rollout_losses",
    "validation_one_step_losses",
    "validation_rollout_losses",
    "best_epoch",
    "test_metrics",
):
    legacy_payload.pop(key)
legacy_path = directory_path / "world_model_v1.pt"
torch.save(legacy_payload, legacy_path)
loaded_v1 = load_checkpoint(legacy_path)
self.assertEqual(loaded_v1.train_one_step_losses, loaded_v1.train_losses)
self.assertEqual(
    loaded_v1.train_rollout_losses,
    [0.0] * len(loaded_v1.train_losses),
)
self.assertEqual(loaded_v1.validation_losses, [])
```

- [ ] **Step 2: Run checkpoint tests and verify RED**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model.TrainWorldModelTest.test_checkpoint_round_trip_preserves_predictions_and_metadata -v
```

Expected: FAIL because version 3 fields are not saved or loaded.

- [ ] **Step 3: Save and load component histories**

Append the component fields to `LoadedWorldModel` with `field(default_factory=list)`.
In `save_checkpoint`, derive safe histories for old synthetic `TrainingResult`
objects and write version 3:

```python
train_one_step_losses = result.train_one_step_losses or result.train_losses
train_rollout_losses = result.train_rollout_losses or [0.0] * len(result.train_losses)
validation_one_step_losses = (
    result.validation_one_step_losses or result.validation_losses
)
validation_rollout_losses = (
    result.validation_rollout_losses
    or [0.0] * len(result.validation_losses)
)
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
```

Accept `(1, 2, 3)` in `load_checkpoint`. For versions 1 and 2 synthesize
one-step and zero-rollout histories. For version 3 read all four arrays as
floats. Pass them into `LoadedWorldModel`.

- [ ] **Step 4: Verify checkpoint GREEN**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model -v
```

Expected: round-trip and legacy-compatibility tests pass.

- [ ] **Step 5: Write failing multi-step plot test**

In `tests/test_plot_training.py`, construct a `TrainingResult` with explicit
non-zero rollout histories, save it, call `plot_training_history`, and assert
the PNG signature. Patch `matplotlib.axes.Axes.plot` and assert labels include:

```python
{
    "Train total",
    "Validation total",
    "Train one-step",
    "Validation one-step",
    "Train rollout",
    "Validation rollout",
}
```

- [ ] **Step 6: Render total and component curves**

Update `plot_training_history` labels from `Train loss`/`Validation loss` to
`Train total`/`Validation total`. If any rollout history value is non-zero,
add dashed one-step curves and dotted rollout curves for both available splits.
Keep the selected best-epoch marker on total validation loss and keep the log
scale.

- [ ] **Step 7: Run plot and checkpoint tests**

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_plot_training tests.test_train_world_model -v
```

Expected: old and new checkpoint plots are valid PNG files.

- [ ] **Step 8: Commit checkpoint and plot support**

```bash
git add src/world_model_lab/train_world_model.py src/world_model_lab/plot_training.py \
  tests/test_train_world_model.py tests/test_plot_training.py
git commit -m "feat: persist multi-step training histories"
```

---

### Task 5: NPZ Orchestration and CLI Controls

**Files:**
- Modify: `src/world_model_lab/train_world_model.py`
- Modify: `tests/test_train_world_model.py`

**Interfaces:**
- Extends: `run_training(..., rollout_horizon=1, rollout_loss_weight=1.0)`.
- Extends: `world-model-train` with `--rollout-horizon` and `--rollout-loss-weight`.
- Adds: sequence counts and component values to the JSON training summary.

- [ ] **Step 1: Write failing horizon-2 orchestration test**

Create an NPZ from `make_sequence_dynamics(episodes=10, steps=4)` including
`step_ids`. Call:

```python
summary = run_training(
    data_path=dataset_path,
    output_path=checkpoint_path,
    hidden_size=16,
    epochs=2,
    batch_size=8,
    learning_rate=1e-3,
    seed=7,
    rollout_horizon=2,
    rollout_loss_weight=0.5,
)
loaded = load_checkpoint(checkpoint_path)
```

Assert:

```python
self.assertEqual(summary["rollout_horizon"], 2)
self.assertEqual(summary["rollout_loss_weight"], 0.5)
self.assertGreater(summary["train_sequence_windows"], 0)
self.assertGreater(summary["validation_sequence_windows"], 0)
self.assertEqual(loaded.training_config["rollout_horizon"], 2)
self.assertEqual(loaded.training_config["rollout_loss_weight"], 0.5)
self.assertGreater(summary["final_train_rollout_loss"], 0.0)
```

- [ ] **Step 2: Run and verify RED**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model.TrainWorldModelTest.test_run_training_supports_multistep_sequences -v
```

Expected: FAIL because `run_training` does not accept rollout controls.

- [ ] **Step 3: Build split-specific sequences only when enabled**

Extend `run_training`:

```python
def run_training(
    *,
    data_path: Path | str,
    output_path: Path | str,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    rollout_horizon: int = 1,
    rollout_loss_weight: float = 1.0,
) -> dict[str, Any]:
```

Validate both values before loading. Keep the four current required arrays for
horizon 1. For `rollout_horizon > 1`, also require and load `step_ids`, then
build train and validation sequences from the original raw arrays and the
already computed split episode IDs:

```python
train_sequences = validation_sequences = None
if rollout_horizon > 1:
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
```

Pass the sequences and weight to `train_model`. Add `rollout_horizon` and the
effective weight (`0.0` for horizon 1, otherwise the requested value) to
`training_config`.

- [ ] **Step 4: Extend the summary with exact selected-epoch semantics**

Compute `best_index = result.best_epoch - 1` and add:

```python
"rollout_horizon": rollout_horizon,
"rollout_loss_weight": (
    rollout_loss_weight if rollout_horizon > 1 else 0.0
),
"train_sequence_windows": train_sequences.count if train_sequences else 0,
"validation_sequence_windows": (
    validation_sequences.count if validation_sequences else 0
),
"initial_train_one_step_loss": result.train_one_step_losses[0],
"final_train_one_step_loss": result.train_one_step_losses[-1],
"initial_train_rollout_loss": result.train_rollout_losses[0],
"final_train_rollout_loss": result.train_rollout_losses[-1],
"best_validation_one_step_loss": (
    result.validation_one_step_losses[best_index]
),
"best_validation_rollout_loss": (
    result.validation_rollout_losses[best_index]
),
```

- [ ] **Step 5: Add CLI flags and validation regression tests**

Add parser arguments:

```python
parser.add_argument("--rollout-horizon", type=int, default=1)
parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
```

Forward both values to `run_training`. Add tests that horizon 1 still accepts
an NPZ without `step_ids`, while horizon 2 rejects that same NPZ with a message
containing `step_ids`:

```python
summary = run_training(
    data_path=dataset_path_without_step_ids,
    output_path=directory_path / "h1.pt",
    hidden_size=8,
    epochs=1,
    batch_size=16,
    rollout_horizon=1,
)
self.assertEqual(summary["rollout_horizon"], 1)
self.assertEqual(summary["train_sequence_windows"], 0)
with self.assertRaisesRegex(ValueError, "step_ids"):
    run_training(
        data_path=dataset_path_without_step_ids,
        output_path=directory_path / "h2.pt",
        hidden_size=8,
        epochs=1,
        batch_size=16,
        rollout_horizon=2,
    )
```

- [ ] **Step 6: Run training orchestration tests**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model -v
```

Expected: horizon-1 and horizon-2 orchestration tests pass.

- [ ] **Step 7: Commit orchestration and CLI behavior**

```bash
git add src/world_model_lab/train_world_model.py tests/test_train_world_model.py
git commit -m "feat: expose multi-step training controls"
```

---

### Task 6: Documentation, Full Verification, and Controlled Experiment

**Files:**
- Modify: `README.md`
- Verify only: all source and test files
- Generate but do not commit: `artifacts/world_model_h1.pt`
- Generate but do not commit: `artifacts/world_model_h10.pt`
- Generate but do not commit: `artifacts/diagnostics/h1/`
- Generate but do not commit: `artifacts/diagnostics/h10/`

**Interfaces:**
- Documents: backward-compatible baseline command and horizon-10 experiment.
- Produces: two locally reproducible checkpoint/diagnostic bundles for comparison.

- [ ] **Step 1: Document the training modes and loss definitions**

Add a `多步训练实验` section to `README.md` containing these commands:

```bash
# Exact one-step baseline path
.venv/bin/python -m world_model_lab.train_world_model \
  --data data/transitions.npz \
  --output artifacts/world_model_h1.pt \
  --rollout-horizon 1 \
  --epochs 100 \
  --seed 0

# Hybrid one-step + differentiable free-rollout training
.venv/bin/python -m world_model_lab.train_world_model \
  --data data/transitions.npz \
  --output artifacts/world_model_h10.pt \
  --rollout-horizon 10 \
  --rollout-loss-weight 1.0 \
  --epochs 100 \
  --seed 0
```

Explain that horizon 1 does not require `step_ids`, horizon greater than one
does, actions remain recorded, and `train_losses`/`validation_losses` are total
loss while component histories remain separately available.

- [ ] **Step 2: Run the complete test suite**

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v
```

Expected: all existing and new tests pass with zero failures and zero errors.

- [ ] **Step 3: Train the real-data horizon-1 baseline**

Because the worktree does not copy ignored data, reference the standalone
repository inputs explicitly:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_world_model \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --output artifacts/world_model_h1.pt \
  --rollout-horizon 1 \
  --epochs 100 \
  --seed 0
```

Expected: JSON summary with finite total/one-step losses, zero rollout losses,
and checkpoint format version 3.

- [ ] **Step 4: Train the real-data horizon-10 model**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_world_model \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --output artifacts/world_model_h10.pt \
  --rollout-horizon 10 \
  --rollout-loss-weight 1.0 \
  --epochs 100 \
  --seed 0
```

Expected: finite non-zero rollout losses and positive train/validation sequence
counts.

- [ ] **Step 5: Run identical held-out diagnostics**

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_model \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --checkpoint artifacts/world_model_h1.pt \
  --output-dir artifacts/diagnostics/h1 \
  --horizons 1 5 10 20 50 \
  --windows-per-episode 8 --xy-bins 12 --feature-bins 8 --min-bin-count 5

env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_model \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --checkpoint artifacts/world_model_h10.pt \
  --output-dir artifacts/diagnostics/h10 \
  --horizons 1 5 10 20 50 \
  --windows-per-episode 8 --xy-bins 12 --feature-bins 8 --min-bin-count 5
```

Expected: both directories contain `metrics.json`, `manifest.json`,
`overview.png`, and `rollout_errors.png`.

- [ ] **Step 6: Compare the fixed-protocol metrics without changing them**

Run this read-only comparison:

```bash
/Users/andyzhao/Workspace/world_model_lab/.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = {
    "h1": Path("artifacts/diagnostics/h1/metrics.json"),
    "h10": Path("artifacts/diagnostics/h10/metrics.json"),
}
metrics = {name: json.loads(path.read_text()) for name, path in paths.items()}
for name, report in metrics.items():
    overall = report["one_step"]["overall"]
    print(name, "one_step", {
        error: overall[error]["mean"]
        for error in ("position", "heading_degrees", "velocity")
    })
    for horizon in (1, 5, 10, 20, 50):
        result = report["rollout"]["horizons"][str(horizon)]
        print(name, "horizon", horizon, {
            mode: {
                error: result[mode][error]["mean"]
                for error in ("position", "heading_degrees", "velocity")
            }
            for mode in ("teacher_forcing", "free_rollout")
        })
PY
```

Record whether the hypothesis was supported or rejected. Do not rerun with a
different seed based on the result.

- [ ] **Step 7: Verify generated artifacts are ignored and source diff is clean**

```bash
git status --short
git diff --check
```

Expected: only `README.md` is uncommitted; no `artifacts/` files appear.

- [ ] **Step 8: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain multi-step world model training"
```

- [ ] **Step 9: Run final verification after the last commit**

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v
git status --short --branch
```

Expected: the complete suite passes and the feature worktree is clean.
