# Ensemble Uncertainty Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine five compatible H10 checkpoints, measure ensemble prediction accuracy and disagreement calibration on held-out one-step and free-rollout data, and write a reproducible diagnostic bundle.

**Architecture:** Add `ensemble.py` as the only checkpoint-compatibility and multi-model inference boundary. Add `diagnose_ensemble.py` for pure calibration/rollout metrics plus artifact orchestration; reuse the existing checkpoint loader, dataset encoders, rollout-window selector, and error functions instead of duplicating their math.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Matplotlib, `unittest`, existing `world_model_lab` package APIs.

## Global Constraints

- Reuse independently trained H10 checkpoints; do not retrain or bootstrap members in this feature.
- Require `training_config["rollout_horizon"] == 10` for every member.
- Require identical split IDs, model configuration, rollout weight, and normalizers across members.
- Aggregate heading with circular math at every one-step and rollout prediction.
- Roll each member forward independently; never feed the ensemble mean back into all members.
- Evaluate only the common held-out test episodes.
- Treat disagreement as descriptive evidence until its relation to error is measured.
- Do not change the MLP architecture, training loss, checkpoint format, MPC, or PPO.
- Add each production behavior only after its focused test has failed for the expected reason.
- Preserve Python 3.10 compatibility; do not use Python 3.11-only library APIs.

---

## File Structure

- Create `src/world_model_lab/ensemble.py`: compatible-member validation, loading, circular aggregation, next-state inference, and independent free rollout.
- Create `src/world_model_lab/diagnose_ensemble.py`: calibration statistics, rollout aggregation, plots, files, orchestration, and CLI.
- Create `tests/test_ensemble.py`: core ensemble validation and inference behavior.
- Create `tests/test_diagnose_ensemble.py`: metric, artifact, CLI, and end-to-end behavior.
- Modify `src/world_model_lab/diagnostics.py`: expose the already-tested normalized state-error calculation through a public wrapper.
- Modify `pyproject.toml`: register `world-model-diagnose-ensemble`.
- Modify `README.md`: document the experiment, artifacts, command, and interpretation limits.

---

### Task 1: Compatible ensemble inference and circular aggregation

**Files:**
- Create: `src/world_model_lab/ensemble.py`
- Create: `tests/test_ensemble.py`

**Interfaces:**
- Consumes: `LoadedWorldModel`, `load_checkpoint`, `diagnostics.predict_next_states`, and the checkpoint `training_config` contract.
- Produces: `WorldModelEnsemble`, `EnsemblePrediction`, `EnsembleRollout`, `build_ensemble`, `load_ensemble`, `predict_ensemble_next_states`, and `rollout_ensemble`.

- [ ] **Step 1: Write deterministic member fixtures and failing aggregation tests**

Create `tests/test_ensemble.py` with a helper that zeroes an actual
`WorldModelMLP` and sets its last-layer bias to a constant normalized delta.
Use real `Normalizer` and `LoadedWorldModel` objects:

```python
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from world_model_lab.dataset import Normalizer
from world_model_lab.ensemble import (
    build_ensemble,
    predict_ensemble_next_states,
    rollout_ensemble,
)
from world_model_lab.model import WorldModelMLP
from world_model_lab.train_world_model import LoadedWorldModel


def make_member(seed: int, delta: np.ndarray) -> LoadedWorldModel:
    model = WorldModelMLP(hidden_size=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(
            torch.as_tensor(delta, dtype=torch.float32)
        )
    return LoadedWorldModel(
        model=model,
        input_normalizer=Normalizer(np.zeros(7), np.ones(7)),
        target_normalizer=Normalizer(np.zeros(4), np.ones(4)),
        split_episode_ids={
            "train": np.asarray([0, 1]),
            "validation": np.asarray([2]),
            "test": np.asarray([3]),
        },
        training_config={
            "seed": seed,
            "split_seed": 0,
            "hidden_size": 4,
            "rollout_horizon": 10,
            "rollout_loss_weight": 1.0,
        },
        train_losses=[1.0],
        validation_losses=[1.0],
        best_epoch=1,
        test_metrics={},
    )


class EnsembleTest(unittest.TestCase):
    def test_prediction_uses_arithmetic_mean_and_rms_disagreement(self):
        ensemble = build_ensemble(
            (
                make_member(1, np.asarray([3.0, 4.0, 0.0, 2.0])),
                make_member(0, np.asarray([1.0, 2.0, 0.0, 0.0])),
            )
        )
        prediction = predict_ensemble_next_states(
            ensemble,
            np.zeros((1, 4)),
            np.zeros((1, 2)),
        )

        self.assertEqual(ensemble.seeds, (0, 1))
        np.testing.assert_allclose(
            prediction.mean_next_states[0],
            [2.0, 3.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(prediction.disagreement["position"], math.sqrt(2.0))
        np.testing.assert_allclose(prediction.disagreement["velocity"], 1.0)

    def test_heading_mean_is_circular_across_wrap_boundary(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([0.0, 0.0, math.radians(179.0), 0.0])),
                make_member(1, np.asarray([0.0, 0.0, math.radians(-179.0), 0.0])),
            )
        )
        prediction = predict_ensemble_next_states(
            ensemble,
            np.zeros((1, 4)),
            np.zeros((1, 2)),
        )

        self.assertAlmostEqual(
            abs(math.degrees(prediction.mean_next_states[0, 2])),
            180.0,
            places=5,
        )
        self.assertAlmostEqual(
            prediction.disagreement["heading_degrees"][0],
            1.0,
            places=5,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_ensemble.EnsembleTest.test_prediction_uses_arithmetic_mean_and_rms_disagreement \
  tests.test_ensemble.EnsembleTest.test_heading_mean_is_circular_across_wrap_boundary -v
```

Expected: import failure because `world_model_lab.ensemble` does not exist.

- [ ] **Step 3: Implement the minimal inference types and circular math**

Create `src/world_model_lab/ensemble.py` with these public types and helpers:

```python
"""Inference and disagreement for compatible world-model ensembles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset import wrap_angle
from .diagnostics import predict_next_states as predict_member_next_states
from .train_world_model import LoadedWorldModel, load_checkpoint


DISAGREEMENT_NAMES = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)


@dataclass(frozen=True)
class WorldModelEnsemble:
    members: tuple[LoadedWorldModel, ...]
    seeds: tuple[int, ...]
    target_std: np.ndarray
    checkpoint_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class EnsemblePrediction:
    member_next_states: np.ndarray
    mean_next_states: np.ndarray
    disagreement: dict[str, np.ndarray]


@dataclass(frozen=True)
class EnsembleRollout:
    member_states: np.ndarray
    mean_states: np.ndarray
    disagreement: dict[str, np.ndarray]


def _circular_mean(values: np.ndarray, *, axis: int) -> np.ndarray:
    return np.arctan2(
        np.mean(np.sin(values), axis=axis),
        np.mean(np.cos(values), axis=axis),
    )


def _aggregate_member_states(
    member_states: np.ndarray,
    *,
    target_std: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    members = np.asarray(member_states, dtype=np.float64)
    if members.ndim != 3 or members.shape[2] != 4:
        raise ValueError("member states must have shape [M, N, 4]")
    if not np.all(np.isfinite(members)):
        raise ValueError("member states must contain only finite values")

    mean = np.mean(members, axis=0)
    mean[:, 2] = _circular_mean(members[:, :, 2], axis=0)
    deviations = members - mean[None, :, :]
    deviations[:, :, 2] = wrap_angle(deviations[:, :, 2])
    position = np.sqrt(np.mean(np.sum(np.square(deviations[:, :, :2]), axis=2), axis=0))
    heading = np.degrees(np.sqrt(np.mean(np.square(deviations[:, :, 2]), axis=0)))
    velocity = np.sqrt(np.mean(np.square(deviations[:, :, 3]), axis=0))
    normalized_total = np.sqrt(
        np.mean(np.square(deviations / target_std[None, None, :]), axis=(0, 2))
    )
    return mean, {
        "position": position,
        "heading_degrees": heading,
        "velocity": velocity,
        "normalized_total": normalized_total,
    }


def predict_ensemble_next_states(
    ensemble: WorldModelEnsemble,
    states: np.ndarray,
    actions: np.ndarray,
) -> EnsemblePrediction:
    member_next_states = np.stack(
        [
            predict_member_next_states(member, states, actions)
            for member in ensemble.members
        ]
    )
    mean, disagreement = _aggregate_member_states(
        member_next_states,
        target_std=ensemble.target_std,
    )
    return EnsemblePrediction(member_next_states, mean, disagreement)
```

In the same file, implement `build_ensemble` validation as described in Step 5
before exposing the constructor to callers.

- [ ] **Step 4: Run the two focused tests and verify GREEN**

Run the Step 2 command again.

Expected: both tests pass.

- [ ] **Step 5: Add failing compatibility and independent-rollout tests**

Extend `EnsembleTest` with table-driven compatibility cases and a recursive
rollout assertion:

```python
    def test_build_rejects_incompatible_or_invalid_members(self):
        valid = make_member(0, np.zeros(4))
        cases = (
            ((valid,), "at least two"),
            ((valid, make_member(0, np.ones(4))), "unique"),
        )
        for members, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_ensemble(members)

        for field, value, message in (
            ("split_seed", 1, "split_seed"),
            ("rollout_horizon", 1, "rollout_horizon"),
            ("rollout_loss_weight", 2.0, "rollout_loss_weight"),
            ("hidden_size", 8, "hidden_size"),
        ):
            incompatible = make_member(1, np.ones(4))
            incompatible.training_config[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    build_ensemble((valid, incompatible))

    def test_rollout_recursively_advances_each_member_independently(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([1.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([3.0, 0.0, 0.0, 0.0])),
            )
        )
        rollout = rollout_ensemble(
            ensemble,
            np.zeros(4),
            np.zeros((2, 2)),
        )

        np.testing.assert_allclose(rollout.member_states[:, :, 0], [[0, 1, 2], [0, 3, 6]])
        np.testing.assert_allclose(rollout.mean_states[:, 0], [0, 2, 4])
        np.testing.assert_allclose(rollout.disagreement["position"], [0, 1, 2])
```

Also add a normalizer mismatch, split-ID mismatch, missing/negative seed, and
non-finite prediction subtest. Add `load_ensemble` tests that save two real
checkpoint payloads through the existing `save_checkpoint` API, supply paths
in reverse seed order, and reject duplicate or missing paths.

- [ ] **Step 6: Run the expanded focused tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_ensemble -v
```

Expected: failures for missing compatibility validation, loading, and rollout.

- [ ] **Step 7: Implement validation, loading, and independent rollout**

Complete `ensemble.py` with:

```python
def _config_value(member: LoadedWorldModel, name: str):
    if name not in member.training_config:
        raise ValueError(f"checkpoint training_config is missing {name}")
    return member.training_config[name]


def build_ensemble(
    members: Iterable[LoadedWorldModel],
    *,
    checkpoint_paths: Iterable[Path | str] = (),
) -> WorldModelEnsemble:
    loaded = list(members)
    paths = tuple(Path(path).resolve() for path in checkpoint_paths)
    if len(loaded) < 2:
        raise ValueError("ensemble requires at least two checkpoints")
    if paths and len(paths) != len(loaded):
        raise ValueError("checkpoint path count must match member count")

    paired = []
    for index, member in enumerate(loaded):
        seed = _config_value(member, "seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
            raise ValueError("checkpoint training seeds must be non-negative integers")
        paired.append((int(seed), member, paths[index] if paths else None))
    paired.sort(key=lambda item: item[0])
    seeds = tuple(item[0] for item in paired)
    if len(set(seeds)) != len(seeds):
        raise ValueError("checkpoint training seeds must be unique")

    reference = paired[0][1]
    if int(_config_value(reference, "rollout_horizon")) != 10:
        raise ValueError("rollout_horizon must equal 10")
    for _, member, _ in paired[1:]:
        for name in ("split_seed", "rollout_horizon", "rollout_loss_weight", "hidden_size"):
            if _config_value(member, name) != _config_value(reference, name):
                raise ValueError(f"checkpoint {name} values differ")
        if int(_config_value(member, "rollout_horizon")) != 10:
            raise ValueError("rollout_horizon must equal 10")
        for split_name in ("train", "validation", "test"):
            if not np.array_equal(
                member.split_episode_ids.get(split_name),
                reference.split_episode_ids.get(split_name),
            ):
                raise ValueError(f"checkpoint {split_name} split episode IDs differ")
        for label, left, right in (
            ("input mean", member.input_normalizer.mean, reference.input_normalizer.mean),
            ("input std", member.input_normalizer.std, reference.input_normalizer.std),
            ("target mean", member.target_normalizer.mean, reference.target_normalizer.mean),
            ("target std", member.target_normalizer.std, reference.target_normalizer.std),
        ):
            if not np.array_equal(left, right):
                raise ValueError(f"checkpoint {label} values differ")

    ordered_members = tuple(item[1] for item in paired)
    ordered_paths = tuple(item[2] for item in paired if item[2] is not None)
    return WorldModelEnsemble(
        members=ordered_members,
        seeds=seeds,
        target_std=reference.target_normalizer.std.copy(),
        checkpoint_paths=ordered_paths,
    )


def load_ensemble(paths: Iterable[Path | str]) -> WorldModelEnsemble:
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("checkpoint paths must be unique")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint is not a regular file: {path}")
    return build_ensemble(
        (load_checkpoint(path) for path in resolved),
        checkpoint_paths=resolved,
    )


def rollout_ensemble(
    ensemble: WorldModelEnsemble,
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> EnsembleRollout:
    initial = np.asarray(initial_state, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    if initial.shape != (4,) or action_array.ndim != 2 or action_array.shape[1] != 2:
        raise ValueError("initial_state must have shape [4] and actions [H, 2]")
    member_states = []
    for member in ensemble.members:
        trajectory = [initial.copy()]
        for action in action_array:
            trajectory.append(
                predict_member_next_states(
                    member,
                    trajectory[-1][None, :],
                    action[None, :],
                )[0]
            )
        member_states.append(np.asarray(trajectory))
    stacked = np.stack(member_states)
    mean, disagreement = _aggregate_member_states(
        stacked,
        target_std=ensemble.target_std,
    )
    return EnsembleRollout(stacked, mean, disagreement)
```

Add explicit finite/shape validation around normalizers, states, actions, and
outputs so invalid fixture cases fail before aggregation.

- [ ] **Step 8: Run focused and affected tests and verify GREEN**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_ensemble tests.test_diagnostics tests.test_train_world_model -v
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/world_model_lab/ensemble.py tests/test_ensemble.py
git commit -m "feat: add world model ensemble inference"
```

---

### Task 2: One-step error and disagreement calibration

**Files:**
- Create: `src/world_model_lab/diagnose_ensemble.py`
- Create: `tests/test_diagnose_ensemble.py`
- Modify: `src/world_model_lab/diagnostics.py`

**Interfaces:**
- Consumes: `WorldModelEnsemble`, `predict_ensemble_next_states`, `compute_state_errors`, `summarize_values`, and target standard deviations.
- Produces: `compute_normalized_squared_errors`, `pearson_correlation`, `build_calibration_bins`, and `evaluate_one_step_calibration`.

- [ ] **Step 1: Write failing tests for the public normalized-error wrapper**

Add to `tests/test_diagnostics.py`:

```python
    def test_public_normalized_squared_errors_matches_component_contract(self):
        result = compute_normalized_squared_errors(
            np.asarray([[2.0, 4.0, 0.2, 3.0]]),
            np.asarray([[1.0, 2.0, 0.1, 1.0]]),
            np.asarray([1.0, 2.0, 0.1, 2.0]),
        )
        np.testing.assert_allclose(
            [result[name][0] for name in ("x", "y", "heading", "velocity", "total")],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        )
```

Import `compute_normalized_squared_errors` from `world_model_lab.diagnostics`.

- [ ] **Step 2: Run the wrapper test and verify RED**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnostics.DiagnosticsTest.test_public_normalized_squared_errors_matches_component_contract -v
```

Expected: import failure because the public function is missing.

- [ ] **Step 3: Expose the existing implementation without duplicating math**

Rename `_compute_normalized_squared_errors` to
`compute_normalized_squared_errors` in `diagnostics.py`, and update its current
internal callers and tests to use the public name. Preserve its existing body
and validation exactly.

- [ ] **Step 4: Run diagnostics tests and verify GREEN**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnostics -v
```

Expected: all diagnostics tests pass.

- [ ] **Step 5: Write failing pure calibration tests**

Create `tests/test_diagnose_ensemble.py` with:

```python
import unittest

import numpy as np

from tests.test_ensemble import make_member
from world_model_lab.diagnose_ensemble import (
    build_calibration_bins,
    evaluate_one_step_calibration,
    pearson_correlation,
)
from world_model_lab.ensemble import build_ensemble


class DiagnoseEnsembleTest(unittest.TestCase):
    def test_pearson_returns_none_for_constant_values(self):
        self.assertIsNone(
            pearson_correlation(np.ones(3), np.asarray([1.0, 2.0, 3.0]))
        )

    def test_calibration_bins_are_equal_count_and_sorted_by_disagreement(self):
        bins = build_calibration_bins(
            np.asarray([4.0, 1.0, 3.0, 2.0]),
            np.asarray([40.0, 10.0, 30.0, 20.0]),
            bin_count=2,
        )
        self.assertEqual([item["count"] for item in bins], [2, 2])
        self.assertEqual([item["disagreement_mean"] for item in bins], [1.5, 3.5])
        self.assertEqual([item["error_mean"] for item in bins], [15.0, 35.0])

    def test_one_step_reports_ensemble_gain_and_calibration(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([0.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([2.0, 0.0, 0.0, 0.0])),
            )
        )
        result = evaluate_one_step_calibration(
            ensemble,
            states=np.zeros((2, 4)),
            actions=np.zeros((2, 2)),
            true_next_states=np.asarray([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]),
            calibration_bins=2,
        )

        position = result["metrics"]["position"]
        self.assertEqual(result["samples"], 2)
        self.assertEqual(position["ensemble_error"]["mean"], 1.0)
        self.assertEqual(position["mean_member_error"]["mean"], 1.5)
        self.assertEqual(position["ensemble_gain_mean"], 0.5)
        self.assertIsNone(position["pearson_correlation"])
```

Add invalid shape, non-finite input, non-positive bin count, and bin-count
larger than sample-count cases.

- [ ] **Step 6: Run calibration tests and verify RED**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble -v
```

Expected: import failure because `diagnose_ensemble.py` is missing.

- [ ] **Step 7: Implement pure one-step metrics**

Create `src/world_model_lab/diagnose_ensemble.py` with:

```python
"""Held-out calibration and rollout diagnostics for H10 ensembles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import (
    compute_normalized_squared_errors,
    compute_state_errors,
    select_rollout_windows,
    summarize_values,
)
from .ensemble import (
    DISAGREEMENT_NAMES,
    WorldModelEnsemble,
    load_ensemble,
    predict_ensemble_next_states,
    rollout_ensemble,
)


METRIC_NAMES = DISAGREEMENT_NAMES


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError("correlation values must be matching non-empty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation values must be finite")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_calibration_bins(
    disagreement: np.ndarray,
    errors: np.ndarray,
    *,
    bin_count: int,
) -> list[dict[str, float | int]]:
    uncertainty = np.asarray(disagreement, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    if bin_count <= 0:
        raise ValueError("calibration_bins must be positive")
    if uncertainty.ndim != 1 or error.shape != uncertainty.shape or uncertainty.size == 0:
        raise ValueError("calibration inputs must be matching non-empty vectors")
    if not np.all(np.isfinite(uncertainty)) or not np.all(np.isfinite(error)):
        raise ValueError("calibration inputs must be finite")
    ordered = np.argsort(uncertainty, kind="stable")
    return [
        {
            "count": int(indices.size),
            "disagreement_mean": float(np.mean(uncertainty[indices])),
            "error_mean": float(np.mean(error[indices])),
        }
        for indices in np.array_split(ordered, min(bin_count, ordered.size))
        if indices.size
    ]
```

Add private `_state_error_arrays(predicted, true, target_std)` returning the
three physical arrays plus `normalized_total` from
`compute_normalized_squared_errors(...)["total"]`. Implement
`evaluate_one_step_calibration` by computing ensemble and per-member error
arrays, summarizing them, and attaching calibration bins:

```python
def evaluate_one_step_calibration(
    ensemble: WorldModelEnsemble,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    true_next_states: np.ndarray,
    calibration_bins: int,
) -> dict[str, Any]:
    prediction = predict_ensemble_next_states(ensemble, states, actions)
    ensemble_errors = _state_error_arrays(
        prediction.mean_next_states,
        true_next_states,
        ensemble.target_std,
    )
    member_errors = {
        name: np.stack(
            [
                _state_error_arrays(member_prediction, true_next_states, ensemble.target_std)[name]
                for member_prediction in prediction.member_next_states
            ]
        )
        for name in METRIC_NAMES
    }
    metrics = {}
    for name in METRIC_NAMES:
        mean_member_per_sample = np.mean(member_errors[name], axis=0)
        bins = build_calibration_bins(
            prediction.disagreement[name],
            ensemble_errors[name],
            bin_count=calibration_bins,
        )
        lowest = float(bins[0]["error_mean"])
        highest = float(bins[-1]["error_mean"])
        metrics[name] = {
            "ensemble_error": summarize_values(ensemble_errors[name]),
            "mean_member_error": summarize_values(mean_member_per_sample),
            "member_error_means": {
                str(seed): float(np.mean(values))
                for seed, values in zip(ensemble.seeds, member_errors[name])
            },
            "ensemble_gain_mean": float(
                np.mean(mean_member_per_sample) - np.mean(ensemble_errors[name])
            ),
            "pearson_correlation": pearson_correlation(
                prediction.disagreement[name], ensemble_errors[name]
            ),
            "lowest_bin_error_mean": lowest,
            "highest_bin_error_mean": highest,
            "highest_to_lowest_risk_ratio": None if lowest == 0.0 else highest / lowest,
            "calibration_bins": bins,
        }
    return {"samples": int(np.asarray(states).shape[0]), "metrics": metrics}
```

- [ ] **Step 8: Run focused and affected tests and verify GREEN**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble tests.test_diagnostics tests.test_ensemble -v
```

Expected: all selected tests pass and no NaN reaches the metric dictionaries.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/world_model_lab/diagnostics.py src/world_model_lab/diagnose_ensemble.py \
  tests/test_diagnostics.py tests/test_diagnose_ensemble.py
git commit -m "feat: measure ensemble uncertainty calibration"
```

---

### Task 3: Free-rollout ensemble metrics and plots

**Files:**
- Modify: `src/world_model_lab/diagnose_ensemble.py`
- Modify: `tests/test_diagnose_ensemble.py`

**Interfaces:**
- Consumes: `WindowSelection.windows`, `rollout_ensemble`, member seeds, and the one-step metric schema.
- Produces: `evaluate_ensemble_rollouts`, `plot_one_step_calibration`, and `plot_rollout_uncertainty`.

- [ ] **Step 1: Write a failing independent-rollout aggregation test**

Use two episodes with two windows each and constant-delta members. Assert the
step-2 ensemble position error differs from a shared-mean feedback rollout and
that aggregate error weights the two episode IDs equally:

```python
    def test_rollout_metrics_weight_episodes_equally_and_keep_member_divergence(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([1.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([3.0, 0.0, 0.0, 0.0])),
            )
        )
        windows = (
            RolloutWindow(10, 0, np.asarray([[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]], dtype=float), np.zeros((2, 2))),
            RolloutWindow(10, 1, np.asarray([[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]], dtype=float), np.zeros((2, 2))),
            RolloutWindow(20, 0, np.asarray([[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]], dtype=float), np.zeros((2, 2))),
        )
        result = evaluate_ensemble_rollouts(
            ensemble,
            windows=windows,
            eligible_episode_ids=np.asarray([10, 20]),
        )

        self.assertEqual(result["steps"], [1, 2])
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["windows"], 3)
        self.assertEqual(
            result["metrics"]["position"]["ensemble_error_mean"],
            [1.0, 2.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["disagreement_mean"],
            [1.0, 2.0],
        )
```

Import `RolloutWindow` from `diagnostics`. Add heading-wrap and constant-vector
correlation cases.

- [ ] **Step 2: Run the rollout metric test and verify RED**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble.DiagnoseEnsembleTest.test_rollout_metrics_weight_episodes_equally_and_keep_member_divergence -v
```

Expected: failure because `evaluate_ensemble_rollouts` is missing.

- [ ] **Step 3: Implement per-window evaluation and equal-episode aggregation**

Add `evaluate_ensemble_rollouts` to `diagnose_ensemble.py`. For every window,
call `rollout_ensemble`, discard initial step 0, and store:

```python
def evaluate_ensemble_rollouts(
    ensemble: WorldModelEnsemble,
    *,
    windows: Iterable[RolloutWindow],
    eligible_episode_ids: np.ndarray,
) -> dict[str, Any]:
```

Use dictionaries keyed by episode ID. For each step and metric:

1. average all ensemble errors from windows belonging to the same episode;
2. average those episode means for `ensemble_error_mean`;
3. perform the same two-level average separately for every member;
4. report mean/min/max across member aggregate means;
5. average disagreement within episode and then across episodes;
6. compute Pearson correlation from the original per-window disagreement and
   ensemble error vectors.

Return this exact shape:

```python
{
    "steps": [1, 2],
    "episodes": 2,
    "windows": 3,
    "aggregation": {
        "accuracy": "window mean within episode, then equal episode mean",
        "correlation": "rollout windows",
    },
    "metrics": {
        "position": {
            "ensemble_error_mean": [0.0, 0.0],
            "mean_member_error_mean": [1.0, 2.0],
            "min_member_error_mean": [1.0, 2.0],
            "max_member_error_mean": [1.0, 2.0],
            "disagreement_mean": [1.0, 2.0],
            "pearson_correlation": [None, None],
        }
    },
}
```

The numeric example above is a schema illustration; calculate values from the
test fixtures rather than hard-coding them.

- [ ] **Step 4: Run rollout tests and verify GREEN**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble -v
```

Expected: rollout and one-step tests pass.

- [ ] **Step 5: Write failing plot tests**

Build minimal metrics fixtures with all four metric names. Patch
`matplotlib.pyplot.subplots` only to inspect axes labels; use real Matplotlib to
verify both functions save non-empty PNGs:

```python
    def test_ensemble_plots_save_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = plot_one_step_calibration(
                make_one_step_metrics(), root / "calibration.png"
            )
            rollout_path = plot_rollout_uncertainty(
                make_rollout_metrics(), root / "rollout.png"
            )
            self.assertGreater(calibration_path.stat().st_size, 0)
            self.assertGreater(rollout_path.stat().st_size, 0)
```

- [ ] **Step 6: Run plot test and verify RED**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble.DiagnoseEnsembleTest.test_ensemble_plots_save_png_files -v
```

Expected: failure because the plot functions are missing.

- [ ] **Step 7: Implement the two four-panel plots**

Implement:

```python
def plot_one_step_calibration(metrics: Mapping[str, Any], output_path: Path | str) -> Path:
    # Four axes: disagreement bin mean on x, observed error mean on y.


def plot_rollout_uncertainty(metrics: Mapping[str, Any], output_path: Path | str) -> Path:
    # Four axes: ensemble and mean-member errors on left y-axis;
    # disagreement on a labelled secondary right y-axis.
```

Use metric labels:

```python
METRIC_LABELS = {
    "position": ("Position", "error (m)", "disagreement (m)"),
    "heading_degrees": ("Heading", "error (degrees)", "disagreement (degrees)"),
    "velocity": ("Velocity", "error (m/s)", "disagreement (m/s)"),
    "normalized_total": ("Normalized total", "normalized MSE", "normalized RMS disagreement"),
}
```

Create output parents, call `tight_layout`, save at 160 DPI, and close figures.
Validate required metric keys before plotting so malformed fixtures raise a
field-specific `ValueError`.

- [ ] **Step 8: Run Task 3 tests and verify GREEN**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble tests.test_ensemble tests.test_diagnostic_plots -v
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add src/world_model_lab/diagnose_ensemble.py tests/test_diagnose_ensemble.py
git commit -m "feat: evaluate ensemble rollout uncertainty"
```

---

### Task 4: Atomic artifact bundle, CLI, and documentation

**Files:**
- Modify: `src/world_model_lab/diagnose_ensemble.py`
- Modify: `tests/test_diagnose_ensemble.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: the pure one-step and rollout evaluators, existing `sha256_file`, NPZ dataset contract, and checkpoint paths.
- Produces: `write_calibration_csv`, `run_ensemble_diagnostics`, `main`, and the `world-model-diagnose-ensemble` command.

- [ ] **Step 1: Write failing CLI and package-registration tests**

Add tests that assert:

```python
    def test_pyproject_registers_ensemble_diagnostic_command(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'world-model-diagnose-ensemble = "world_model_lab.diagnose_ensemble:main"',
            pyproject,
        )

    def test_cli_help_lists_all_protocol_parameters(self):
        for flag in (
            "--data",
            "--checkpoints",
            "--output-dir",
            "--horizons",
            "--windows-per-episode",
            "--calibration-bins",
        ):
            self.assertIn(flag, captured_help_text)
```

Follow the existing `test_multiseed_experiment.py` pattern with `patch.object`
on `sys.argv`, `redirect_stdout`, and `redirect_stderr`. Verify missing data and
checkpoint files exit with code 2 and the CLI prints sorted, indented JSON from
a patched `run_ensemble_diagnostics` return value.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble -v
```

Expected: failures because the script entry and `main` are missing.

- [ ] **Step 3: Register and implement the CLI**

Add to `[project.scripts]` in `pyproject.toml`:

```toml
world-model-diagnose-ensemble = "world_model_lab.diagnose_ensemble:main"
```

Add `main()` with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/diagnostics/h10-ensemble"),
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()
    try:
        result = run_ensemble_diagnostics(
            data_path=args.data,
            checkpoint_paths=args.checkpoints,
            output_dir=args.output_dir,
            horizons=args.horizons,
            windows_per_episode=args.windows_per_episode,
            calibration_bins=args.calibration_bins,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
```

- [ ] **Step 4: Write a failing end-to-end artifact test**

Use `save_checkpoint` with two `make_member` fixtures and a temporary dataset
containing train, validation, and test episodes with contiguous `step_ids`.
Call:

```python
result = run_ensemble_diagnostics(
    data_path=data_path,
    checkpoint_paths=(seed_1_path, seed_0_path),
    output_dir=output_dir,
    horizons=(1, 2),
    windows_per_episode=2,
    calibration_bins=2,
)
```

Assert exactly these files exist and are non-empty:

```python
{
    "manifest.json",
    "metrics.json",
    "one_step_calibration.csv",
    "one_step_calibration.png",
    "rollout_uncertainty.png",
}
```

Parse both JSON files, assert `allow_nan=False` compatibility by re-encoding,
assert checkpoint seeds are sorted `[0, 1]`, hashes match `sha256_file`, only
test transitions were counted, and the returned paths point to the final
directory. Add rejection tests for non-empty output, missing arrays, invalid
horizons, absent test IDs, and too-long horizons.

- [ ] **Step 5: Run end-to-end test and verify RED**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble.DiagnoseEnsembleTest.test_run_writes_complete_atomic_bundle -v
```

Expected: failure because orchestration and artifact writing are missing.

- [ ] **Step 6: Implement CSV, manifest, staging, and orchestration**

Implement `write_calibration_csv` with stable columns:

```python
fieldnames = (
    "metric",
    "bin_index",
    "count",
    "disagreement_mean",
    "error_mean",
)
```

Implement `run_ensemble_diagnostics` exactly as specified:

```python
def run_ensemble_diagnostics(
    *,
    data_path: Path | str,
    checkpoint_paths: Iterable[Path | str],
    output_dir: Path | str,
    horizons: Iterable[int] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    calibration_bins: int = 10,
) -> dict[str, Any]:
```

Execution order:

1. materialize and validate strictly increasing unique positive horizons;
2. validate positive window/bin counts and absent-or-empty output directory;
3. validate dataset path and load `states`, `actions`, `next_states`,
   `episode_ids`, and `step_ids` with `allow_pickle=False`;
4. call `load_ensemble` and require every checkpoint test ID in the dataset;
5. evaluate one-step metrics using only `np.isin(episode_ids, test_ids)`;
6. call `select_rollout_windows(... max_horizon=horizons[-1])`;
7. evaluate free rollouts;
8. build dense step curves, requested-horizon snapshots, and the manifest
   entirely in memory;
9. create a temporary sibling directory with `tempfile.mkdtemp`;
10. write JSON with `allow_nan=False`, CSV, and plots inside staging;
11. write `manifest.json` last inside staging;
12. remove the pre-existing empty output directory with `Path.rmdir()` only;
13. atomically rename staging to the requested output path;
14. remove staging with `shutil.rmtree` if any write or plot fails.

Use `diagnose_model.sha256_file` for the dataset and every sorted checkpoint.
Store resolved paths, member seeds, common H10 configuration, test IDs, and
protocol values in manifest schema version 1. Store one-step and rollout blocks
in metrics schema version 1.

- [ ] **Step 7: Run end-to-end and CLI tests and verify GREEN**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_ensemble -v
```

Expected: all ensemble diagnostic tests pass.

- [ ] **Step 8: Document the experiment and interpretation boundary**

Append `## Ensemble uncertainty diagnostics` to `README.md` containing:

- the five-H10 command from the design specification;
- the five output filenames and their responsibilities;
- definitions of ensemble mean, position/heading/velocity disagreement, and
  normalized total disagreement;
- positive `ensemble_gain_mean` means the ensemble mean beats a typical member;
- positive disagreement/error correlation means uncertainty tends to rank risk;
- correlation near zero means disagreement is not yet useful for MPC;
- different-seed disagreement captures epistemic variation only and is not
  aleatoric environment noise;
- bootstrap members and MPC penalty are explicitly follow-up stages.

- [ ] **Step 9: Run full regression tests**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v
```

Expected: all existing 85 tests plus new tests pass with zero failures.

- [ ] **Step 10: Run static repository checks**

```bash
git diff --check
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests
```

Expected: both commands exit 0 with no output.

- [ ] **Step 11: Commit Task 4**

```bash
git add src/world_model_lab/diagnose_ensemble.py tests/test_diagnose_ensemble.py \
  pyproject.toml README.md
git commit -m "feat: add ensemble diagnostic bundle"
```

---

### Task 5: Real five-member H10 experiment and conclusion

**Files:**
- Generated, ignored: `artifacts/experiments/h1-vs-h10-seeds-0-4/`
- Generated, ignored: `artifacts/diagnostics/h10-ensemble-seeds-0-4/`
- Modify only if the measured conclusion needs documentation correction: `README.md`

**Interfaces:**
- Consumes: `world-model-multiseed`, the five generated H10 checkpoints, and `world-model-diagnose-ensemble`.
- Produces: a real diagnostic bundle and an evidence-based conclusion; no tracked experiment binaries.

- [ ] **Step 1: Verify the real dataset and output preconditions**

Run from the primary project root so the existing ignored data and artifact
locations are available:

```bash
test -f /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz
test ! -e /Users/andyzhao/Workspace/world_model_lab/artifacts/experiments/h1-vs-h10-seeds-0-4
```

Expected: both commands exit 0. If the experiment directory already exists,
validate its manifest and reuse it only when it contains seeds 0-4, split seed
0, 100 epochs, H10 weight 1.0, and the current dataset SHA-256; otherwise choose
a new versioned output directory instead of deleting evidence.

- [ ] **Step 2: Generate the five H10 member checkpoints through the existing protocol**

If a valid existing bundle is unavailable, run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  PYTHONPATH=/private/tmp/world_model_lab-ensemble/src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.multiseed_experiment \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --output-dir /Users/andyzhao/Workspace/world_model_lab/artifacts/experiments/h1-vs-h10-seeds-0-4 \
  --seeds 0 1 2 3 4 \
  --split-seed 0 \
  --epochs 100 \
  --diagnostic-horizons 1 5 10 20 50
```

Expected: command exits 0 and the experiment manifest lists ten paired runs.

- [ ] **Step 3: Run the real ensemble diagnostic**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  PYTHONPATH=/private/tmp/world_model_lab-ensemble/src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_ensemble \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --checkpoints \
    /Users/andyzhao/Workspace/world_model_lab/artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_0/h10/world_model.pt \
    /Users/andyzhao/Workspace/world_model_lab/artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_1/h10/world_model.pt \
    /Users/andyzhao/Workspace/world_model_lab/artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_2/h10/world_model.pt \
    /Users/andyzhao/Workspace/world_model_lab/artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_3/h10/world_model.pt \
    /Users/andyzhao/Workspace/world_model_lab/artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_4/h10/world_model.pt \
  --output-dir /Users/andyzhao/Workspace/world_model_lab/artifacts/diagnostics/h10-ensemble-seeds-0-4 \
  --horizons 1 5 10 20 50 \
  --windows-per-episode 8 \
  --calibration-bins 10
```

Expected: command exits 0 and prints paths to all five ensemble artifacts.

- [ ] **Step 4: Validate generated evidence**

Run:

```bash
PYTHONPATH=/private/tmp/world_model_lab-ensemble/src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -c \
  "import json, pathlib; p=pathlib.Path('/Users/andyzhao/Workspace/world_model_lab/artifacts/diagnostics/h10-ensemble-seeds-0-4'); m=json.loads((p/'metrics.json').read_text()); a=json.loads((p/'manifest.json').read_text()); assert a['member_seeds']==[0,1,2,3,4]; assert m['schema_version']==1; assert m['rollout']['steps']==[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]; print({k:{'gain':v['ensemble_gain_mean'],'corr':v['pearson_correlation'],'risk_ratio':v['highest_to_lowest_risk_ratio']} for k,v in m['one_step']['metrics'].items()})"
```

Expected: assertions pass and the command prints finite gains plus correlation
or `None` for each metric.

- [ ] **Step 5: Interpret without overstating calibration**

Record the conclusion in the task handoff using these rules:

- `ensemble_gain_mean > 0`: ensemble mean improved average error;
- correlation clearly positive and highest/lowest risk ratio above 1:
  disagreement contains useful risk-ranking information;
- correlation near zero or unstable in sign across horizons: disagreement is
  not ready for an MPC penalty;
- strong one-step but weak long-horizon calibration: improve member diversity
  or train with episode bootstrap before MPC;
- negative ensemble gain: inspect member outliers and do not assume averaging
  is beneficial.

- [ ] **Step 6: Final verification and local review**

Use the `superpowers:verification-before-completion` skill, rerun the full test
suite and `git diff --check`, inspect the complete branch diff from `main`, and
confirm no generated `data/` or `artifacts/` files are tracked.

- [ ] **Step 7: Commit any evidence-driven README correction**

Only if Step 5 reveals that README wording contradicts the measured semantics:

```bash
git add README.md
git commit -m "docs: clarify ensemble uncertainty interpretation"
```

Otherwise leave generated evidence ignored and create no empty commit.
