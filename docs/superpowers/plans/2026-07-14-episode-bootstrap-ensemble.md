# Episode Bootstrap Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train five H10 world-model members from deterministic episode bootstrap samples and compare their held-out uncertainty calibration with the existing seed-only H10 ensemble.

**Architecture:** Add pure episode-sampling utilities, extend the existing training path with optional shared normalizers and bootstrap-expanded training data, then add a focused experiment module that reuses `load_ensemble()` and `run_ensemble_diagnostics()`. Validation/test splits, model structure, H10 loss, and diagnostic definitions remain unchanged so episode sampling is the only intentional experimental variable.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Matplotlib, `unittest`, setuptools console scripts.

## Global Constraints

- Use exactly one existing seed-only H10 checkpoint and one bootstrap H10 checkpoint for each training seed.
- The first real experiment uses training seeds `0 1 2 3 4` and `split_seed=0`.
- Bootstrap samples exactly `N` episode IDs with replacement from the `N` unique training episode IDs.
- Fit input and target normalizers on the complete unbootstrapped training split and reuse them for every member.
- Apply identical episode multiplicities to one-step transitions and H10 sequence windows.
- Never bootstrap validation or test episodes.
- Preserve `WorldModelMLP`, checkpoint format version 3, H10 rollout loss, and current ensemble diagnostic definitions.
- Keep the experiment runner serial; parallel scheduling is outside this change.
- Do not serialize NaN or infinity. A correlation delta is JSON `null` when either input correlation is `null`.
- Follow red-green-refactor for every behavior change.

---

### Task 1: Deterministic Episode Bootstrap Utilities

**Files:**
- Create: `src/world_model_lab/bootstrap.py`
- Create: `tests/test_bootstrap.py`

**Interfaces:**
- Produces: `EpisodeBootstrap`, `sample_episode_bootstrap(episode_ids, *, seed)`, and `expand_episode_transition_indices(dataset_episode_ids, drawn_episode_ids)`.
- Consumers: Task 3 uses the sampler and expansion function inside `run_training()`; Task 6 reads `EpisodeBootstrap.episode_counts` for manifests.

- [ ] **Step 1: Write failing tests for exact draws and transition expansion**

Create `tests/test_bootstrap.py`:

```python
import unittest

import numpy as np

from world_model_lab.bootstrap import (
    expand_episode_transition_indices,
    sample_episode_bootstrap,
)


class EpisodeBootstrapTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_keeps_zero_count_episodes(self):
        result = sample_episode_bootstrap(
            np.asarray([10, 11, 12, 13]),
            seed=3,
        )

        np.testing.assert_array_equal(
            result.drawn_episode_ids,
            [13, 10, 10, 10],
        )
        self.assertEqual(
            result.episode_counts,
            {10: 3, 11: 0, 12: 0, 13: 1},
        )
        self.assertEqual(result.draw_count, 4)
        self.assertEqual(result.unique_count, 2)

        repeated = sample_episode_bootstrap(
            np.asarray([10, 11, 12, 13]),
            seed=3,
        )
        np.testing.assert_array_equal(
            repeated.drawn_episode_ids,
            result.drawn_episode_ids,
        )

    def test_transition_expansion_repeats_complete_episode_groups(self):
        indices = expand_episode_transition_indices(
            np.asarray([10, 11, 10, 12, 13, 10]),
            np.asarray([13, 10, 10]),
        )

        np.testing.assert_array_equal(indices, [4, 0, 2, 5, 0, 2, 5])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_bootstrap -v
```

Expected: import failure because `world_model_lab.bootstrap` does not exist.

- [ ] **Step 3: Implement the bootstrap result and pure helpers**

Create `src/world_model_lab/bootstrap.py`:

```python
"""Deterministic episode-level bootstrap sampling for world-model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeBootstrap:
    drawn_episode_ids: np.ndarray
    episode_counts: dict[int, int]

    @property
    def draw_count(self) -> int:
        return int(self.drawn_episode_ids.size)

    @property
    def unique_count(self) -> int:
        return sum(count > 0 for count in self.episode_counts.values())


def _validate_episode_ids(
    values: np.ndarray,
    *,
    name: str,
    unique: bool,
) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if np.issubdtype(ids.dtype, np.bool_) or not np.issubdtype(
        ids.dtype,
        np.integer,
    ):
        raise ValueError(f"{name} must use a non-boolean integer dtype")
    if np.any(ids < 0):
        raise ValueError(f"{name} must contain only non-negative values")
    normalized = ids.astype(np.int64, copy=True)
    if unique and np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def sample_episode_bootstrap(
    episode_ids: np.ndarray,
    *,
    seed: int,
) -> EpisodeBootstrap:
    source = _validate_episode_ids(
        episode_ids,
        name="episode_ids",
        unique=True,
    )
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("bootstrap seed must be a non-negative integer")
    drawn = np.random.default_rng(int(seed)).choice(
        source,
        size=source.size,
        replace=True,
    ).astype(np.int64, copy=False)
    counts = {
        int(episode_id): int(np.count_nonzero(drawn == episode_id))
        for episode_id in np.sort(source)
    }
    return EpisodeBootstrap(drawn.copy(), counts)


def expand_episode_transition_indices(
    dataset_episode_ids: np.ndarray,
    drawn_episode_ids: np.ndarray,
) -> np.ndarray:
    dataset_ids = _validate_episode_ids(
        dataset_episode_ids,
        name="dataset_episode_ids",
        unique=False,
    )
    drawn_ids = _validate_episode_ids(
        drawn_episode_ids,
        name="drawn_episode_ids",
        unique=False,
    )
    groups = []
    for episode_id in drawn_ids.tolist():
        indices = np.flatnonzero(dataset_ids == episode_id)
        if indices.size == 0:
            raise ValueError(
                f"drawn episode {episode_id} is missing from the dataset"
            )
        groups.append(indices)
    return np.concatenate(groups).astype(np.int64, copy=False)
```

- [ ] **Step 4: Verify GREEN**

Run the command from Step 2.

Expected: 2 tests pass.

- [ ] **Step 5: Add validation tests**

Append to `EpisodeBootstrapTest`:

```python
    def test_sampling_rejects_invalid_source_ids_and_seeds(self):
        invalid_ids = (
            np.asarray([]),
            np.asarray([[1, 2]]),
            np.asarray([1.0, 2.0]),
            np.asarray([True, False]),
            np.asarray([-1, 2]),
            np.asarray([1, 1]),
        )
        for values in invalid_ids:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    sample_episode_bootstrap(values, seed=0)

        for seed in (True, -1, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "bootstrap seed"):
                    sample_episode_bootstrap(np.asarray([1, 2]), seed=seed)

    def test_transition_expansion_rejects_missing_drawn_episode(self):
        with self.assertRaisesRegex(ValueError, "episode 3.*missing"):
            expand_episode_transition_indices(
                np.asarray([1, 1, 2]),
                np.asarray([1, 3]),
            )
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_bootstrap -v
git add src/world_model_lab/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: add episode bootstrap sampling"
```

Expected: 4 tests pass before commit.

---

### Task 2: Explicit Shared Normalizers in `train_model`

**Files:**
- Modify: `src/world_model_lab/train_world_model.py:109-172`
- Modify: `tests/test_train_world_model.py`

**Interfaces:**
- Consumes: existing `Normalizer` and `fit_normalizer()`.
- Produces: optional keyword arguments `input_normalizer` and `target_normalizer` on `train_model()`.
- Consumers: Task 3 passes normalizers fitted on the complete train split while training on bootstrap-expanded arrays.

- [ ] **Step 1: Write failing tests for explicit normalizers and paired validation**

Import `fit_normalizer` from `world_model_lab.dataset`, then add:

```python
    def test_train_model_uses_explicit_shared_normalizers(self):
        inputs, targets = make_linear_dynamics(count=80)
        shared_input = fit_normalizer(inputs[:64])
        shared_target = fit_normalizer(targets[:64])

        result = train_model(
            inputs[:32],
            targets[:32],
            validation_inputs=inputs[64:],
            validation_targets=targets[64:],
            input_normalizer=shared_input,
            target_normalizer=shared_target,
            hidden_size=8,
            epochs=1,
            batch_size=16,
            seed=3,
        )

        np.testing.assert_array_equal(
            result.input_normalizer.mean,
            shared_input.mean,
        )
        np.testing.assert_array_equal(
            result.target_normalizer.std,
            shared_target.std,
        )

    def test_train_model_requires_both_explicit_normalizers(self):
        inputs, targets = make_linear_dynamics(count=64)
        with self.assertRaisesRegex(ValueError, "provided together"):
            train_model(
                inputs[:48],
                targets[:48],
                validation_inputs=inputs[48:],
                validation_targets=targets[48:],
                input_normalizer=fit_normalizer(inputs[:48]),
                hidden_size=8,
                epochs=1,
            )
```

- [ ] **Step 2: Run both tests and verify RED**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_train_world_model.TrainWorldModelTest.test_train_model_uses_explicit_shared_normalizers \
  tests.test_train_world_model.TrainWorldModelTest.test_train_model_requires_both_explicit_normalizers \
  -v
```

Expected: `TypeError` because the new keyword arguments do not exist.

- [ ] **Step 3: Add strict normalizer validation and optional arguments**

Add before `train_model()`:

```python
def _validated_normalizer(
    normalizer: Normalizer,
    *,
    size: int,
    name: str,
) -> Normalizer:
    mean = np.asarray(normalizer.mean, dtype=np.float64)
    std = np.asarray(normalizer.std, dtype=np.float64)
    if (
        mean.shape != (size,)
        or std.shape != (size,)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0.0)
    ):
        raise ValueError(
            f"{name} must have finite mean/std shape [{size}] "
            "and positive std"
        )
    return Normalizer(mean.copy(), std.copy())
```

Extend the keyword-only section of `train_model()` with:

```python
    input_normalizer: Normalizer | None = None,
    target_normalizer: Normalizer | None = None,
```

Replace the unconditional normalizer fitting with:

```python
    if (input_normalizer is None) != (target_normalizer is None):
        raise ValueError("input and target normalizers must be provided together")
    if input_normalizer is None:
        input_normalizer = fit_normalizer(inputs)
        target_normalizer = fit_normalizer(targets)
    else:
        input_normalizer = _validated_normalizer(
            input_normalizer,
            size=WorldModelMLP.input_size,
            name="input_normalizer",
        )
        assert target_normalizer is not None
        target_normalizer = _validated_normalizer(
            target_normalizer,
            size=4,
            name="target_normalizer",
        )
```

- [ ] **Step 4: Verify GREEN and legacy behavior**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_world_model -v
```

Expected: all training tests pass, including existing implicit-normalizer tests.

- [ ] **Step 5: Commit**

```bash
git add src/world_model_lab/train_world_model.py tests/test_train_world_model.py
git commit -m "feat: support shared training normalizers"
```

---

### Task 3: Bootstrap-Aware Training Entry Point and Checkpoint Provenance

**Files:**
- Modify: `src/world_model_lab/train_world_model.py:469-645`
- Modify: `tests/test_train_world_model.py`

**Interfaces:**
- Consumes: Task 1 bootstrap helpers and Task 2 explicit normalizers.
- Produces: `run_training(..., bootstrap_seed: int | None = None)` and CLI flag `--bootstrap-seed`.
- Produces checkpoint config keys only when bootstrap is enabled: `bootstrap_seed`, `bootstrap_episode_draws`, `bootstrap_unique_episodes`, and `bootstrap_episode_counts`.
- Consumers: Task 6 invokes this once per seed.

- [ ] **Step 1: Write a failing integration test for shared normalizers, split invariants, and provenance**

Add to `TrainWorldModelTest`:

```python
    def test_bootstrap_training_preserves_split_and_shared_normalizers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            _save_sequence_dataset(data_path)
            baseline_path = root / "baseline.pt"
            bootstrap_path = root / "bootstrap.pt"

            baseline_summary = run_training(
                data_path=data_path,
                output_path=baseline_path,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                seed=3,
                split_seed=0,
                rollout_horizon=10,
                rollout_loss_weight=1.0,
            )
            bootstrap_summary = run_training(
                data_path=data_path,
                output_path=bootstrap_path,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                seed=3,
                split_seed=0,
                rollout_horizon=10,
                rollout_loss_weight=1.0,
                bootstrap_seed=3,
            )
            baseline = load_checkpoint(baseline_path)
            bootstrap = load_checkpoint(bootstrap_path)

        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                baseline.split_episode_ids[name],
                bootstrap.split_episode_ids[name],
            )
        np.testing.assert_array_equal(
            baseline.input_normalizer.mean,
            bootstrap.input_normalizer.mean,
        )
        np.testing.assert_array_equal(
            baseline.target_normalizer.std,
            bootstrap.target_normalizer.std,
        )
        config = bootstrap.training_config
        self.assertEqual(config["bootstrap_seed"], 3)
        self.assertEqual(
            sum(config["bootstrap_episode_counts"].values()),
            config["bootstrap_episode_draws"],
        )
        self.assertEqual(
            sum(value > 0 for value in config["bootstrap_episode_counts"].values()),
            config["bootstrap_unique_episodes"],
        )
        self.assertNotIn("bootstrap_seed", baseline.training_config)
        self.assertEqual(
            bootstrap_summary["split_transitions"]["validation"],
            baseline_summary["split_transitions"]["validation"],
        )
        self.assertEqual(
            bootstrap_summary["split_transitions"]["test"],
            baseline_summary["split_transitions"]["test"],
        )
        self.assertEqual(
            bootstrap_summary["bootstrap_train_transitions"],
            12 * config["bootstrap_episode_draws"],
        )
        self.assertEqual(
            bootstrap_summary["train_sequence_windows"],
            3 * config["bootstrap_episode_draws"],
        )
```

- [ ] **Step 2: Run the integration test and verify RED**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_train_world_model.TrainWorldModelTest.test_bootstrap_training_preserves_split_and_shared_normalizers \
  -v
```

Expected: `TypeError` because `run_training()` lacks `bootstrap_seed`.

- [ ] **Step 3: Integrate bootstrap after the episode split**

Import:

```python
from .bootstrap import (
    EpisodeBootstrap,
    expand_episode_transition_indices,
    sample_episode_bootstrap,
)
```

Extend `run_training()` with `bootstrap_seed: int | None = None`. Immediately
after masks are built, add:

```python
    full_train_indices = np.flatnonzero(masks["train"])
    shared_input_normalizer = fit_normalizer(inputs[full_train_indices])
    shared_target_normalizer = fit_normalizer(targets[full_train_indices])
    bootstrap: EpisodeBootstrap | None = None
    training_episode_ids = splits["train"]
    training_indices = full_train_indices
    if bootstrap_seed is not None:
        bootstrap = sample_episode_bootstrap(
            splits["train"],
            seed=bootstrap_seed,
        )
        training_episode_ids = bootstrap.drawn_episode_ids
        training_indices = expand_episode_transition_indices(
            episode_ids,
            training_episode_ids,
        )
```

Build training sequences with `selected_episode_ids=training_episode_ids`, but
keep validation sequences on `splits["validation"]`. Pass
`inputs[training_indices]`, `targets[training_indices]`, and the two shared
normalizers to `train_model()`.

- [ ] **Step 4: Persist provenance and return expanded counts**

After creating the existing `training_config`, add:

```python
    if bootstrap is not None:
        training_config.update(
            {
                "bootstrap_seed": int(bootstrap_seed),
                "bootstrap_episode_draws": bootstrap.draw_count,
                "bootstrap_unique_episodes": bootstrap.unique_count,
                "bootstrap_episode_counts": {
                    str(episode_id): count
                    for episode_id, count in bootstrap.episode_counts.items()
                },
            }
        )
```

Add summary fields:

```python
        "bootstrap_seed": (
            int(bootstrap_seed) if bootstrap_seed is not None else None
        ),
        "bootstrap_train_transitions": int(training_indices.size),
```

Keep existing `transitions` and `split_transitions` meanings unchanged.

- [ ] **Step 5: Add CLI and validation tests**

Add `parser.add_argument("--bootstrap-seed", type=int)` and pass it to
`run_training()`. Add a help assertion for the flag. Add a test using a valid
small dataset and `bootstrap_seed=-1`; assert `ValueError` names
`bootstrap seed`. Add a deterministic-repeat test that runs the same training
and bootstrap seeds twice, then asserts identical bootstrap provenance, loss
histories, normalizers, and model state tensors.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_bootstrap tests.test_train_world_model -v
git add src/world_model_lab/train_world_model.py tests/test_train_world_model.py
git commit -m "feat: train from episode bootstrap samples"
```

Expected: all bootstrap and training tests pass.

---

### Task 4: Baseline-vs-Bootstrap Comparison Contract

**Files:**
- Create: `src/world_model_lab/bootstrap_experiment.py`
- Create: `tests/test_bootstrap_experiment.py`

**Interfaces:**
- Consumes: schema-v1 metrics generated by `run_ensemble_diagnostics()`.
- Produces: `build_bootstrap_comparison(baseline_metrics, bootstrap_metrics, *, horizons)`.
- Consumers: Task 5 serializes and plots the comparison; Task 6 calls it after both diagnostics complete.

- [ ] **Step 1: Write failing tests for finite deltas and null propagation**

Create `tests/test_bootstrap_experiment.py`:

```python
import unittest

from world_model_lab.bootstrap_experiment import build_bootstrap_comparison


METRICS = ("position", "heading_degrees", "velocity", "normalized_total")


def make_ensemble_metrics(*, error: float, correlation: float | None):
    one_step = {
        name: {
            "ensemble_error": {"mean": error},
            "pearson_correlation": correlation,
        }
        for name in METRICS
    }
    rollout_metrics = {
        name: {
            "ensemble_error_mean": [error, error + 1.0],
            "disagreement_mean": [0.2, 0.4],
            "pearson_correlation": [correlation, correlation],
        }
        for name in METRICS
    }
    return {
        "schema_version": 1,
        "one_step": {"metrics": one_step},
        "rollout": {
            "steps": [1, 2],
            "metrics": rollout_metrics,
            "horizons": {
                str(horizon): {
                    "metrics": {
                        name: {
                            "ensemble_error_mean": error + horizon - 1,
                            "disagreement_mean": 0.2 * horizon,
                            "pearson_correlation": correlation,
                        }
                        for name in METRICS
                    }
                }
                for horizon in (1, 2)
            },
        },
    }


class BootstrapExperimentTest(unittest.TestCase):
    def test_comparison_uses_bootstrap_minus_baseline_deltas(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=0.1),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )

        one_step = comparison["one_step"]["position"]
        self.assertEqual(one_step["error_delta"], -0.5)
        self.assertAlmostEqual(one_step["correlation_delta"], 0.3)
        rollout = comparison["rollout"]["2"]["position"]
        self.assertEqual(rollout["error_delta"], -0.5)
        self.assertAlmostEqual(rollout["correlation_delta"], 0.3)

    def test_comparison_keeps_null_correlation_delta_json_safe(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=None),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )

        self.assertIsNone(
            comparison["one_step"]["position"]["correlation_delta"]
        )
        self.assertIsNone(
            comparison["rollout"]["1"]["position"]["correlation_delta"]
        )
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_bootstrap_experiment.BootstrapExperimentTest.test_comparison_uses_bootstrap_minus_baseline_deltas \
  tests.test_bootstrap_experiment.BootstrapExperimentTest.test_comparison_keeps_null_correlation_delta_json_safe \
  -v
```

Expected: import failure because `bootstrap_experiment.py` does not exist.

- [ ] **Step 3: Implement strict extraction and comparison**

Create the module with imports, `METRIC_NAMES = DISAGREEMENT_NAMES`, and:

```python
def _finite_float(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _correlation(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name=name)


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline
```

Implement `build_bootstrap_comparison()` so it:

1. Requires schema version 1 for both inputs.
2. Requires strictly increasing positive unique horizons.
3. Reads one-step error from `ensemble_error["mean"]` and correlation from
   `pearson_correlation`.
4. Reads rollout values from
   `metrics["rollout"]["horizons"][str(horizon)]["metrics"][metric]`.
5. Validates every numeric value is finite, except correlations may be `None`.
6. Returns:

```python
{
    "schema_version": 1,
    "delta_definition": "bootstrap minus baseline",
    "horizons": list(horizon_values),
    "one_step": {
        metric: {
            "baseline_error": baseline_error,
            "bootstrap_error": bootstrap_error,
            "error_delta": bootstrap_error - baseline_error,
            "baseline_correlation": baseline_correlation,
            "bootstrap_correlation": bootstrap_correlation,
            "correlation_delta": _delta(
                bootstrap_correlation,
                baseline_correlation,
            ),
        }
        for metric in METRIC_NAMES
    },
    "rollout": {
        str(horizon): {
            metric: {
                "baseline_error": baseline_error,
                "bootstrap_error": bootstrap_error,
                "error_delta": bootstrap_error - baseline_error,
                "baseline_disagreement": baseline_disagreement,
                "bootstrap_disagreement": bootstrap_disagreement,
                "disagreement_delta": (
                    bootstrap_disagreement - baseline_disagreement
                ),
                "baseline_correlation": baseline_correlation,
                "bootstrap_correlation": bootstrap_correlation,
                "correlation_delta": _delta(
                    bootstrap_correlation,
                    baseline_correlation,
                ),
            }
            for metric in METRIC_NAMES
        }
        for horizon in horizon_values
    },
}
```

Use small private extraction helpers rather than duplicating dictionary access.

- [ ] **Step 4: Add malformed-input tests and verify GREEN**

Add subtests for a missing metric, non-finite error, duplicate horizon, and
missing horizon snapshot. Each must raise `ValueError` naming the field. Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_bootstrap_experiment -v
```

Expected: all comparison tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/world_model_lab/bootstrap_experiment.py tests/test_bootstrap_experiment.py
git commit -m "feat: compare bootstrap ensemble metrics"
```

---

### Task 5: Stable Comparison CSV and Plot

**Files:**
- Modify: `src/world_model_lab/bootstrap_experiment.py`
- Modify: `tests/test_bootstrap_experiment.py`

**Interfaces:**
- Consumes: Task 4 comparison dictionary.
- Produces: `write_comparison_csv(comparison, output_path)` and `plot_bootstrap_comparison(comparison, output_path)`.
- Consumers: Task 6 writes both artifacts after diagnostics succeed.

- [ ] **Step 1: Write failing CSV and PNG tests**

Add imports for `csv`, `tempfile`, and `Path`, and import the two new functions.
Add:

```python
    def test_comparison_csv_has_stable_one_step_and_rollout_rows(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=0.1),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_comparison_csv(
                comparison,
                Path(directory) / "comparison.csv",
            )
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]["evaluation"], "one_step")
        self.assertEqual(rows[0]["horizon"], "")
        self.assertEqual(rows[4]["evaluation"], "rollout")
        self.assertEqual(rows[4]["horizon"], "1")
        self.assertEqual(rows[-1]["metric"], "normalized_total")

    def test_comparison_plot_is_a_nonempty_png_with_null_correlations(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=None),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = plot_bootstrap_comparison(
                comparison,
                Path(directory) / "comparison.png",
            )
            contents = path.read_bytes()

        self.assertEqual(contents[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(contents), 1000)
```

- [ ] **Step 2: Run tests and verify RED**

Run the two new tests. Expected: import failure for the missing functions.

- [ ] **Step 3: Implement stable CSV serialization**

Use this exact field order:

```python
COMPARISON_FIELDS = (
    "evaluation",
    "horizon",
    "metric",
    "baseline_error",
    "bootstrap_error",
    "error_delta",
    "baseline_disagreement",
    "bootstrap_disagreement",
    "disagreement_delta",
    "baseline_correlation",
    "bootstrap_correlation",
    "correlation_delta",
)
```

`write_comparison_csv()` writes one-step rows first in `METRIC_NAMES` order,
then rollout rows in horizon-major and metric-minor order. One-step disagreement
cells and horizon are empty strings. Use `csv.DictWriter`, create the parent
directory, and return the output path.

- [ ] **Step 4: Implement the four-panel comparison plot**

`plot_bootstrap_comparison()` creates `plt.subplots(2, 2, figsize=(11, 8))`.
Each metric panel plots baseline/bootstrap rollout error against horizon on the
primary axis and baseline/bootstrap correlation on a `twinx()` axis. Convert
`None` correlations to `np.nan` so Matplotlib renders gaps. Label every panel,
both y axes, and bottom x axes; save at `dpi=160`; close the figure; return the
output path.

- [ ] **Step 5: Verify focused tests and commit**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_bootstrap_experiment -v
git add src/world_model_lab/bootstrap_experiment.py tests/test_bootstrap_experiment.py
git commit -m "feat: render bootstrap ensemble comparison"
```

---

### Task 6: End-to-End Bootstrap Experiment Runner and CLI

**Files:**
- Modify: `src/world_model_lab/bootstrap_experiment.py`
- Modify: `tests/test_bootstrap_experiment.py`
- Modify: `pyproject.toml:17-25`

**Interfaces:**
- Consumes: Task 3 `run_training()`, Task 4 comparison builder, Task 5 writers, existing `load_ensemble()`, `run_ensemble_diagnostics()`, and `sha256_file()`.
- Produces: `run_bootstrap_experiment(...) -> dict[str, Any]` and console command `world-model-bootstrap-ensemble`.

- [ ] **Step 1: Write failing CLI-registration and help tests**

Assert `pyproject.toml` contains exactly:

```toml
world-model-bootstrap-ensemble = "world_model_lab.bootstrap_experiment:main"
```

Patch `sys.argv` with `--help`, call `main()`, and assert help contains:

```text
--data
--baseline-checkpoints
--output-dir
--seeds
--split-seed
--hidden-size
--epochs
--batch-size
--learning-rate
--rollout-loss-weight
--diagnostic-horizons
--windows-per-episode
--calibration-bins
```

Run the tests and confirm RED because the script and `main()` are absent.

- [ ] **Step 2: Write the failing two-member end-to-end test**

Add a local `save_sequence_dynamics()` helper with 10 episodes and 12 steps,
matching `tests/test_multiseed_experiment.py`. In a temporary directory, train
seed-only baseline H10 checkpoints for seeds `0` and `1`, one epoch, hidden size
8, batch size 32, and `split_seed=0`, then call:

```python
result = run_bootstrap_experiment(
    data_path=data_path,
    baseline_checkpoint_paths=(baseline_seed_1, baseline_seed_0),
    output_dir=output_dir,
    seeds=(1, 0),
    split_seed=0,
    hidden_size=8,
    epochs=1,
    batch_size=32,
    learning_rate=1e-3,
    rollout_loss_weight=1.0,
    diagnostic_horizons=(1, 2),
    windows_per_episode=2,
    calibration_bins=2,
)
```

Assert top-level names are exactly:

```python
{
    "experiment_manifest.json",
    "comparison.json",
    "comparison.csv",
    "comparison.png",
    "baseline_diagnostics",
    "bootstrap_diagnostics",
    "runs",
}
```

Assert each `runs/seed_N/world_model.pt` loads with matching `seed` and
`bootstrap_seed`; both diagnostics contain the five existing ensemble files;
manifest checkpoint records are seed-sorted; both JSON objects pass
`json.dumps(..., allow_nan=False)`; and returned paths name every artifact.

Run this test and confirm RED because `run_bootstrap_experiment()` is absent.

- [ ] **Step 3: Implement validation and cross-ensemble compatibility**

Implement:

```python
def _validate_comparable_ensembles(
    baseline: WorldModelEnsemble,
    bootstrap: WorldModelEnsemble,
) -> None:
```

Require identical seeds; train/validation/test arrays; input/target mean and std
arrays; and per-seed `split_seed`, `hidden_size`, `epochs`, `batch_size`,
`learning_rate`, `rollout_horizon`, and `rollout_loss_weight`. Require
`bootstrap_seed == seed` for every bootstrap member and reject bootstrap fields
on baseline members. Resolve every member's `training_config["data_path"]` and
require it to equal the current resolved dataset path before training starts.

At the start of `run_bootstrap_experiment()`, validate regular input files,
unique non-negative seeds, positive counts, finite optimizer values, strictly
increasing positive diagnostic horizons, and absent-or-empty output. Load and
validate the baseline ensemble before creating the output directory.

- [ ] **Step 4: Implement serial member training and diagnostics**

Use this signature:

```python
def run_bootstrap_experiment(
    *,
    data_path: Path | str,
    baseline_checkpoint_paths: Iterable[Path | str],
    output_dir: Path | str,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    split_seed: int = 0,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    rollout_loss_weight: float = 1.0,
    diagnostic_horizons: Iterable[int] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    calibration_bins: int = 10,
) -> dict[str, Any]:
```

For each sorted seed, call `run_training()` with `rollout_horizon=10` and
`bootstrap_seed=seed`, writing `runs/seed_{seed}/world_model.pt`. Load the
bootstrap ensemble and call `_validate_comparable_ensembles()`.

Run `run_ensemble_diagnostics()` twice with identical parameters:

```text
baseline checkpoint paths -> baseline_diagnostics/
bootstrap checkpoint paths -> bootstrap_diagnostics/
```

Read both `metrics.json` files and call `build_bootstrap_comparison()`.

- [ ] **Step 5: Write final artifacts only after prerequisites succeed**

Write `comparison.json` with `allow_nan=False`, then CSV and PNG. Build
`experiment_manifest.json` schema version 1 with:

- absolute dataset path and SHA-256;
- seed-sorted external baseline paths and SHA-256 values;
- relative bootstrap checkpoint paths and SHA-256 values;
- seeds and complete H10 training protocol;
- `bootstrap_strategy="episode_with_replacement"`;
- `bootstrap_draws="train_episode_count"`;
- diagnostic horizons, windows per episode, and calibration bins;
- relative paths to both diagnostic bundles and comparison artifacts.

Return paths for the manifest, comparison, CSV, plot, diagnostics, and bootstrap
checkpoints. Preserve partial run/diagnostic directories on failure, but do not
write comparison or manifest files before all prerequisite work succeeds.

- [ ] **Step 6: Implement `main()` and register the script**

Follow existing CLIs: parse all signature parameters, catch `FileNotFoundError`
and `ValueError` with `parser.error(str(error))`, and print sorted indented JSON.
Add the exact script line to `pyproject.toml`.

- [ ] **Step 7: Add mismatch and failure-order tests**

Add tests for baseline seeds differing from `--seeds`, baseline hidden-size,
split, or dataset-path mismatch, non-empty output, and missing dataset CLI
error. Patch `run_training` only for a test where the second seed raises, and
patch `run_ensemble_diagnostics` only for a diagnostic-failure test; in both
cases assert neither `comparison.json` nor `experiment_manifest.json` exists.

- [ ] **Step 8: Run focused and affected suites**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_bootstrap \
  tests.test_train_world_model \
  tests.test_ensemble \
  tests.test_diagnose_ensemble \
  tests.test_bootstrap_experiment \
  -v
```

Expected: all focused and affected tests pass.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/world_model_lab/bootstrap_experiment.py tests/test_bootstrap_experiment.py
git commit -m "feat: run bootstrap ensemble experiment"
```

---

### Task 7: Documentation, Full Verification, and Real Five-Member Run

**Files:**
- Modify: `README.md` after the existing `Ensemble uncertainty diagnostics` section.

**Interfaces:**
- Documents: bootstrap hypothesis, shared-normalizer invariant, CLI, artifacts, and delta signs.
- Verifies: complete repository and real five-member experiment.

- [ ] **Step 1: Add README usage and interpretation**

Add `## Episode bootstrap ensemble experiment` stating:

- only training episodes are resampled;
- baseline checkpoints are reused and not retrained;
- normalizers and held-out splits remain shared;
- error delta is `bootstrap - baseline`, so negative is better;
- correlation delta is `bootstrap - baseline`, so positive is better;
- lack of improvement is a valid result and blocks MPC integration.

Document:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/world-model-bootstrap-ensemble \
  --data data/transitions.npz \
  --baseline-checkpoints \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_0/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_1/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_2/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_3/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_4/h10/world_model.pt \
  --output-dir artifacts/experiments/bootstrap-h10-seeds-0-4 \
  --seeds 0 1 2 3 4 \
  --split-seed 0 \
  --rollout-loss-weight 1.0
```

List every designed artifact and explain that `runs/` contains only the five
new bootstrap checkpoints.

- [ ] **Step 2: Run full automated verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests
git diff --check
git status --short
```

Expected: every test passes, compileall and diff check exit zero, and only the
intended README change remains uncommitted.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain bootstrap ensemble experiment"
```

- [ ] **Step 4: Run the real five-member experiment**

Use the existing verified H10 baseline checkpoints and canonical dataset:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.bootstrap_experiment \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --baseline-checkpoints \
    /private/tmp/world_model_lab-ensemble/artifacts/experiments/h10-seeds-0-4/runs/seed_0/h10/world_model.pt \
    /private/tmp/world_model_lab-ensemble/artifacts/experiments/h10-seeds-0-4/runs/seed_1/h10/world_model.pt \
    /private/tmp/world_model_lab-ensemble/artifacts/experiments/h10-seeds-0-4/runs/seed_2/h10/world_model.pt \
    /private/tmp/world_model_lab-ensemble/artifacts/experiments/h10-seeds-0-4/runs/seed_3/h10/world_model.pt \
    /private/tmp/world_model_lab-ensemble/artifacts/experiments/h10-seeds-0-4/runs/seed_4/h10/world_model.pt \
  --output-dir artifacts/experiments/bootstrap-h10-seeds-0-4 \
  --seeds 0 1 2 3 4 \
  --split-seed 0 \
  --rollout-loss-weight 1.0
```

Do not commit generated `artifacts/` files.

Expected: five bootstrap checkpoints and both diagnostic bundles are produced;
the CLI prints paths and seeds; JSON writing does not fail.

- [ ] **Step 5: Validate the real artifact bundle**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -c '
import json
from pathlib import Path
root = Path("artifacts/experiments/bootstrap-h10-seeds-0-4")
manifest = json.loads((root / "experiment_manifest.json").read_text())
comparison = json.loads((root / "comparison.json").read_text())
assert manifest["training"]["seeds"] == [0, 1, 2, 3, 4]
assert comparison["horizons"] == [1, 5, 10, 20, 50]
assert len(list((root / "runs").glob("seed_*/world_model.pt"))) == 5
json.dumps(manifest, allow_nan=False)
json.dumps(comparison, allow_nan=False)
print(json.dumps(comparison["rollout"]["10"], indent=2, sort_keys=True))
'
```

Expected: assertions pass and H10 comparison is printed for interpretation.

- [ ] **Step 6: Final branch verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v
git diff --check main...HEAD
git status --short --branch
```

Expected: full suite passes, branch diff is clean, and generated artifacts
remain ignored rather than staged.
