# Rollout Step Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the held-out world-model diagnostic bundle with dense rollout-step physical errors and normalized loss-component curves so the H1 and H10 models can be compared at every recursive prediction step.

**Architecture:** Keep all state-error math and episode-macro aggregation in `diagnostics.py`, reusing the teacher-forced and free-rollout predictions that are already cached for the maximum-horizon windows. Keep `diagnostic_plots.py` presentation-only, preserve the schema-version-1 sparse plotting fallback, and let `diagnose_model.py` remain the thin file-output orchestrator.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, PyTorch checkpoint structures, unittest

## Global Constraints

- Do not change `WorldModelMLP`, checkpoint contents, training loss, optimizer behavior, sequence sampling, dataset splits, diagnostic CLI options, MPC, PPO, rewards, or termination prediction.
- Dense arrays must contain exactly `max_horizon` finite floats and `steps` must equal `[1, 2, ..., max_horizon]`.
- Physical metrics are position in metres, wrapped absolute heading error in degrees, and absolute velocity error in metres per second.
- Normalized components are squared wrapped state errors divided by checkpoint `target_std`; `total` is the arithmetic mean of `x`, `y`, `heading`, and `velocity`.
- Every dense value uses episode-macro aggregation: average windows within each episode, then average episodes with equal weight.
- Dense metrics and sparse-horizon summaries must reuse the same cached predictions; the feature must not add model inference calls.
- Top-level diagnostic `schema_version` becomes `2`; the manifest stays at schema version `1`.
- `plot_rollout_errors()` must accept both schema-version-1 sparse reports and schema-version-2 dense reports.
- Generated checkpoints, JSON reports, and PNG images remain ignored and must not be committed.

---

### Task 1: Wrapped State Differences and Normalized Loss Components

**Files:**
- Modify: `src/world_model_lab/diagnostics.py:12-73`
- Modify: `tests/test_diagnostics.py:8-87`

**Interfaces:**
- Consumes: state arrays with shape `[N, 4]` and checkpoint `target_std` with shape `[4]`.
- Produces: `_state_differences(predicted_states, true_states) -> np.ndarray`, `_validate_target_std(target_std) -> np.ndarray`, and `_compute_normalized_squared_errors(predicted_states, true_states, target_std) -> dict[str, np.ndarray]`.
- Preserves: `compute_state_errors(predicted_states, true_states) -> dict[str, np.ndarray]` with the existing public behavior.

- [ ] **Step 1: Write failing tests for normalization and heading wrapping**

Add `_compute_normalized_squared_errors` to the import from `world_model_lab.diagnostics`, then add these tests after `test_state_errors_use_euclidean_position_and_wrapped_heading`:

```python
    def test_normalized_squared_errors_use_target_std_and_wrapped_heading(self):
        true = np.asarray([[0.0, 0.0, math.radians(-179.0), 1.0]])
        predicted = np.asarray([[3.0, 4.0, math.radians(179.0), 1.25]])
        target_std = np.asarray([2.0, 4.0, math.radians(1.0), 0.5])

        errors = _compute_normalized_squared_errors(
            predicted,
            true,
            target_std,
        )

        np.testing.assert_allclose(errors["x"], [2.25])
        np.testing.assert_allclose(errors["y"], [1.0])
        np.testing.assert_allclose(errors["heading"], [4.0], atol=1e-10)
        np.testing.assert_allclose(errors["velocity"], [0.25])
        np.testing.assert_allclose(errors["total"], [1.875], atol=1e-10)

    def test_normalized_squared_errors_reject_invalid_target_std(self):
        states = np.zeros((1, 4), dtype=np.float64)
        invalid_values = (
            np.ones(3),
            np.asarray([1.0, 1.0, 0.0, 1.0]),
            np.asarray([1.0, 1.0, np.inf, 1.0]),
        )

        for target_std in invalid_values:
            with self.subTest(target_std=target_std):
                with self.assertRaisesRegex(
                    ValueError,
                    "target_std must have shape.*finite positive",
                ):
                    _compute_normalized_squared_errors(
                        states,
                        states,
                        target_std,
                    )
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_diagnostics.DiagnosticsTest.test_normalized_squared_errors_use_target_std_and_wrapped_heading \
  tests.test_diagnostics.DiagnosticsTest.test_normalized_squared_errors_reject_invalid_target_std -v
```

Expected: import error because `_compute_normalized_squared_errors` does not exist.

- [ ] **Step 3: Implement one wrapped-difference path for physical and normalized errors**

Add the state-component constant after `ERROR_NAMES`, add the three helpers below before `compute_state_errors`, and replace the body of `compute_state_errors` with the shown version:

```python
ERROR_NAMES = ("position", "heading_degrees", "velocity")
STATE_COMPONENT_NAMES = ("x", "y", "heading", "velocity")


def _state_differences(
    predicted_states: np.ndarray,
    true_states: np.ndarray,
) -> np.ndarray:
    predicted = np.asarray(predicted_states, dtype=np.float64)
    true = np.asarray(true_states, dtype=np.float64)
    if (
        predicted.shape != true.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 4
    ):
        raise ValueError("predicted and true states must have matching shape [N, 4]")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(true)):
        raise ValueError("predicted and true states must contain only finite values")

    difference = predicted - true
    difference[:, 2] = wrap_angle(difference[:, 2])
    return difference


def _validate_target_std(target_std: np.ndarray) -> np.ndarray:
    array = np.asarray(target_std, dtype=np.float64)
    if (
        array.shape != (4,)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise ValueError(
            "target_std must have shape [4] and contain only finite positive values"
        )
    return array


def _compute_normalized_squared_errors(
    predicted_states: np.ndarray,
    true_states: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, np.ndarray]:
    difference = _state_differences(predicted_states, true_states)
    normalized_squared = np.square(difference / _validate_target_std(target_std))
    result = {
        name: normalized_squared[:, index]
        for index, name in enumerate(STATE_COMPONENT_NAMES)
    }
    result["total"] = np.mean(normalized_squared, axis=1)
    return result


def compute_state_errors(
    predicted_states: np.ndarray,
    true_states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute absolute errors in physical units for matching state batches."""

    difference = _state_differences(predicted_states, true_states)
    return {
        "position": np.linalg.norm(difference[:, :2], axis=1),
        "heading_degrees": np.degrees(np.abs(difference[:, 2])),
        "velocity": np.abs(difference[:, 3]),
    }
```

- [ ] **Step 4: Run primitive and existing diagnostic tests**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnostics -v
```

Expected: every test in `tests.test_diagnostics` passes, including the existing 179-degree to -179-degree physical-heading assertion.

- [ ] **Step 5: Commit the numerical primitives**

```bash
git add src/world_model_lab/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: compute normalized rollout error components"
```

---

### Task 2: Dense Per-Step Metrics with Episode-Macro Aggregation

**Files:**
- Modify: `src/world_model_lab/diagnostics.py:375-603`
- Modify: `tests/test_diagnostics.py:26-301`

**Interfaces:**
- Consumes: `window_predictions: list[tuple[RolloutWindow, np.ndarray, np.ndarray]]`, validated checkpoint `target_std`, and `max_horizon`.
- Produces: `_build_step_curves(window_predictions, target_std, max_horizon) -> dict[str, object]` and `rollout.step_curves` in metrics schema version `2`.
- Preserves: existing `rollout.protocol` and `rollout.horizons` values and their distribution summaries.

- [ ] **Step 1: Make the synthetic checkpoint helper support an exact target standard deviation**

Replace `make_constant_delta_world_model` in `tests/test_diagnostics.py` with:

```python
def make_constant_delta_world_model(
    delta: np.ndarray,
    *,
    test_episode_ids: np.ndarray,
    target_std: np.ndarray | None = None,
) -> LoadedWorldModel:
    target_std_array = (
        np.ones(4, dtype=np.float64)
        if target_std is None
        else np.asarray(target_std, dtype=np.float64)
    )
    model = WorldModelMLP(hidden_size=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(
            torch.as_tensor(delta / target_std_array, dtype=torch.float32)
        )
    model.eval()
    return LoadedWorldModel(
        model=model,
        input_normalizer=Normalizer(mean=np.zeros(7), std=np.ones(7)),
        target_normalizer=Normalizer(
            mean=np.zeros(4),
            std=target_std_array,
        ),
        split_episode_ids={
            "train": np.asarray([0]),
            "validation": np.asarray([99]),
            "test": np.asarray(test_episode_ids),
        },
        training_config={},
        train_losses=[],
        validation_losses=[],
        best_epoch=0,
        test_metrics={},
    )
```

- [ ] **Step 2: Extend the compounding-error test with exact dense curves**

In `test_free_rollout_exposes_compounding_error`, pass `target_std=np.asarray([2.0, 1.0, 1.0, 1.0])` when building the model. After the existing horizon assertions, add:

```python
        self.assertEqual(metrics["schema_version"], 2)
        curves = metrics["rollout"]["step_curves"]
        self.assertEqual(curves["steps"], [1, 2, 3])
        self.assertEqual(curves["aggregation"], "episode_macro_mean")
        np.testing.assert_allclose(
            curves["teacher_forcing"]["physical"]["position"],
            [0.0, 1.0, 2.0],
        )
        np.testing.assert_allclose(
            curves["free_rollout"]["physical"]["position"],
            [0.0, 1.0, 3.0],
        )
        np.testing.assert_allclose(
            curves["teacher_forcing"]["normalized_mse"]["x"],
            [0.0, 0.25, 1.0],
        )
        np.testing.assert_allclose(
            curves["free_rollout"]["normalized_mse"]["x"],
            [0.0, 0.25, 2.25],
        )
        np.testing.assert_allclose(
            curves["teacher_forcing"]["normalized_mse"]["total"],
            [0.0, 0.0625, 0.25],
        )
        np.testing.assert_allclose(
            curves["free_rollout"]["normalized_mse"]["total"],
            [0.0, 0.0625, 0.5625],
        )
        for horizon in (1, 2, 3):
            sparse = metrics["rollout"]["horizons"][str(horizon)]
            for mode_name in ("teacher_forcing", "free_rollout"):
                self.assertAlmostEqual(
                    sparse[mode_name]["position"]["mean"],
                    curves[mode_name]["physical"]["position"][horizon - 1],
                )
        for mode_name in ("teacher_forcing", "free_rollout"):
            for group in ("physical", "normalized_mse"):
                for values in curves[mode_name][group].values():
                    self.assertEqual(len(values), 3)
                    self.assertTrue(np.all(np.isfinite(values)))
```

In `test_rollout_metrics_weight_episodes_equally`, add these assertions after the sparse summary checks:

```python
        curves = metrics["rollout"]["step_curves"]
        self.assertEqual(
            curves["free_rollout"]["physical"]["position"],
            [1.0],
        )
        self.assertEqual(
            curves["free_rollout"]["normalized_mse"]["x"],
            [2.0],
        )
        self.assertEqual(
            curves["free_rollout"]["normalized_mse"]["total"],
            [0.5],
        )
```

Add this integration test after the episode-weighting test so the invalid
checkpoint value is rejected at the metric-builder boundary, not only by the
numerical helper:

```python
    def test_build_diagnostic_metrics_rejects_invalid_checkpoint_target_std(self):
        train_episode = make_state_sequence(
            np.asarray([-1.0, 0.0, 0.0, 0.0]),
            np.zeros(4),
            steps=1,
        )
        test_episode = make_state_sequence(
            np.asarray([0.0, 0.0, 0.0, 0.0]),
            np.zeros(4),
            steps=1,
        )
        arrays = arrays_from_episodes({0: train_episode, 1: test_episode})
        world_model = make_constant_delta_world_model(
            np.zeros(4),
            test_episode_ids=np.asarray([1]),
        )
        world_model.target_normalizer = Normalizer(
            mean=np.zeros(4),
            std=np.asarray([1.0, 1.0, 0.0, 1.0]),
        )

        with self.assertRaisesRegex(
            ValueError,
            "target_std must have shape.*finite positive",
        ):
            build_diagnostic_metrics(
                world_model,
                arrays=arrays,
                split_episode_ids=world_model.split_episode_ids,
                horizons=(1,),
                windows_per_episode=1,
                xy_bins=2,
                feature_bins=2,
                min_bin_count=1,
            )
```

- [ ] **Step 3: Run the dense-curve tests and confirm RED**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_diagnostics.DiagnosticsTest.test_free_rollout_exposes_compounding_error \
  tests.test_diagnostics.DiagnosticsTest.test_rollout_metrics_weight_episodes_equally \
  tests.test_diagnostics.DiagnosticsTest.test_build_diagnostic_metrics_rejects_invalid_checkpoint_target_std -v
```

Expected: failure because `schema_version` is still `1` and `step_curves` is absent.

- [ ] **Step 4: Add named record and macro-curve helpers**

Add the constant after `STATE_COMPONENT_NAMES`:

```python
NORMALIZED_MSE_NAMES = (*STATE_COMPONENT_NAMES, "total")
```

Add these helpers after `_summarize_episode_records`:

```python
def _empty_named_records(names: tuple[str, ...]) -> dict[str, list[float]]:
    return {name: [] for name in names}


def _append_episode_errors(
    records: dict[int, dict[str, list[float]]],
    *,
    episode_id: int,
    errors: Mapping[str, np.ndarray],
    step_index: int,
    names: tuple[str, ...],
) -> None:
    episode_records = records.setdefault(
        episode_id,
        _empty_named_records(names),
    )
    for name in names:
        episode_records[name].append(float(errors[name][step_index]))


def _macro_mean_curves(
    records_by_step: list[dict[int, dict[str, list[float]]]],
    *,
    names: tuple[str, ...],
) -> dict[str, list[float]]:
    curves = {name: [] for name in names}
    for records in records_by_step:
        episode_ids = sorted(records)
        if not episode_ids:
            raise ValueError("step curves require at least one episode")
        for name in names:
            per_episode_means = np.asarray(
                [np.mean(records[episode_id][name]) for episode_id in episode_ids],
                dtype=np.float64,
            )
            value = float(np.mean(per_episode_means))
            if not np.isfinite(value):
                raise ValueError("step curves must contain only finite values")
            curves[name].append(value)
    return curves


def _build_step_curves(
    window_predictions: list[tuple[RolloutWindow, np.ndarray, np.ndarray]],
    *,
    target_std: np.ndarray,
    max_horizon: int,
) -> dict[str, object]:
    mode_records = {
        mode_name: {
            "physical": [{} for _ in range(max_horizon)],
            "normalized_mse": [{} for _ in range(max_horizon)],
        }
        for mode_name in ("teacher_forcing", "free_rollout")
    }

    for window, teacher_predictions, free_predictions in window_predictions:
        true_states = window.true_states[1 : max_horizon + 1]
        predictions_by_mode = {
            "teacher_forcing": teacher_predictions,
            "free_rollout": free_predictions[1:],
        }
        for mode_name, predictions in predictions_by_mode.items():
            if predictions.shape != (max_horizon, 4):
                raise ValueError(
                    "cached rollout predictions must have shape [max_horizon, 4]"
                )
            physical_errors = compute_state_errors(predictions, true_states)
            normalized_errors = _compute_normalized_squared_errors(
                predictions,
                true_states,
                target_std,
            )
            for step_index in range(max_horizon):
                _append_episode_errors(
                    mode_records[mode_name]["physical"][step_index],
                    episode_id=window.episode_id,
                    errors=physical_errors,
                    step_index=step_index,
                    names=ERROR_NAMES,
                )
                _append_episode_errors(
                    mode_records[mode_name]["normalized_mse"][step_index],
                    episode_id=window.episode_id,
                    errors=normalized_errors,
                    step_index=step_index,
                    names=NORMALIZED_MSE_NAMES,
                )

    return {
        "steps": list(range(1, max_horizon + 1)),
        "aggregation": "episode_macro_mean",
        **{
            mode_name: {
                "physical": _macro_mean_curves(
                    records["physical"],
                    names=ERROR_NAMES,
                ),
                "normalized_mse": _macro_mean_curves(
                    records["normalized_mse"],
                    names=NORMALIZED_MSE_NAMES,
                ),
            }
            for mode_name, records in mode_records.items()
        },
    }
```

- [ ] **Step 5: Assemble schema version 2 from the cached predictions**

In `build_diagnostic_metrics`, validate the checkpoint standard deviation immediately after the existing numeric option checks:

```python
    target_std = _validate_target_std(world_model.target_normalizer.std)
```

Immediately after the loop that constructs `window_predictions`, add:

```python
    step_curves = _build_step_curves(
        window_predictions,
        target_std=target_std,
        max_horizon=horizon_values[-1],
    )
```

In the return value, change the top-level version and add the new rollout key without changing the existing protocol or sparse metrics:

```python
        "schema_version": 2,
```

```python
        "rollout": {
            "protocol": {
                "horizons": list(horizon_values),
                "max_horizon": horizon_values[-1],
                "windows_per_episode": windows_per_episode,
                "eligible_episode_ids": selection.eligible_episode_ids.tolist(),
                "skipped_episode_ids": selection.skipped_episode_ids.tolist(),
                "windows": [
                    {
                        "episode_id": window.episode_id,
                        "start_step": window.start_step,
                    }
                    for window in selection.windows
                ],
            },
            "horizons": horizon_metrics,
            "step_curves": step_curves,
        },
```

This call occurs after all teacher-forced and free-rollout arrays have been cached and therefore performs only NumPy error computation and aggregation.

- [ ] **Step 6: Run focused and complete diagnostic-core tests**

Run:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_diagnostics.DiagnosticsTest.test_free_rollout_exposes_compounding_error \
  tests.test_diagnostics.DiagnosticsTest.test_rollout_metrics_weight_episodes_equally \
  tests.test_diagnostics.DiagnosticsTest.test_build_diagnostic_metrics_rejects_invalid_checkpoint_target_std -v

env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnostics -v
```

Expected: both focused tests pass; the complete diagnostic-core suite passes; `json.dumps(metrics, allow_nan=False)` remains successful.

- [ ] **Step 7: Commit schema version 2 metrics**

```bash
git add src/world_model_lab/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: add dense rollout step metrics"
```

---

### Task 3: Dense Physical Plot and Normalized Component Plot

**Files:**
- Modify: `src/world_model_lab/diagnostic_plots.py:151-192`
- Modify: `tests/test_diagnostic_plots.py:1-115`

**Interfaces:**
- Consumes: schema-version-2 `rollout.step_curves` and legacy schema-version-1 `rollout.horizons`.
- Produces: `_physical_rollout_curve(rollout, mode_name, metric_name) -> tuple[np.ndarray, np.ndarray]`, updated `plot_rollout_errors(metrics, output_path) -> Path`, and new `plot_rollout_loss_components(metrics, output_path) -> Path`.
- Preserves: valid PNG output for the current schema-version-1 fixture.

- [ ] **Step 1: Add a schema-version-2 plotting fixture**

Add `import copy`, `import numpy as np`, `import matplotlib.pyplot as plt`, and `from unittest.mock import patch` to `tests/test_diagnostic_plots.py`. Add `_physical_rollout_curve` and `plot_rollout_loss_components` to the diagnostic-plot imports. Then add:

```python
def make_dense_metrics() -> dict[str, object]:
    metrics = copy.deepcopy(make_metrics())
    metrics["schema_version"] = 2
    rollout = metrics["rollout"]
    rollout["protocol"]["max_horizon"] = 3
    rollout["step_curves"] = {
        "steps": [1, 2, 3],
        "aggregation": "episode_macro_mean",
        "teacher_forcing": {
            "physical": {
                "position": [0.1, 0.2, 0.3],
                "heading_degrees": [1.0, 2.0, 3.0],
                "velocity": [0.01, 0.02, 0.03],
            },
            "normalized_mse": {
                "x": [0.01, 0.02, 0.03],
                "y": [0.02, 0.03, 0.04],
                "heading": [0.03, 0.04, 0.05],
                "velocity": [0.04, 0.05, 0.06],
                "total": [0.025, 0.035, 0.045],
            },
        },
        "free_rollout": {
            "physical": {
                "position": [0.1, 0.4, 0.9],
                "heading_degrees": [1.0, 4.0, 9.0],
                "velocity": [0.01, 0.04, 0.09],
            },
            "normalized_mse": {
                "x": [0.01, 0.04, 0.09],
                "y": [0.02, 0.05, 0.10],
                "heading": [0.03, 0.06, 0.11],
                "velocity": [0.04, 0.07, 0.12],
                "total": [0.025, 0.055, 0.105],
            },
        },
    }
    return metrics
```

- [ ] **Step 2: Write failing tests for dense selection, v1 fallback, and the four-panel plot**

Replace `test_plot_rollout_errors_saves_png` and add the component test:

```python
    def test_physical_rollout_curve_prefers_dense_steps_and_falls_back_to_v1(self):
        dense_rollout = make_dense_metrics()["rollout"]
        dense_steps, dense_values = _physical_rollout_curve(
            dense_rollout,
            "free_rollout",
            "position",
        )
        np.testing.assert_array_equal(dense_steps, [1, 2, 3])
        np.testing.assert_allclose(dense_values, [0.1, 0.4, 0.9])

        sparse_rollout = make_metrics()["rollout"]
        sparse_steps, sparse_values = _physical_rollout_curve(
            sparse_rollout,
            "free_rollout",
            "position",
        )
        np.testing.assert_array_equal(sparse_steps, [1, 2])
        np.testing.assert_allclose(sparse_values, [0.1, 0.4])

    def test_plot_rollout_errors_saves_v1_and_v2_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1_output = root / "rollout-errors-v1.png"
            v2_output = root / "rollout-errors-v2.png"

            self.assertEqual(plot_rollout_errors(make_metrics(), v1_output), v1_output)
            self.assertEqual(
                plot_rollout_errors(make_dense_metrics(), v2_output),
                v2_output,
            )
            signatures = (v1_output.read_bytes()[:8], v2_output.read_bytes()[:8])

        self.assertEqual(signatures, (b"\x89PNG\r\n\x1a\n",) * 2)

    def test_plot_rollout_loss_components_draws_four_named_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rollout-loss-components.png"
            with patch.object(plt, "close"):
                returned = plot_rollout_loss_components(
                    make_dense_metrics(),
                    output,
                )
                figure = plt.gcf()
                titles = [axis.get_title() for axis in figure.axes]
                labels = [line.get_label() for line in figure.axes[0].lines]
                y_bottoms = [axis.get_ylim()[0] for axis in figure.axes]
            plt.close(figure)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(titles, ["x", "y", "heading", "velocity"])
        self.assertEqual(labels, ["Teacher forcing", "Free rollout"])
        self.assertEqual(y_bottoms, [0.0, 0.0, 0.0, 0.0])

    def test_schema_v2_rollout_plot_requires_step_curves(self):
        metrics = make_dense_metrics()
        del metrics["rollout"]["step_curves"]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "schema version 2.*step_curves"):
                plot_rollout_errors(
                    metrics,
                    Path(directory) / "rollout-errors.png",
                )
```

- [ ] **Step 3: Run plotting tests and confirm RED**

Run:

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnostic_plots -v
```

Expected: import error because `_physical_rollout_curve` and `plot_rollout_loss_components` do not exist.

- [ ] **Step 4: Add dense physical-curve selection with legacy fallback**

Add this helper before `plot_rollout_errors`:

```python
def _physical_rollout_curve(
    rollout: Mapping[str, Any],
    mode_name: str,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    step_curves = rollout.get("step_curves")
    if step_curves is not None:
        return (
            np.asarray(step_curves["steps"], dtype=np.int64),
            np.asarray(
                step_curves[mode_name]["physical"][metric_name],
                dtype=np.float64,
            ),
        )

    horizons = np.asarray(rollout["protocol"]["horizons"], dtype=np.int64)
    values = np.asarray(
        [
            rollout["horizons"][str(int(horizon))][mode_name][metric_name]["mean"]
            for horizon in horizons
        ],
        dtype=np.float64,
    )
    return horizons, values
```

In `plot_rollout_errors`, remove the single `horizons` list and replace the inner value construction and plot call with:

```python
    if int(metrics.get("schema_version", 1)) >= 2 and "step_curves" not in rollout:
        raise ValueError("schema version 2 rollout metrics require step_curves")

```

Place that validation immediately after `rollout = metrics["rollout"]`. Then
replace the inner value construction and plot call with:

```python
            steps, values = _physical_rollout_curve(
                rollout,
                mode_name,
                metric_name,
            )
            axis.plot(
                steps,
                values,
                marker=None if "step_curves" in rollout else "o",
                linewidth=2,
                label=label,
                color=color,
            )
```

Keep the current titles, units, colors, legends, and `Teacher Forcing vs Free Rollout` figure title.

- [ ] **Step 5: Implement the normalized four-component plot**

Add after `plot_rollout_errors`:

```python
def plot_rollout_loss_components(
    metrics: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    """Plot normalized rollout MSE components for every diagnostic step."""

    rollout = metrics["rollout"]
    step_curves = rollout.get("step_curves")
    if step_curves is None:
        raise ValueError("normalized rollout components require step_curves")
    steps = np.asarray(step_curves["steps"], dtype=np.int64)
    modes = (
        ("teacher_forcing", "Teacher forcing", "#4c78a8"),
        ("free_rollout", "Free rollout", "#e45756"),
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, component_name in zip(
        axes.flat,
        ("x", "y", "heading", "velocity"),
        strict=True,
    ):
        for mode_name, label, color in modes:
            values = np.asarray(
                step_curves[mode_name]["normalized_mse"][component_name],
                dtype=np.float64,
            )
            axis.plot(
                steps,
                values,
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set(
            title=component_name,
            xlabel="rollout step",
            ylabel="normalized MSE",
        )
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("Normalized Rollout Loss Components")
    return _save_figure(figure, output_path)
```

- [ ] **Step 6: Run plotting tests**

Run:

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnostic_plots -v
```

Expected: overview, legacy rollout, dense rollout, and component-plot tests all pass and all written files have a PNG signature.

- [ ] **Step 7: Commit plotting behavior**

```bash
git add src/world_model_lab/diagnostic_plots.py tests/test_diagnostic_plots.py
git commit -m "feat: plot dense rollout diagnostics"
```

---

### Task 4: Diagnostic Bundle, Documentation, and Controlled H1/H10 Rerun

**Files:**
- Modify: `src/world_model_lab/diagnose_model.py:13-137`
- Modify: `tests/test_diagnose_model.py:83-120`
- Modify: `README.md:306-328`
- Verify only: all source and test files
- Generate but do not commit: `artifacts/world_model_h1.pt`
- Generate but do not commit: `artifacts/world_model_h10.pt`
- Generate but do not commit: `artifacts/diagnostics/h1/`
- Generate but do not commit: `artifacts/diagnostics/h10/`

**Interfaces:**
- Consumes: `plot_rollout_loss_components(metrics, output_path) -> Path` from Task 3.
- Produces: `rollout_loss_components.png` and summary key `rollout_loss_components_plot`.
- Documents: dense physical curves, normalized component curves, schema version 2, and the unchanged CLI protocol.

- [ ] **Step 1: Extend the bundle regression test**

In `test_run_diagnostics_writes_reproducible_output_bundle`, create `second_output_dir = root / "diagnostics-second"`, invoke `run_diagnostics` a second time with the same arguments and that output directory, and capture the JSON bytes before leaving the temporary directory:

```python
            second_output_dir = root / "diagnostics-second"
            second_summary = run_diagnostics(
                data_path=data_path,
                checkpoint_path=checkpoint_path,
                output_dir=second_output_dir,
                horizons=(1, 2),
                windows_per_episode=2,
                xy_bins=2,
                feature_bins=2,
                min_bin_count=1,
            )

            output_names = {
                path.name for path in output_dir.iterdir() if path.is_file()
            }
            metrics_bytes = (output_dir / "metrics.json").read_bytes()
            manifest_bytes = (output_dir / "manifest.json").read_bytes()
            metrics = json.loads(metrics_bytes)
            manifest = json.loads(manifest_bytes)
            dataset_hash = sha256_file(data_path)
            second_metrics_bytes = (second_output_dir / "metrics.json").read_bytes()
            second_manifest_bytes = (second_output_dir / "manifest.json").read_bytes()
```

Replace the output and version assertions with:

```python
        self.assertEqual(
            output_names,
            {
                "metrics.json",
                "manifest.json",
                "overview.png",
                "rollout_errors.png",
                "rollout_loss_components.png",
            },
        )
        self.assertEqual(metrics["schema_version"], 2)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(metrics_bytes, second_metrics_bytes)
        self.assertEqual(manifest_bytes, second_manifest_bytes)
        self.assertEqual(
            summary["rollout_loss_components_plot"],
            str(output_dir / "rollout_loss_components.png"),
        )
        self.assertEqual(
            second_summary["rollout_loss_components_plot"],
            str(second_output_dir / "rollout_loss_components.png"),
        )
```

Keep the existing hash, hidden-size, test-episode, horizon, and longest-horizon assertions.

- [ ] **Step 2: Run the bundle test and confirm RED**

Run:

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_diagnose_model.DiagnoseModelTest.test_run_diagnostics_writes_reproducible_output_bundle -v
```

Expected: failure because the new image and summary key are absent.

- [ ] **Step 3: Write and return the new component plot**

Replace the diagnostic-plot import in `diagnose_model.py` with:

```python
from .diagnostic_plots import (
    plot_diagnostic_overview,
    plot_rollout_errors,
    plot_rollout_loss_components,
)
```

Add the path beside the current rollout path:

```python
    rollout_components_path = output / "rollout_loss_components.png"
```

Call the plotter after `plot_rollout_errors`:

```python
    plot_rollout_loss_components(metrics, rollout_components_path)
```

Add the path to the returned summary after `rollout_plot`:

```python
        "rollout_loss_components_plot": str(rollout_components_path),
```

- [ ] **Step 4: Run bundle and affected tests**

Run:

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_model tests.test_diagnostic_plots -v
```

Expected: both modules pass; repeated JSON and manifest files are byte-identical; the manifest remains schema version 1.

- [ ] **Step 5: Document the schema and both rollout plots**

Update the `metrics.json` row in the README bundle table to:

```markdown
| `metrics.json` | schema v2：单步误差、状态/动作分箱、稀疏 horizon 汇总，以及每一步的物理误差和归一化 MSE 分量 |
```

Update the `rollout_errors.png` row and add the new row immediately after it:

```markdown
| `rollout_errors.png` | Teacher Forcing 与 Free Rollout 从第 1 步到最大 horizon 的稠密物理误差曲线 |
| `rollout_loss_components.png` | `x`、`y`、heading、velocity 四个归一化 MSE 分量的 2×2 对比图 |
```

Add this paragraph after the Teacher Forcing and Free Rollout definitions:

```markdown
`--horizons` 仍定义稀疏 benchmark 点，其中最大值同时决定稠密曲线长度。例如
`--horizons 1 5 10 20 50` 会保留五个带分布统计的 horizon，同时在
`metrics.json` 和两张 rollout 图中记录第 1 到第 50 步。归一化分量使用 checkpoint
保存的 target-delta 标准差，与多步训练目标处于同一尺度；`total` 是四个分量的算术
平均值。所有曲线先在同一 episode 内平均窗口，再对 episode 等权平均。
```

- [ ] **Step 6: Run the complete test suite and source checks**

Run:

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v

git diff --check
```

Expected: all existing and new tests pass with zero failures and zero errors; `git diff --check` prints no output.

- [ ] **Step 7: Recreate the deterministic H1 and H10 checkpoints**

Train both checkpoints from the same standalone-repository dataset, seed, epoch
count, and configuration as the previous controlled experiment:

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_world_model \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --output artifacts/world_model_h1.pt \
  --rollout-horizon 1 \
  --epochs 100 \
  --seed 0

env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_world_model \
  --data /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz \
  --output artifacts/world_model_h10.pt \
  --rollout-horizon 10 \
  --rollout-loss-weight 1.0 \
  --epochs 100 \
  --seed 0
```

Expected: both commands finish with finite JSON summaries; H1 reports zero
rollout loss and H10 reports finite non-zero rollout loss.

- [ ] **Step 8: Generate identical H1 and H10 diagnostic bundles**

Use the newly recreated checkpoints with the same held-out diagnostic protocol:

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

Expected: each output directory contains the five-file bundle; both `metrics.json` files report schema version 2 and exactly 50 dense steps.

- [ ] **Step 9: Identify the first heading divergence and compare loss components**

Run this read-only comparison:

```bash
/Users/andyzhao/Workspace/world_model_lab/.venv/bin/python - <<'PY'
import json
from pathlib import Path

reports = {
    name: json.loads(
        Path(f"artifacts/diagnostics/{name}/metrics.json").read_text()
    )
    for name in ("h1", "h10")
}
for report in reports.values():
    curves = report["rollout"]["step_curves"]
    assert curves["steps"] == list(range(1, 51))

h1 = reports["h1"]["rollout"]["step_curves"]["free_rollout"]
h10 = reports["h10"]["rollout"]["step_curves"]["free_rollout"]
heading_pairs = zip(
    h1["physical"]["heading_degrees"],
    h10["physical"]["heading_degrees"],
    strict=True,
)
first_worse = next(
    (step for step, (baseline, candidate) in enumerate(heading_pairs, 1)
     if candidate > baseline),
    None,
)
print("first_h10_heading_worse_step", first_worse)
for step in (1, 5, 10, 20, 50):
    index = step - 1
    print("step", step, {
        name: {
            component: report["rollout"]["step_curves"]["free_rollout"]
            ["normalized_mse"][component][index]
            for component in ("x", "y", "heading", "velocity", "total")
        }
        for name, report in reports.items()
    })
PY
```

Expected: the script prints either a concrete first step or `None`, followed by finite H1/H10 component values at steps 1, 5, 10, 20, and 50. Report the observed result without changing the seed or protocol after seeing it.

- [ ] **Step 10: Verify generated files are ignored and commit source changes**

Run:

```bash
git status --short
git diff --check
```

Expected: `artifacts/` does not appear in status. Then commit the bundle and documentation changes:

```bash
git add README.md src/world_model_lab/diagnose_model.py tests/test_diagnose_model.py
git commit -m "feat: add rollout component diagnostic bundle"
```

- [ ] **Step 11: Perform final verification on the committed tree**

Run:

```bash
env PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v

git status --short
git log --oneline -5
```

Expected: all tests pass, the worktree is clean, and the four feature commits appear above the design and plan commits.
