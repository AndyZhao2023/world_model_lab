# Model Diagnostics Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic diagnostic benchmark that explains one-step error, error regions, and teacher-forcing versus recursive-rollout error for the existing state-space world model.

**Architecture:** Add reusable inference helpers to the existing dataset/training boundary, then build a file-I/O-free diagnostic core, a plotting-only module, and a thin CLI orchestrator. Every run writes a versioned JSON/PNG bundle tied to SHA-256 fingerprints of the dataset and checkpoint.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, `unittest`

## Global Constraints

- Evaluate model errors only on checkpoint test episodes.
- Use deterministic maximum-horizon windows and the same windows at every requested horizon.
- Aggregate rollout windows within episode before aggregating across episodes.
- Compare teacher forcing and free rollout on identical states and recorded actions.
- Report position in metres, heading in degrees, and velocity in metres per second.
- Represent sparse or empty bin metrics as JSON `null`, never NaN.
- Do not modify model architecture, training behavior, or checkpoint format.
- Keep tests independent of generated real-data artifacts.

---

### Task 1: Reusable batch inference boundary

**Files:**
- Modify: `src/world_model_lab/dataset.py`
- Modify: `src/world_model_lab/train_world_model.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_train_world_model.py`

**Interfaces:**
- Produces: `build_model_inputs(states: np.ndarray, actions: np.ndarray) -> np.ndarray`.
- Produces: `predict_deltas(result, inputs: np.ndarray) -> np.ndarray` for `TrainingResult | LoadedWorldModel`.
- Preserves: `build_model_arrays(...)` and `evaluate_model(...)` behavior.

- [x] **Step 1: Write failing feature-construction test**

Add to `tests/test_dataset.py`:

```python
from world_model_lab.dataset import build_model_inputs

def test_build_model_inputs_encodes_heading_without_targets(self):
    states = np.asarray([[1.0, 2.0, math.pi / 2.0, 0.5]])
    actions = np.asarray([[0.1, -0.2]])

    inputs = build_model_inputs(states, actions)

    np.testing.assert_allclose(inputs, [[1.0, 2.0, 1.0, 0.0, 0.5, 0.1, -0.2]], atol=1e-12)
```

- [x] **Step 2: Run the test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_dataset.DatasetTest.test_build_model_inputs_encodes_heading_without_targets -v`

Expected: import failure because `build_model_inputs` does not exist.

- [x] **Step 3: Implement input construction and refactor the existing array builder**

Add to `dataset.py` and make `build_model_arrays` call it after validating `next_states`:

```python
def build_model_inputs(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("states must have shape [N, 4]")
    if actions.ndim != 2 or actions.shape != (states.shape[0], 2):
        raise ValueError("actions must have shape [N, 2]")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("state and action arrays must contain only finite values")
    return np.column_stack(
        (
            states[:, 0],
            states[:, 1],
            np.sin(states[:, 2]),
            np.cos(states[:, 2]),
            states[:, 3],
            actions[:, 0],
            actions[:, 1],
        )
    )
```

- [x] **Step 4: Run dataset tests and verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_dataset -v`

Expected: all dataset tests pass.

- [x] **Step 5: Write failing raw-delta prediction test**

Add to `tests/test_train_world_model.py`:

```python
from world_model_lab.train_world_model import predict_deltas

def test_predict_deltas_returns_denormalized_physical_values(self):
    inputs, targets = make_linear_dynamics(count=64)
    result = train_model(
        inputs[:48], targets[:48],
        validation_inputs=inputs[48:], validation_targets=targets[48:],
        hidden_size=16, epochs=2, batch_size=32, learning_rate=1e-3, seed=3,
    )

    predictions = predict_deltas(result, inputs[:5])

    self.assertEqual(predictions.shape, (5, 4))
    self.assertTrue(np.all(np.isfinite(predictions)))
```

- [x] **Step 6: Run the prediction test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_train_world_model.TrainWorldModelTest.test_predict_deltas_returns_denormalized_physical_values -v`

Expected: import failure because `predict_deltas` does not exist.

- [x] **Step 7: Extract the batch prediction function**

Add to `train_world_model.py` and call it from `evaluate_model`:

```python
def predict_deltas(
    result: TrainingResult | LoadedWorldModel,
    inputs: np.ndarray,
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != WorldModelMLP.input_size:
        raise ValueError("inputs must have shape [N, 7]")
    normalized_inputs = _as_float_tensor(result.input_normalizer.normalize(inputs))
    result.model.eval()
    with torch.no_grad():
        normalized_predictions = result.model(normalized_inputs).cpu().numpy()
    return result.target_normalizer.denormalize(normalized_predictions)
```

- [x] **Step 8: Verify inference tests and commit**

Run: `.venv/bin/python -m unittest tests.test_dataset tests.test_train_world_model -v`

Expected: all tests pass.

Commit: `git add src/world_model_lab/dataset.py src/world_model_lab/train_world_model.py tests/test_dataset.py tests/test_train_world_model.py && git commit -m "refactor: expose world model batch inference"`

---

### Task 2: Diagnostic statistics, bins, and deterministic windows

**Files:**
- Create: `src/world_model_lab/diagnostics.py`
- Create: `tests/test_diagnostics.py`

**Interfaces:**
- Produces: `RolloutWindow`, `WindowSelection` dataclasses.
- Produces: `summarize_values`, `compute_state_errors`, `select_rollout_windows`, `build_feature_slice`, and `build_xy_grid`.

- [x] **Step 1: Write failing statistics and angle-wrap tests**

Create `tests/test_diagnostics.py` with:

```python
import math
import unittest
import numpy as np

from world_model_lab.diagnostics import compute_state_errors, summarize_values

class DiagnosticsTest(unittest.TestCase):
    def test_state_errors_use_euclidean_position_and_wrapped_heading(self):
        true = np.asarray([[0.0, 0.0, math.radians(-179.0), 1.0]])
        predicted = np.asarray([[3.0, 4.0, math.radians(179.0), 1.25]])

        errors = compute_state_errors(predicted, true)

        np.testing.assert_allclose(errors["position"], [5.0])
        np.testing.assert_allclose(errors["heading_degrees"], [2.0], atol=1e-10)
        np.testing.assert_allclose(errors["velocity"], [0.25])

    def test_summary_reports_distribution_statistics(self):
        summary = summarize_values(np.asarray([1.0, 2.0, 3.0, 10.0]))
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 4.0)
        self.assertEqual(summary["median"], 2.5)
        self.assertAlmostEqual(summary["p90"], 7.9)
        self.assertEqual(summary["max"], 10.0)
```

- [x] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_diagnostics -v`

Expected: import failure because `world_model_lab.diagnostics` does not exist.

- [x] **Step 3: Implement statistics and state errors**

Create `diagnostics.py` with the dataclass imports and these functions:

```python
def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }

def compute_state_errors(predicted_states, true_states) -> dict[str, np.ndarray]:
    predicted = np.asarray(predicted_states, dtype=np.float64)
    true = np.asarray(true_states, dtype=np.float64)
    if predicted.shape != true.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("predicted and true states must have matching shape [N, 4]")
    difference = predicted - true
    return {
        "position": np.linalg.norm(difference[:, :2], axis=1),
        "heading_degrees": np.degrees(np.abs(wrap_angle(difference[:, 2]))),
        "velocity": np.abs(difference[:, 3]),
    }
```

- [x] **Step 4: Verify GREEN for statistics**

Run: `.venv/bin/python -m unittest tests.test_diagnostics.DiagnosticsTest.test_state_errors_use_euclidean_position_and_wrapped_heading tests.test_diagnostics.DiagnosticsTest.test_summary_reports_distribution_statistics -v`

Expected: both tests pass.

- [x] **Step 5: Add failing deterministic-window tests**

Add a synthetic two-episode dataset and assert:

```python
selection = select_rollout_windows(
    states=states,
    actions=actions,
    next_states=next_states,
    episode_ids=episode_ids,
    step_ids=step_ids,
    selected_episode_ids=np.asarray([3, 4]),
    max_horizon=3,
    windows_per_episode=3,
)
self.assertEqual(
    [(window.episode_id, window.start_step) for window in selection.windows],
    [(3, 0), (3, 2), (3, 4)],
)
np.testing.assert_array_equal(selection.eligible_episode_ids, [3])
np.testing.assert_array_equal(selection.skipped_episode_ids, [4])
```

- [x] **Step 6: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_diagnostics.DiagnosticsTest.test_rollout_windows_are_evenly_spaced_and_skip_short_episodes -v`

Expected: import or attribute failure for the window API.

- [x] **Step 7: Implement validated trajectory extraction and window selection**

Add immutable dataclasses and deterministic index selection:

```python
@dataclass(frozen=True)
class RolloutWindow:
    episode_id: int
    start_step: int
    true_states: np.ndarray
    actions: np.ndarray

@dataclass(frozen=True)
class WindowSelection:
    windows: tuple[RolloutWindow, ...]
    eligible_episode_ids: np.ndarray
    skipped_episode_ids: np.ndarray

def _evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    if limit == 1:
        return np.asarray([0], dtype=np.int64)
    return np.rint(np.linspace(0, count - 1, limit)).astype(np.int64)
```

`select_rollout_windows` must sort each requested episode by `step_ids`, require
steps `0..T-1`, verify `states[1:] == next_states[:-1]`, build the `T+1` true
state sequence, and slice exactly `max_horizon + 1` states plus `max_horizon`
actions for each selected start.

- [x] **Step 8: Add failing bin tests**

Test that `build_feature_slice` and `build_xy_grid` retain counts but return
`None` summaries below `min_bin_count`, and that the final edge includes the
maximum observed value.

- [x] **Step 9: Implement shared bin summaries**

Use `np.histogram_bin_edges`-equivalent linear edges from explicit minimum and
maximum values. Assign the maximum to the final bin using `np.searchsorted(...,
side="right") - 1` followed by clipping. Each bin/cell has this JSON-safe form:

```python
{
    "count": count,
    "position": None if count < min_bin_count else summarize_values(position_values),
    "heading_degrees": None if count < min_bin_count else summarize_values(heading_values),
    "velocity": None if count < min_bin_count else summarize_values(velocity_values),
}
```

- [x] **Step 10: Verify diagnostics primitives and commit**

Run: `.venv/bin/python -m unittest tests.test_diagnostics -v`

Expected: all primitive, bin, and window tests pass.

Commit: `git add src/world_model_lab/diagnostics.py tests/test_diagnostics.py && git commit -m "feat: add model diagnostic primitives"`

---

### Task 3: One-step and rollout benchmark assembly

**Files:**
- Modify: `src/world_model_lab/diagnostics.py`
- Modify: `tests/test_diagnostics.py`

**Interfaces:**
- Produces: `predict_next_states(world_model, states, actions)`.
- Produces: `build_diagnostic_metrics(world_model, arrays, split_episode_ids, horizons, windows_per_episode, xy_bins, feature_bins, min_bin_count) -> dict[str, Any]`.

- [x] **Step 1: Write failing teacher-forcing/free-rollout test**

Reuse a constant-delta loaded model. Construct true dynamics whose first step
matches the model and whose later state-dependent deltas diverge. Assert:

```python
metrics = build_diagnostic_metrics(
    world_model,
    arrays=arrays,
    split_episode_ids=split_ids,
    horizons=(1, 2, 3),
    windows_per_episode=2,
    xy_bins=2,
    feature_bins=2,
    min_bin_count=1,
)
h1 = metrics["rollout"]["horizons"]["1"]
h3 = metrics["rollout"]["horizons"]["3"]
self.assertAlmostEqual(
    h1["teacher_forcing"]["position"]["mean"],
    h1["free_rollout"]["position"]["mean"],
)
self.assertGreater(
    h3["free_rollout"]["position"]["mean"],
    h3["teacher_forcing"]["position"]["mean"],
)
```

- [x] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_diagnostics.DiagnosticsTest.test_free_rollout_exposes_compounding_error -v`

Expected: missing `build_diagnostic_metrics`.

- [x] **Step 3: Implement batch next-state prediction**

```python
def predict_next_states(world_model, states, actions) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    inputs = build_model_inputs(states, actions)
    deltas = predict_deltas(world_model, inputs)
    predictions = states + deltas
    predictions[:, 2] = wrap_angle(predictions[:, 2])
    return predictions
```

- [x] **Step 4: Implement episode-balanced rollout summaries**

For every selected window, compute teacher-forced predictions from
`true_states[:-1]`, and compute free predictions with `rollout_episode`. At each
horizon, group scalar errors by `episode_id`, average windows within each group,
then call `summarize_values` over those per-episode means. Return both
`episodes` and `windows` counts alongside the three error summaries.

- [x] **Step 5: Assemble the complete JSON-safe metrics structure**

The top-level mapping must be:

```python
{
    "schema_version": 1,
    "population": {
        "train_episode_ids": [...],
        "test_episode_ids": [...],
        "test_transitions": int_value,
    },
    "one_step": {
        "overall": {
            "position": {...},
            "heading_degrees": {...},
            "velocity": {...},
        },
        "xy_grid": {...},
        "feature_slices": {
            "velocity": {...},
            "steering": {...},
            "acceleration": {...},
        },
    },
    "coverage": {"xy_edges": {...}, "train_counts": [...], "test_counts": [...]},
    "rollout": {
        "protocol": {
            "horizons": [...],
            "max_horizon": int_value,
            "windows_per_episode": int_value,
            "eligible_episode_ids": [...],
            "skipped_episode_ids": [...],
            "windows": [{"episode_id": int_value, "start_step": int_value}],
        },
        "horizons": {...},
    },
}
```

- [x] **Step 6: Add an episode-balancing regression test**

Use one episode with one window and large error and another with multiple
windows and zero error. Assert the reported macro mean gives the two episodes
equal weight rather than weighting all windows equally.

- [x] **Step 7: Verify benchmark core and commit**

Run: `.venv/bin/python -m unittest tests.test_diagnostics -v`

Expected: all tests pass and no JSON tree contains NumPy scalar objects or NaN.

Commit: `git add src/world_model_lab/diagnostics.py tests/test_diagnostics.py && git commit -m "feat: assemble model diagnostics benchmark"`

---

### Task 4: Diagnostic plots

**Files:**
- Create: `src/world_model_lab/diagnostic_plots.py`
- Create: `tests/test_diagnostic_plots.py`

**Interfaces:**
- Produces: `plot_diagnostic_overview(metrics, output_path) -> Path`.
- Produces: `plot_rollout_errors(metrics, output_path) -> Path`.

- [x] **Step 1: Write failing PNG output tests**

Build a minimal JSON-style metrics fixture containing a 2x2 coverage/error grid,
two-bin feature slices, and horizons 1 and 2. Assert both functions return the
requested paths and create files beginning with `b"\x89PNG\r\n\x1a\n"`.

- [x] **Step 2: Run and verify RED**

Run: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m unittest tests.test_diagnostic_plots -v`

Expected: import failure because the plotting module does not exist.

- [x] **Step 3: Implement overview plotting**

Create a 2x3 figure containing train XY counts, test XY counts, test position
MAE, and position-MAE slices for velocity, steering, and acceleration. Convert
JSON `None` cells to masked `np.nan` values only inside the plot function. Each
feature-slice panel uses a secondary Y axis for sample counts.

- [x] **Step 4: Implement rollout plotting**

Create one panel each for position, heading, and velocity. Plot macro mean at
every horizon with separate labelled lines for `Teacher forcing` and `Free
rollout`. Use physical units in axis labels and close the figure after saving.

- [x] **Step 5: Verify plots and commit**

Run: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m unittest tests.test_diagnostic_plots -v`

Expected: both PNG tests pass.

Commit: `git add src/world_model_lab/diagnostic_plots.py tests/test_diagnostic_plots.py && git commit -m "feat: plot model diagnostic reports"`

---

### Task 5: Reproducible output bundle and CLI

**Files:**
- Create: `src/world_model_lab/diagnose_model.py`
- Create: `tests/test_diagnose_model.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `sha256_file(path) -> str`.
- Produces: `run_diagnostics(...) -> dict[str, Any]`.
- Produces: module CLI and `world-model-diagnose` console command.

- [x] **Step 1: Write failing SHA-256 test**

```python
def test_sha256_file_is_content_based(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.bin"
        path.write_bytes(b"world-model")
        digest = sha256_file(path)
    self.assertEqual(
        digest,
        "f6fafe16f3d1018f98141e97f8c60ad98233b537e62e6d2383a53704d28016df",
    )
```

- [x] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_diagnose_model -v`

Expected: import failure because `diagnose_model` does not exist.

- [x] **Step 3: Implement hashing and strict JSON writing**

```python
def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
```

- [x] **Step 4: Write failing output-bundle smoke test**

Create a temporary NPZ with train/test episodes and save a constant-delta
checkpoint. Call `run_diagnostics` with horizons `(1, 2)`, then assert that
`metrics.json`, `manifest.json`, `overview.png`, and `rollout_errors.png` exist.
Load both JSON files and assert hashes, resolved paths, training config, test
episode IDs, and diagnostic parameters are present.

- [x] **Step 5: Implement orchestration and manifest**

`run_diagnostics` must validate arrays `states`, `actions`, `next_states`,
`episode_ids`, and `step_ids`; load the checkpoint; call
`build_diagnostic_metrics`; write both JSON documents; create both figures; and
return a short summary containing output paths plus overall and longest-horizon
means.

The manifest structure is:

```python
{
    "schema_version": 1,
    "dataset": {"path": str(data_path.resolve()), "sha256": sha256_file(data_path)},
    "checkpoint": {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "hidden_size": world_model.model.hidden_size,
        "training_config": world_model.training_config,
        "test_episode_ids": world_model.split_episode_ids["test"].tolist(),
    },
    "diagnostics": {
        "horizons": list(horizons),
        "windows_per_episode": windows_per_episode,
        "xy_bins": xy_bins,
        "feature_bins": feature_bins,
        "min_bin_count": min_bin_count,
    },
}
```

- [x] **Step 6: Add CLI arguments and console entry point**

Add `main()` with defaults from the design and catch `FileNotFoundError` plus
`ValueError` with `parser.error`. Add to `pyproject.toml`:

```toml
world-model-diagnose = "world_model_lab.diagnose_model:main"
```

- [x] **Step 7: Verify CLI/bundle tests and commit**

Run: `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m unittest tests.test_diagnose_model -v`

Expected: hashing, bundle, JSON, and validation tests pass.

Commit: `git add src/world_model_lab/diagnose_model.py tests/test_diagnose_model.py pyproject.toml && git commit -m "feat: add reproducible diagnostics command"`

---

### Task 6: Documentation and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-13-model-diagnostics-lab.md`

**Interfaces:**
- Documents: command, output bundle, metric semantics, and interpretation cautions.

- [x] **Step 1: Document the exact command and outputs**

Add a `模型诊断实验室` section to `README.md` containing the CLI from the design,
the four output files, the fixed-window/episode-balanced protocol, and the
meaning of teacher forcing versus free rollout. State that sparse bins are
masked and that error metrics use only checkpoint test episodes.

- [x] **Step 2: Run the full test suite**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python -m unittest discover -s tests -v
```

Expected: the original 35 tests plus all new diagnostics tests pass with zero
failures and zero errors.

- [x] **Step 3: Run the real-data benchmark smoke test**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python -m world_model_lab.diagnose_model \
  --data data/transitions.npz \
  --checkpoint artifacts/world_model.pt \
  --output-dir artifacts/diagnostics/baseline \
  --horizons 1 5 10 20 50 \
  --windows-per-episode 8 \
  --xy-bins 12 \
  --feature-bins 8 \
  --min-bin-count 5
```

Expected: exit code zero, finite overall/longest-horizon summary values, and all
four output files.

- [x] **Step 4: Validate generated artifacts**

Run:

```bash
.venv/bin/python -m json.tool artifacts/diagnostics/baseline/metrics.json >/dev/null
.venv/bin/python -m json.tool artifacts/diagnostics/baseline/manifest.json >/dev/null
file artifacts/diagnostics/baseline/overview.png artifacts/diagnostics/baseline/rollout_errors.png
git diff --check
```

Expected: both JSON files parse, both images are PNG files, and `git diff
--check` exits zero.

- [x] **Step 5: Mark this plan complete and commit documentation**

Change every completed checkbox in this plan to `[x]` only after its command has
been run successfully.

Commit: `git add README.md docs/superpowers/plans/2026-07-13-model-diagnostics-lab.md && git commit -m "docs: explain model diagnostics workflow"`
