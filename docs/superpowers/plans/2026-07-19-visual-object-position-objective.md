# Visual Object-Position Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a fresh H5 spatial latent dynamics candidate with a frozen,
state-supervised linear car-position probe, then evaluate whether the new
objective closes the matched cumulative moving-pixel gap without losing the
existing latent and action-effect gains.

**Architecture:** Fit one deterministic ridge probe from normalized frozen
autoencoder latents to normalized car centre `(x, y)` using training frames
only. Freeze the probe and add its position MSE to both the one-step and every
recursive H5 objective. Extend the matched simulator diagnostic with direct
and action-effect position errors, while preserving the existing branch,
masking, and episode-equal aggregation protocol.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, unittest.

## Global Constraints

- Use `artifacts/visual_latent_spatial8_objective_w01.pt` as the source
  checkpoint for autoencoder, normalizers, split IDs, configuration, and fresh
  dynamics initialization.
- Keep the visual dataset and its SHA-256 unchanged.
- Keep the source spatial autoencoder exactly frozen and byte-equal after
  checkpoint reload.
- Keep latent/action normalizers and train/validation/test episode IDs exactly
  unchanged.
- Keep spatial CNN hidden channels, learning rate `1e-3`, batch size `256`,
  epoch budget `50`, and seed `0` unchanged.
- Keep the one-step and H5 base objectives unchanged:
  `normalized latent MSE + 0.1 * decoded changed-pixel MAE`.
- Keep the H5 recursive objective weight at `1.0`.
- Fit the position probe on source-normalized latents from training frames only.
- Normalize world `x` and `y` independently to `[-1, 1]` using the fixed scene
  bounds.
- Use ridge coefficient `1e-3`; do not tune it against candidate results.
- Add normalized position MSE with weight `1.0` to one-step and every H5 step.
- Do not train the probe jointly with dynamics.
- Do not add heading supervision because its validation probe error is not
  reliable enough for this controlled experiment.
- Do not sweep position weights, seeds, horizons, or rollout weights inside
  this run.
- Preserve the ten-seed matched simulator counterfactual protocol.
- Do not connect any checkpoint to MPC inside this experiment.
- Refuse to overwrite checkpoint, preview, or non-empty diagnostic paths.
- Require the new candidate's H5 matched normalized position MSE to be
  strictly below the old H5 candidate's locked `0.0969399159`, in addition to
  the H1-relative promotion gates.

## File Structure

- Create `src/world_model_lab/visual_object_position.py`: position
  normalization, deterministic ridge fitting, frozen torch probe, and probe
  metadata.
- Modify `src/world_model_lab/visual_recursive_training.py`: append position
  targets to one-step/H5 batches and include the frozen-probe loss.
- Modify `src/world_model_lab/train_visual_dynamics_recursive.py`: fit the
  training-only probe, expose the object loss settings, and record the
  controlled protocol in the checkpoint.
- Modify `src/world_model_lab/diagnose_visual_counterfactual.py`: fit the same
  probe, report matched direct/effect position metrics, and add the
  pre-registered position gate.
- Create `tests/test_visual_object_position.py`: unit tests for normalization,
  ridge fitting, frozen gradients, and validation.
- Modify `tests/test_visual_recursive_training.py`: objective and gradient
  tests for one-step/H5 position supervision.
- Modify `tests/test_train_visual_dynamics_recursive.py`: runner/CLI/checkpoint
  contract tests.
- Modify `tests/test_diagnose_visual_counterfactual.py`: masking,
  episode-macro, aggregation, and decision-gate tests for position metrics.
- Create
  `docs/experiments/2026-07-19-visual-object-position-objective.md`: fixed
  protocol, results, artifacts, and decision.

---

### Task 1: Add the frozen car-position probe

**Files:**
- Create: `src/world_model_lab/visual_object_position.py`
- Create: `tests/test_visual_object_position.py`

**Interfaces:**
- Consumes: schema-v1 scene bounds, `[N, D]` normalized latents, `[N, 4]`
  physical states, and a non-negative ridge coefficient.
- Produces:
  `normalize_object_positions(states, world_bounds) -> np.ndarray`.
- Produces:
  `fit_linear_object_position_probe(normalized_latents, normalized_positions,
  *, ridge) -> LinearObjectPositionProbe`.
- Produces:
  `LinearObjectPositionProbe.forward(normalized_latents) -> torch.Tensor`.
- Produces:
  `world_position_errors(predicted_normalized, true_normalized, world_bounds)
  -> np.ndarray`.

- [x] **Step 1: Write failing normalization and probe tests**

Use world bounds `(0, 10, 0, 8)` and require:

```python
positions = normalize_object_positions(
    np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [5.0, 4.0, 0.0, 0.0],
         [10.0, 8.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
    np.asarray([0.0, 10.0, 0.0, 8.0], dtype=np.float64),
)
np.testing.assert_array_equal(
    positions,
    [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
)
```

Fit a noiseless affine dataset, require predictions within `1e-5`, and assert
every probe parameter has `requires_grad=False` while gradients still reach
the input latents.

- [x] **Step 2: Run the focused test and verify the missing module**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_object_position
```

Expected: import failure for `world_model_lab.visual_object_position`.

- [x] **Step 3: Implement deterministic ridge fitting**

Fit an affine map in float64:

```python
design = np.concatenate(
    (normalized_latents.astype(np.float64),
     np.ones((normalized_latents.shape[0], 1))),
    axis=1,
)
regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
regularizer[-1, -1] = 0.0
solution = np.linalg.solve(
    design.T @ design + regularizer,
    design.T @ normalized_positions,
)
```

Store the first `D` rows as a float32 `[2, D]` torch weight and the final row
as a float32 `[2]` bias. Reject empty, non-finite, rank-invalid, or
dimension-mismatched inputs and negative/non-finite ridge values.

- [x] **Step 4: Run the focused tests**

Expected: all position-probe tests pass.

---

### Task 2: Add position supervision to one-step and H5 objectives

**Files:**
- Modify: `src/world_model_lab/visual_recursive_training.py`
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_visual_recursive_training.py`

**Interfaces:**
- Consumes: frozen `LinearObjectPositionProbe`, one-step `[B, 2]` and rollout
  `[B, H, 2]` normalized position targets, and position loss weight.
- Produces: the existing objective tensors plus the weighted position term.
- Preserves all existing call sites when position weight is zero.

- [x] **Step 1: Write failing objective tests**

For a probe that selects the first two latent dimensions, require:

```text
base latent/image objective = 0
predicted position error MSE = 0.25
position weight = 2
total objective = 0.5
```

For H2 recursion, require the arithmetic mean of both step position losses.
Backpropagate and assert a dynamics parameter receives a finite non-zero
gradient while probe parameters remain frozen.

- [x] **Step 2: Run focused tests and verify signature failures**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_recursive_training
```

Expected: failures for unsupported position probe/target arguments.

- [x] **Step 3: Extend the one-step objective**

Extend `_dynamics_batch_loss` with optional keyword-only arguments:

```python
position_probe: LinearObjectPositionProbe | None = None
target_positions: torch.Tensor | None = None
object_position_loss_weight: float = 0.0
```

After computing the existing base objective:

```python
predicted_positions = position_probe(prediction)
position_mse = torch.mean(
    torch.square(predicted_positions - target_positions)
)
loss = loss + object_position_loss_weight * position_mse
```

Require complete probe/target supervision for positive weight, exact
`[B, 2]` targets, finite values, and a finite non-negative weight.

- [x] **Step 4: Extend the recursive objective**

Add the same optional probe and `[B, H, 2]` target to
`recursive_rollout_objective`. At each step, evaluate the probe on that
step's self-fed predicted latent and add the weighted position MSE before
averaging H losses.

- [x] **Step 5: Append aligned state targets to training tensors**

For one-step arrays:

```python
states[arrays.target_frame_indices]
```

For rollout arrays:

```python
states[arrays.target_frame_indices]
```

Normalize these positions from scene bounds. Keep frame indices as the only
join key, so targets stay aligned and episode-safe.

- [x] **Step 6: Thread the probe through deterministic training**

Append position tensors only when the weight is positive. Pass the probe and
position targets into training and validation objectives. Keep model
initialization, batch permutations, optimizer, best-epoch selection, and
decoder freezing unchanged.

- [x] **Step 7: Run focused recursive tests**

Expected: existing recursion tests and all new position tests pass.

---

### Task 3: Extend the controlled training runner

**Files:**
- Modify: `src/world_model_lab/train_visual_dynamics_recursive.py`
- Modify: `tests/test_train_visual_dynamics_recursive.py`

**Interfaces:**
- Adds runner/CLI settings:
  `object_position_loss_weight=0.0` and
  `object_position_probe_ridge=1e-3`.
- Produces candidate:
  `artifacts/visual_latent_spatial8_objective_w01_h5_position_w1.pt`.

- [x] **Step 1: Write failing CLI and checkpoint-contract tests**

Require CLI help to contain:

```text
--object-position-loss-weight
--object-position-probe-ridge
```

Run a one-epoch fixture with positive weight and require checkpoint config:

```python
self.assertEqual(
    config["dynamics_loss"],
    "one_step_plus_recursive_rollout_plus_object_position",
)
self.assertEqual(config["dynamics_object_position_loss_weight"], 1.0)
self.assertEqual(config["object_position_probe_ridge"], 1e-3)
self.assertEqual(config["object_position_probe_fit_split"], "train_frames")
self.assertEqual(config["object_position_target"], "normalized_xy")
```

Continue requiring exact autoencoder, normalizer, and split equality.

- [x] **Step 2: Run runner tests and verify missing settings**

Expected: CLI/config assertions fail.

- [x] **Step 3: Fit the probe from training frames only**

Encode all frames once as before. Select training frame indices with
`frame_indices_for_episode_ids`, normalize their source latents, normalize
their state positions, and fit ridge `1e-3`. Measure train and validation
probe normalized MSE and mean world-position error for metadata.

- [x] **Step 4: Record the full protocol**

Add checkpoint config for probe ridge, fit split, target definition, probe
weight/bias SHA-256, fit metrics, and position loss weight. Return the same
metadata in the JSON summary. Reject negative/non-finite settings before
loading files.

- [x] **Step 5: Run runner tests**

Expected: all recursive runner tests pass.

---

### Task 4: Add matched object-position diagnostics

**Files:**
- Modify: `src/world_model_lab/diagnose_visual_counterfactual.py`
- Modify: `tests/test_diagnose_visual_counterfactual.py`

**Interfaces:**
- Adds metrics:
  `normalized_position_mse`,
  `world_position_error`, and
  `normalized_position_effect_mse`.
- Fits the same probe from training frames only.
- Adds one H5 direct-position improvement gate.

- [x] **Step 1: Write failing masked metric tests**

Pass predicted factual/counterfactual normalized positions and true physical
positions into the seed summarizer. Construct three windows across two
episodes with one masked second step. Require masked windows to be ignored and
the remaining windows to be averaged within episode before equal episode
averaging.

- [x] **Step 2: Write the failing decision test**

Require the decision to include six gates, with the additional gate:

```text
horizon_direct_position_improvement
metric = normalized_position_mse
operator = <
```

Equality must fail the new strict-improvement gate.

- [x] **Step 3: Compute direct and action-effect position metrics**

Apply the frozen probe to predicted normalized rollout latents. Compare direct
positions with simulator states normalized from world bounds. Compare:

```text
predicted counterfactual position - predicted factual position
```

against:

```text
true counterfactual position - true factual position
```

Convert direct normalized coordinate errors back to world-axis units and
report Euclidean centre error.

- [x] **Step 4: Preserve aggregation and publication boundaries**

Aggregate every new metric with the existing valid-step mask, recipient
episode macro average, and seed mean/sample standard deviation. Record the
probe protocol and fit metrics in the manifest. Keep terminal handling and
atomic bundle publication unchanged.

- [x] **Step 5: Update the comparison plot**

Replace the pixel action-effect panel with matched world-position error. Keep
latent, cumulative changed-pixel, and latent action-effect panels.

- [x] **Step 6: Run focused diagnostic tests**

Expected: all matched counterfactual tests pass.

---

### Task 5: Establish baseline, train once, and decide

**Files:**
- Modify:
  `docs/experiments/2026-07-19-visual-object-position-objective.md`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_objective_w01_h5_position_w1.pt`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_objective_w01_h5_position_w1_predictions.png`
- Generated but not committed:
  `artifacts/diagnostics/visual-matched-counterfactual-h5-position-comparison/`

**Interfaces:**
- Consumes the fixed source checkpoint, data, and pre-registered settings.
- Produces one candidate and one matched diagnostic bundle.

- [x] **Step 1: Run the full test suite before training**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests
```

Expected: all tests pass.

- [x] **Step 2: Record source object-position baselines before training**

Run the matched diagnostic on the existing H1 source versus existing H5
candidate into a fresh temporary directory. Record H1/H5/H10 position metrics
and the probe validation floor in the experiment document before creating the
new candidate.

- [x] **Step 3: Train the single registered candidate**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_dynamics_recursive \
  --data data/visual_episodes.npz \
  --source-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01.pt \
  --output \
    artifacts/visual_latent_spatial8_objective_w01_h5_position_w1.pt \
  --preview \
    artifacts/visual_latent_spatial8_objective_w01_h5_position_w1_predictions.png \
  --changed-pixel-loss-weight 0.1 \
  --rollout-horizon 5 \
  --rollout-loss-weight 1.0 \
  --object-position-loss-weight 1.0 \
  --object-position-probe-ridge 0.001 \
  --dynamics-epochs 50 \
  --dynamics-batch-size 256
```

Expected: one new checkpoint and preview, with no modification to source/H5
artifacts.

- [x] **Step 4: Run the registered matched diagnostic**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_counterfactual \
  --data data/visual_episodes.npz \
  --source-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01.pt \
  --candidate-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01_h5_position_w1.pt \
  --output-dir \
    artifacts/diagnostics/visual-matched-counterfactual-h5-position-comparison \
  --horizons 1 5 10 \
  --windows-per-episode 8 \
  --counterfactual-seeds 0 1 2 3 4 5 6 7 8 9 \
  --decision-horizon 5
```

Expected: manifest, metrics JSON, and plot with finite values.

- [x] **Step 5: Independently reproduce the diagnostic bundle**

Run the same diagnostic into a new `/private/tmp` directory and require
byte-identical manifest, metrics, and PNG hashes after accounting for the
different output directory only if it is stored in an artifact. The current
protocol does not store the output path, so all three files must be
byte-identical.

- [x] **Step 6: Run final verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests
PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests
git diff --check
```

Expected: tests pass, compilation passes, and Git reports no whitespace
errors.

- [x] **Step 7: Record the strict decision and commit**

Document every gate as pass/fail. Promote only if all gates pass; otherwise
retain the H1 source as default. Commit source, tests, plan, and experiment
document while leaving data, checkpoints, previews, and diagnostics
untracked/ignored.
