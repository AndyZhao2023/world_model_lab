# Visual Object-Aligned Autoencoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one spatial autoencoder whose representation and decoder
preserve the rendered car, gate it on held-out object and oracle-rollout
metrics, and only then retrain H5 latent dynamics.

**Architecture:** Derive an exact binary object mask from the renderer's
`CAR_COLOR` and `HEADING_COLOR` pixels. Keep the existing `[8, 8, 8]` spatial
autoencoder and optimize separately normalized object and background
reconstruction MSE. Compare source and candidate autoencoders with the same
test episodes and rollout frame indices; if the representation gates pass,
freeze the candidate autoencoder and run the existing H5 recursive dynamics
protocol.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, unittest.

## Global Constraints

- Use `data/visual_episodes.npz` without modification.
- Use `artifacts/visual_latent_spatial8_objective_w01.pt` as the representation
  baseline.
- Preserve spatial latent shape `[B, 8, 8, 8]`, base channels `16`, batch size
  `128`, learning rate `1e-3`, epoch budget `20`, seed `0`, and split seed
  `42`.
- Train exactly one object-aligned autoencoder candidate with object loss
  weight `1.0`; do not sweep weights, seeds, capacities, or epoch budgets.
- Define the object mask only from exact `CAR_COLOR` or `HEADING_COLOR`
  renderer pixels. Do not use test-set states or learned segmentation.
- Define positive-weight reconstruction loss as:
  `background MSE + object_loss_weight * object MSE`, with RGB channels
  included in each region's own denominator.
- Preserve the exact ordinary full-frame MSE path when both object and motion
  weights are zero.
- Reject simultaneous positive object and motion reconstruction weights.
- Keep held-out test episodes out of checkpoint selection.
- Refuse to overwrite checkpoints, previews, or non-empty diagnostic paths.
- Do not run H5 dynamics unless every representation gate passes.
- Do not compare raw latent MSE across different autoencoders as a promotion
  metric because the latent coordinate systems differ.
- Do not connect any candidate to MPC in this experiment.

## Locked Baseline and Gates

The source autoencoder's held-out metrics are:

```text
full-frame MSE       0.0015107901
object-region MSE    0.1580440998
object-region MAE    0.3283802569
background MSE       0.0006003271
object pixel fraction 0.0057880387
```

The candidate representation passes only if all gates pass:

```text
held-out object-region MSE < 0.1580440998
held-out full-frame MSE <= 0.0016618691
held-out background MSE <= 0.0006603598
H5 oracle cumulative changed-pixel MAE strictly improves
    over the source value locked by the pre-training diagnostic
```

The H5 gate uses deterministic factual windows selected from the unchanged
test split with horizons `1, 5, 10` and eight windows per eligible episode.
Both models reconstruct the same true target frames through their own
encoder/decoder; the metric is directly comparable in pixel space.

## File Structure

- Modify `src/world_model_lab/visual_latent_data.py`: renderer-object mask
  extraction and object-frame dataset.
- Modify `src/world_model_lab/train_visual_latent_model.py`: object-balanced
  reconstruction objective, region metrics, runner setting, CLI, and
  checkpoint metadata.
- Create `src/world_model_lab/diagnose_visual_autoencoder.py`: matched
  cross-representation oracle reconstruction diagnostic and gates.
- Modify `pyproject.toml`: register the autoencoder diagnostic command.
- Modify `tests/test_visual_latent_data.py`: object-mask adapter tests.
- Modify `tests/test_train_visual_latent_model.py`: loss, metrics, runner, and
  CLI tests.
- Create `tests/test_diagnose_visual_autoencoder.py`: aggregation, compatibility,
  publication, and gate tests.
- Create
  `docs/experiments/2026-07-19-visual-object-aligned-autoencoder.md`: fixed
  protocol, results, artifacts, and decision.

---

### Task 1: Add exact renderer-object masks

**Files:**
- Modify: `src/world_model_lab/visual_latent_data.py`
- Modify: `tests/test_visual_latent_data.py`

**Interfaces:**
- Produces:
  `renderer_object_masks(frames) -> torch.Tensor`.
- Produces:
  `VisualObjectFrameDataset[index] -> (image, object_mask)`.

- [x] **Step 1: Write failing object-mask adapter tests**

Inject exact car and heading colours at known locations. Require a binary
`float32 [1, 64, 64]` mask, Python indexing behavior, selected-episode order,
and owned tensors that cannot mutate the source arrays.

- [x] **Step 2: Run the focused test and verify the import failure**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_data
```

- [x] **Step 3: Implement exact colour extraction and the dataset**

For each HWC frame:

```python
car = np.all(frame == np.asarray(CAR_COLOR), axis=2)
heading = np.all(frame == np.asarray(HEADING_COLOR), axis=2)
mask = car | heading
```

Validate uint8 frame shape and return an owned tensor.

- [x] **Step 4: Run the focused data-adapter tests**

Expected: all visual latent data tests pass.

---

### Task 2: Add the object-balanced reconstruction objective

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Produces:
  `_object_balanced_mse(reconstructions, images, object_masks, *,
  object_loss_weight)`.
- Adds `object_loss_weight=0.0` to autoencoder training and the end-to-end
  runner.
- Adds object/background counts, MSE, and MAE to autoencoder evaluation.

- [x] **Step 1: Write failing numeric and validation tests**

Use one object pixel with RGB error `1.0` and one background pixel with RGB
error `0.5`. Require weight `1.0` to produce `1.25`, while weight `0.0`
exactly equals ordinary full-frame MSE. Reject malformed/non-binary masks,
empty positive-weight regions, non-finite weights, and simultaneous positive
object/motion weights.

- [x] **Step 2: Run the focused tests and verify missing behavior**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_visual_latent_model.VisualAutoencoderTrainingTest
```

- [x] **Step 3: Implement training and validation loss selection**

Use `VisualObjectFrameDataset` only when object weight is positive. Keep the
legacy motion dataset and `_motion_weighted_mse` path otherwise. Select the
best epoch with the same configured objective used for training.

- [x] **Step 4: Add held-out region metrics**

Evaluate exact object and background masks and report spatial pixel counts plus
RGB-normalized MSE/MAE. Return finite zeros for an absent region only in
generic evaluation fixtures; positive-weight training must reject absent
regions.

- [x] **Step 5: Thread the setting through runner, checkpoint, and CLI**

Add `--object-loss-weight`, record it in `training_config`, include the new
metrics in the JSON summary, and preserve old defaults.

- [x] **Step 6: Run focused training tests**

Expected: all visual latent model tests pass.

---

### Task 3: Add cross-representation oracle diagnostics

**Files:**
- Create: `src/world_model_lab/diagnose_visual_autoencoder.py`
- Create: `tests/test_diagnose_visual_autoencoder.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes two spatial autoencoder checkpoints with the same dataset digest
  and test split, but permits different autoencoder weights and normalizers.
- Produces per-step full, object, background, and cumulative changed-pixel
  reconstruction metrics with episode-equal aggregation.
- Produces a four-gate representation decision.

- [x] **Step 1: Write failing aggregation and decision tests**

Require windows to be averaged within episode and episodes equally. Require
strict object/H5 improvements and 10% full/background stability. Equality
must fail strict gates.

- [x] **Step 2: Write failing runner/CLI tests**

Create two compatible checkpoints with different autoencoder tensors. Require
a complete atomic bundle (`manifest.json`, `metrics.json`, plot), and reject
dataset-digest or test-split mismatches.

- [x] **Step 3: Implement matched oracle reconstruction**

Select frame indices once from the source test split. Independently encode and
decode all frames for each autoencoder. Compare reconstructed target frames
with the same true target pixels and exact object masks.

- [x] **Step 4: Implement gates and atomic publication**

Use absent-or-empty output semantics, a sibling staging directory, strict JSON,
and cleanup on failure. Register:

```text
world-model-diagnose-visual-autoencoder
```

- [x] **Step 5: Run focused diagnostic tests**

Expected: all autoencoder diagnostic tests pass.

---

### Task 4: Lock the source rollout baseline before training

**Files:**
- Modify:
  `docs/experiments/2026-07-19-visual-object-aligned-autoencoder.md`
- Generated but not committed:
  `/private/tmp/visual-object-autoencoder-source-baseline/`

- [x] **Step 1: Run the full suite before any candidate training**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests
```

- [x] **Step 2: Run source versus source diagnostics**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_autoencoder \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --candidate-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --output-dir /private/tmp/visual-object-autoencoder-source-baseline \
  --horizons 1 5 10 \
  --windows-per-episode 8
```

- [x] **Step 3: Record the exact H5 source value and bundle digests**

Do this before creating the candidate checkpoint. The source-versus-source
decision is expected to fail both strict-improvement gates.

---

### Task 5: Train the single representation candidate and decide

**Files:**
- Modify:
  `docs/experiments/2026-07-19-visual-object-aligned-autoencoder.md`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_aligned_w1.pt`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_aligned_w1_predictions.png`
- Generated but not committed:
  `artifacts/diagnostics/visual-object-autoencoder-w1/`

- [x] **Step 1: Train the registered candidate once**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_latent_model \
  --data data/visual_episodes.npz \
  --output artifacts/visual_latent_spatial8_object_aligned_w1.pt \
  --preview artifacts/visual_latent_spatial8_object_aligned_w1_predictions.png \
  --latent-layout spatial \
  --spatial-latent-channels 8 \
  --spatial-dynamics-architecture cnn \
  --base-channels 16 \
  --dynamics-hidden-size 64 \
  --autoencoder-epochs 20 \
  --dynamics-epochs 50 \
  --autoencoder-batch-size 128 \
  --dynamics-batch-size 256 \
  --autoencoder-learning-rate 0.001 \
  --dynamics-learning-rate 0.001 \
  --object-loss-weight 1.0 \
  --seed 0 \
  --split-seed 42
```

- [x] **Step 2: Run the registered representation diagnostic**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_autoencoder \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --candidate-checkpoint artifacts/visual_latent_spatial8_object_aligned_w1.pt \
  --output-dir artifacts/diagnostics/visual-object-autoencoder-w1 \
  --horizons 1 5 10 \
  --windows-per-episode 8
```

- [x] **Step 3: Apply all four representation gates**

If any gate fails, stop before recursive dynamics and retain the source
representation. Record the negative result without changing thresholds.

---

### Task 6: Conditionally train and evaluate H5 dynamics

**Files:**
- Modify:
  `docs/experiments/2026-07-19-visual-object-aligned-autoencoder.md`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_aligned_w1_h5.pt`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_aligned_w1_h5_predictions.png`

- [x] **Step 1: Do not train H5 because representation promotion failed**

The candidate passed only the object-MSE gate. Per the registered rule, no H5
checkpoint was created.

- [x] **Step 2: Keep cross-representation dynamics comparison out of scope**

Because the representation failed before dynamics, a simulator
counterfactual comparison would not affect the decision and was not run.

- [x] **Step 3: Record the final promotion decision**

Promotion requires improved H5 cumulative changed-pixel MAE and action-effect
quality without violating H1 stability. Otherwise retain the current H1
default and prohibit MPC integration.

---

### Task 7: Verify, document, and commit

**Files:**
- Modify all files listed above.

- [x] **Step 1: Run fresh final verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests
PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests
git diff --check
```

- [x] **Step 2: Review repository boundaries**

Require only source, tests, plan, experiment document, and `pyproject.toml`
changes to be staged. Leave data, checkpoints, previews, diagnostics, and the
pre-existing `.DS_Store` untracked or ignored.

- [x] **Step 3: Commit the controlled experiment**

Commit implementation and recorded result together. State whether the run
stopped at the representation gate or proceeded to H5.
