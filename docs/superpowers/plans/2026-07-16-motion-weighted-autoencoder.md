# Motion-Weighted Autoencoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible motion-weighted reconstruction objective and compare weights `100` and `500` against the existing 20-epoch visual autoencoder baseline.

**Architecture:** Keep `ConvAutoencoder`, `LatentDynamicsMLP`, episode splits, latent size, and all evaluation metrics unchanged. Add a frame adapter that derives a binary motion mask from each frame and its previous frame in the same episode, then use `1 + motion_loss_weight * mask` inside the autoencoder MSE. Initial episode frames compare with themselves and therefore receive an all-zero motion mask.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, unittest, Matplotlib.

## Global Constraints

- Do not use `states`, rewards, dones, or hand-authored car labels.
- Do not cross episode boundaries when selecting the previous frame.
- `motion_loss_weight=0.0` must preserve the original plain pixel-MSE behavior.
- Reject negative or non-finite motion weights before training.
- Fit and evaluate on the existing episode split with `split_seed=42`.
- Run real experiments with `latent_dim=32`, `base_channels=16`, dynamics hidden size `256`, autoencoder `20` epochs, dynamics `50` epochs, and training seed `0`.
- Publish new artifacts without overwriting prior checkpoints or previews.

---

### Task 1: Motion-aware frame adapter

**Files:**
- Modify: `src/world_model_lab/visual_latent_data.py`
- Test: `tests/test_visual_latent_data.py`

**Interfaces:**
- Consumes: visual dataset arrays `frames`, `episode_ids`, and `frame_offsets`.
- Produces: `VisualMotionFrameDataset(dataset, selected_episode_ids)`, whose items are `(image, motion_mask)` tensors with shapes `[3,64,64]` and `[1,64,64]`.

- [ ] **Step 1: Write a failing episode-boundary test**

Create synthetic changes in the first and second frames of an episode. Assert:

```python
dataset = VisualMotionFrameDataset(visual, np.asarray([11, 10]))
first_image, first_mask = dataset[0]
second_image, second_mask = dataset[1]
assert torch.count_nonzero(first_mask) == 0
assert second_mask[0, 2, 3] == 1
```

Also assert that the first frame of episode `10` compares with itself rather
than with the final frame of episode `11`.

- [ ] **Step 2: Run the focused test and confirm import failure**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_data.VisualFrameAdapterTest -v
```

Expected: failure because `VisualMotionFrameDataset` does not exist.

- [ ] **Step 3: Implement aligned previous-frame indices**

For each selected episode with frame slice `[start, stop)` produce:

```python
current = np.arange(start, stop, dtype=np.int64)
previous = np.concatenate(
    (
        np.asarray([start], dtype=np.int64),
        np.arange(start, stop - 1, dtype=np.int64),
    )
)
```

Store both arrays read-only. Reuse the same strict scalar-index behavior as
`VisualFrameDataset`.

- [ ] **Step 4: Build the binary spatial mask**

Inside `__getitem__`, convert the current frame with `frames_to_tensor` and
derive:

```python
changed = np.any(current_frame != previous_frame, axis=2)
motion_mask = torch.from_numpy(changed.copy()).unsqueeze(0).float()
```

- [ ] **Step 5: Run adapter tests**

Expected: all `tests.test_visual_latent_data` tests pass.

---

### Task 2: Weighted reconstruction loss

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Test: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Consumes: reconstructed images, target images, binary masks, and a finite non-negative float.
- Produces: `_motion_weighted_mse(...) -> torch.Tensor` and a new `motion_loss_weight` parameter on `train_autoencoder`.

- [ ] **Step 1: Write failing numeric loss tests**

Use one two-pixel example:

```python
target = torch.zeros(1, 3, 1, 2)
prediction = torch.tensor([[[[1.0, 0.5]], ...]])
mask = torch.tensor([[[[1.0, 0.0]]]])
```

Assert:

```python
_motion_weighted_mse(prediction, target, mask, 0.0)
    == torch.mean((prediction - target) ** 2)
```

For weight `3`, assert the numerator weights the first spatial pixel by `4`
and the denominator is `3 * (4 + 1)`.

- [ ] **Step 2: Run the numeric tests and confirm import failure**

Expected: failure because `_motion_weighted_mse` does not exist.

- [ ] **Step 3: Implement validation and exact zero-weight compatibility**

Implement:

```python
if not math.isfinite(motion_loss_weight) or motion_loss_weight < 0:
    raise ValueError(...)
if motion_loss_weight == 0:
    return torch.mean(torch.square(reconstructions - images))
weights = 1.0 + motion_loss_weight * motion_masks
return torch.sum(torch.square(reconstructions - images) * weights) / (
    torch.sum(weights) * images.shape[1]
)
```

Validate image and mask shapes before calculating.

- [ ] **Step 4: Train and validate with the same weighted objective**

Replace `VisualFrameDataset` with `VisualMotionFrameDataset` in
`train_autoencoder`. Pass `motion_loss_weight` through both the training loop
and `_mean_autoencoder_loss`, so best-epoch selection matches the training
objective. Keep `evaluate_autoencoder` as ordinary unweighted pixel metrics.

- [ ] **Step 5: Extend invalid-parameter tests**

Assert `train_autoencoder` rejects `-1.0`, `NaN`, and infinity.

- [ ] **Step 6: Run training tests**

Expected: all visual latent training tests pass.

---

### Task 3: CLI, checkpoint metadata, and documentation

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `README.md`
- Test: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Consumes: `--motion-loss-weight FLOAT`.
- Produces: checkpoint `training_config["motion_loss_weight"]` and summary configuration traceability.

- [ ] **Step 1: Write failing CLI and metadata assertions**

Assert `--help` includes `--motion-loss-weight`. In the tiny end-to-end test,
run with a nonzero weight and assert the loaded checkpoint records it.

- [ ] **Step 2: Add the CLI option**

Add:

```python
parser.add_argument("--motion-loss-weight", type=float, default=0.0)
```

Thread the value through `main`, `run_visual_latent_training`, and
`train_autoencoder`, and record it in `training_config`.

- [ ] **Step 3: Document semantics**

Document:

```text
weight 0 = ordinary pixel MSE
weight > 0 = pixels changed from the preceding frame receive 1 + weight
```

State that masks are image-derived and episode-local.

- [ ] **Step 4: Run CLI and README tests**

Expected: all training-module tests pass.

---

### Task 4: Real controlled experiments

**Files:**
- Create: `docs/experiments/2026-07-16-motion-weighted-autoencoder.md`
- Generate ignored artifacts under: `artifacts/`

**Interfaces:**
- Consumes: the committed visual dataset and the exact baseline protocol.
- Produces: two checkpoints, two six-column previews, and a comparison table.

- [ ] **Step 1: Run weight 100**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_latent_model \
  --data data/visual_episodes.npz \
  --output artifacts/visual_latent_motion100.pt \
  --preview artifacts/visual_latent_motion100_predictions.png \
  --latent-dim 32 --base-channels 16 --dynamics-hidden-size 256 \
  --autoencoder-epochs 20 --dynamics-epochs 50 \
  --motion-loss-weight 100 \
  --seed 0 --split-seed 42
```

- [ ] **Step 2: Run weight 500**

Repeat with `motion-loss-weight=500` and `motion500` output names.

- [ ] **Step 3: Compare against fixed gates**

Record Oracle and world-model full-frame MSE and changed-pixel MAE. A candidate
passes only when:

```text
Oracle changed-pixel MAE < 0.271
Oracle full-frame MSE <= 0.00279
car remains visually identifiable in Oracle preview
world-model changed-pixel MAE improves over 0.338870
```

- [ ] **Step 4: Record the decision**

If either candidate passes, select the lower-weight passing candidate. If both
fail, do not tune more weights; recommend a spatial latent as the next model
change.

---

### Task 5: Full verification and commit

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: complete implementation and experiment evidence.
- Produces: one reviewed commit.

- [ ] **Step 1: Run complete verification**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests

git diff --check
```

Expected: compilation succeeds, all tests pass, and no whitespace errors.

- [ ] **Step 2: Review scope**

Confirm `.DS_Store`, `data/`, checkpoints, and previews are not staged.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/plans/2026-07-16-motion-weighted-autoencoder.md \
  docs/experiments/2026-07-16-motion-weighted-autoencoder.md \
  src/world_model_lab/visual_latent_data.py \
  src/world_model_lab/train_visual_latent_model.py \
  tests/test_visual_latent_data.py tests/test_train_visual_latent_model.py
git commit -m "feat: add motion-weighted visual reconstruction"
```
