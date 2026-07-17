# Spatial Latent World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate an ordinary-MSE visual world model whose encoder preserves an `8 x 8` latent grid instead of collapsing each frame into one 32-value global vector.

**Architecture:** Keep the existing global `ConvAutoencoder` and `LatentDynamicsMLP` as the default baseline. Add a spatial autoencoder that emits `[B, C, 8, 8]` and a residual convolutional dynamics model that restores the grid after reversible storage flattening; select the pair with `--latent-layout spatial`. Reuse the current episode splits, latent/action normalization, Oracle reconstruction, changed-pixel, and copy-last evaluation paths.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, unittest, Matplotlib.

## Global Constraints

- Preserve existing global-model behavior and checkpoint loading.
- Use `latent_layout=global` by default.
- Use ordinary pixel MSE (`motion_loss_weight=0.0`) in the controlled spatial experiment.
- Do not read `states`, rewards, dones, or hand-authored foreground labels.
- Preserve the `64 x 64` image contract and four-frame/action alignment.
- Keep the spatial grid reversible when compact window arrays store it as a flat vector.
- Run the real experiment with `latent_channels=8`, grid size `8 x 8`, spatial dynamics hidden channels `64`, autoencoder `20` epochs, dynamics `50` epochs, training seed `0`, and split seed `42`.
- Publish new artifacts without overwriting prior checkpoints or previews.

---

### Task 1: Spatial model contracts

**Files:**
- Modify: `src/world_model_lab/visual_latent_model.py`
- Test: `tests/test_visual_latent_model.py`

**Interfaces:**
- Produces: `SpatialConvAutoencoder(latent_channels, base_channels)` with `encode(images) -> [B,C,8,8]`, `decode(latents) -> [B,3,64,64]`, and `latent_dim=C*8*8`.
- Produces: `SpatialLatentDynamicsCNN(latent_channels, hidden_channels, context_frames)` consuming flattened context `[B,4,C*8*8]` and returning `[B,C*8*8]`.

- [ ] **Step 1: Write failing exact-shape tests**

```python
autoencoder = SpatialConvAutoencoder(latent_channels=3, base_channels=4)
latents = autoencoder.encode(torch.zeros((2, 3, 64, 64)))
assert latents.shape == (2, 3, 8, 8)
assert autoencoder.decode(latents).shape == (2, 3, 64, 64)
assert autoencoder.decode(latents.flatten(start_dim=1)).shape == (2, 3, 64, 64)
```

- [ ] **Step 2: Verify tests fail before implementation**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_model -v
```

Expected: import failure for both spatial classes.

- [ ] **Step 3: Implement the spatial autoencoder**

Use three stride-two convolutions to map `64 -> 32 -> 16 -> 8`, no `Linear` layer, and three transposed convolutions to map `8 -> 16 -> 32 -> 64`. `decode` accepts either the exact grid or its reversible flattened representation and rejects every other shape.

- [ ] **Step 4: Implement residual convolutional dynamics**

Reconstruct `[B,4,C,8,8]` from flattened contexts, concatenate four aligned two-value actions as eight broadcast feature maps, predict a local convolutional delta, and return:

```python
next_grid = context_grid[:, -1] + predicted_delta
return next_grid.flatten(start_dim=1)
```

- [ ] **Step 5: Verify model contracts pass**

Expected: all visual latent model tests pass, including zero-network residual behavior and invalid-shape rejection.

---

### Task 2: Reversible spatial storage and training selection

**Files:**
- Modify: `src/world_model_lab/visual_latent_data.py`
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Test: `tests/test_visual_latent_data.py`
- Test: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- `encode_all_frames` continues returning `[F, latent_dim]`; a spatial output is flattened in channel-major raster order without a learned projection.
- `train_autoencoder(..., latent_layout, spatial_latent_channels)` selects the autoencoder.
- `train_latent_dynamics(..., latent_layout, spatial_latent_channels)` selects MLP or CNN dynamics.
- `run_visual_latent_training` exposes the same layout controls end to end.

- [ ] **Step 1: Write failing storage and tiny-training tests**

Assert that encoding two frames with `latent_channels=3` produces shape `[2,192]` and equals `model.encode(images).flatten(start_dim=1)`. Add a one-epoch end-to-end spatial run and assert the checkpoint reloads the spatial classes and reports `latent_layout=spatial`.

- [ ] **Step 2: Verify the focused tests fail**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_data tests.test_train_visual_latent_model -v
```

Expected: failures for missing spatial layout parameters and unsupported four-dimensional encoder output.

- [ ] **Step 3: Flatten spatial encoder outputs only at the storage boundary**

Inside `encode_all_frames`, validate that the batch dimension is preserved and use:

```python
stored_latents = encoded.flatten(start_dim=1)
```

This keeps every `(channel,row,column)` value and fixed position while satisfying the existing compact `[F,D]` window contract.

- [ ] **Step 4: Add explicit model factories**

Validate layout against `{"global", "spatial"}`. For global layout instantiate the old classes unchanged. For spatial layout instantiate `SpatialConvAutoencoder` and `SpatialLatentDynamicsCNN`, verifying `latent_dim == latent_channels * 8 * 8` before dynamics training.

- [ ] **Step 5: Keep evaluation generic**

Update model type annotations and runtime checks so both model pairs use the same ordinary-MSE training, normalization, Oracle decoding, changed-pixel MAE, copy-last, and preview code.

- [ ] **Step 6: Run focused training tests**

Expected: old global tests and new spatial tests all pass.

---

### Task 3: Backward-compatible checkpoints and CLI

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `README.md`
- Test: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Consumes: `--latent-layout {global,spatial}`, `--spatial-latent-channels INT`.
- Produces: checkpoint `model_config["latent_layout"]` plus spatial channel metadata.
- Loads: legacy format-version-1 checkpoints that omit `latent_layout` as global models.

- [ ] **Step 1: Write failing checkpoint and CLI tests**

Round-trip a spatial pair, assert exact dynamics output equality after reload, and assert CLI help contains both new options. Keep the existing global round-trip test as a legacy/default guard.

- [ ] **Step 2: Extend model metadata without changing format version**

Write `latent_layout`, `spatial_latent_channels`, and the shared dynamics hidden width. On load, use:

```python
latent_layout = str(model_config.get("latent_layout", "global"))
```

so existing checkpoints remain readable.

- [ ] **Step 3: Add CLI arguments and documentation**

Document that the grid is temporarily flattened only for storage and normalization, then restored before convolutional dynamics and decoding. State that the default remains the 32-value global baseline.

- [ ] **Step 4: Run checkpoint and CLI tests**

Expected: global and spatial checkpoint round trips pass; CLI still emits strict sorted JSON.

---

### Task 4: Full verification

**Files:**
- Verify all modified source, tests, and documentation.

**Interfaces:**
- Produces: reproducible evidence before the long experiment.

- [ ] **Step 1: Run compilation and complete unit suite**

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

- [ ] **Step 2: Review compatibility boundaries**

Load the prior `artifacts/visual_latent_ae20_default.pt` and verify it resolves to the global classes. Confirm `.DS_Store`, datasets, checkpoints, and previews remain untracked.

---

### Task 5: Controlled 20/50-epoch experiment

**Files:**
- Create: `docs/experiments/2026-07-17-spatial-latent-world-model.md`
- Generate ignored: `artifacts/visual_latent_spatial8.pt`
- Generate ignored: `artifacts/visual_latent_spatial8_predictions.png`

**Interfaces:**
- Consumes: unchanged visual dataset and split.
- Produces: directly comparable Oracle/world/copy-last metrics and a decision.

- [ ] **Step 1: Train the spatial model**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_latent_model \
  --data data/visual_episodes.npz \
  --output artifacts/visual_latent_spatial8.pt \
  --preview artifacts/visual_latent_spatial8_predictions.png \
  --latent-layout spatial --spatial-latent-channels 8 \
  --dynamics-hidden-size 64 \
  --autoencoder-epochs 20 --dynamics-epochs 50 \
  --motion-loss-weight 0 \
  --seed 0 --split-seed 42
```

- [ ] **Step 2: Apply the existing gates**

Record Oracle and world full-frame MSE and changed-pixel MAE. Compare against the global baseline and copy-last. Do not claim a pure topology effect because spatial `[8,8,8]` contains 512 scalars versus the global baseline's 32.

- [ ] **Step 3: Record the next decision**

If Oracle changed-pixel MAE materially improves while full-frame MSE remains controlled, retain spatial latents and next isolate capacity with a global-512 ablation. If it fails, inspect the preview and reconsider the autoencoder objective or image scale before multi-step rollout work.

