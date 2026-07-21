# Spatial Dynamics Objective Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Train the existing spatial CNN dynamics with normalized latent MSE plus a frozen-decoder changed-pixel MAE, while reusing the successful spatial autoencoder, normalizers, and episode split.

**Architecture:** The dynamics model still predicts a normalized residual latent from four ordered latent grids and four aligned actions. During training only, the predicted latent is denormalized and decoded by the frozen source autoencoder; an auxiliary MAE is computed on pixels that actually differ between the last context frame and target frame. A dedicated runner loads the source checkpoint, verifies the dataset hash, rebuilds the recorded windows, trains only a fresh dynamics model, and publishes a normal visual latent checkpoint plus preview.

**Tech Stack:** Python 3.12, NumPy, PyTorch, unittest, Matplotlib.

## Global Constraints

- Keep `SpatialConvAutoencoder`, `SpatialLatentDynamicsCNN`, dataset, split IDs, latent/action normalizers, seed, optimizer, batch size, learning rate, and 50-epoch budget unchanged.
- Use `total_loss = normalized_latent_mse + weight * changed_pixel_mae`.
- Define changed pixels from exact RGB inequality between the last context frame and target frame; do not read physical states or object labels.
- Use `weight=0.1` for the first controlled experiment.
- Select the best epoch by the same combined validation objective used for training.
- Preserve exact legacy behavior when `weight=0.0`.
- Reject non-finite or negative weights and reject positive weights without both a decoder and visual dataset.
- Do not update autoencoder parameters or normalizer statistics.
- Refuse output overwrites and roll back the checkpoint if preview publication fails.

---

### Task 1: Define the decoded changed-pixel objective

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Consumes: decoded predictions `[B, 3, 64, 64]`, target frames with the same shape, and binary masks `[B, 1, 64, 64]`.
- Produces: `_changed_pixel_mae_loss(predictions, targets, changed_masks) -> torch.Tensor`.
- Produces: `train_latent_dynamics(..., decoder=None, visual_dataset=None, changed_pixel_loss_weight=0.0)`.

- [x] **Step 1: Write failing formula and validation tests**

```python
def test_changed_pixel_mae_loss_uses_only_masked_rgb_values(self):
    target = torch.zeros((1, 3, 1, 2))
    prediction = torch.tensor(
        [[[[1.0, 0.5]], [[1.0, 0.5]], [[1.0, 0.5]]]]
    )
    mask = torch.tensor([[[[1.0, 0.0]]]])

    loss = _changed_pixel_mae_loss(prediction, target, mask)

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_changed_pixel_mae_loss_returns_differentiable_zero_for_empty_mask(self):
    prediction = torch.ones((1, 3, 2, 2), requires_grad=True)
    loss = _changed_pixel_mae_loss(
        prediction,
        torch.zeros_like(prediction),
        torch.zeros((1, 1, 2, 2)),
    )
    loss.backward()
    self.assertEqual(float(loss), 0.0)
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
```

- [x] **Step 2: Run the focused tests and verify the missing helper fails**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_train_visual_latent_model.VisualLatentDynamicsTrainingTest
```

Expected: failure because `_changed_pixel_mae_loss` is not defined.

- [x] **Step 3: Implement the exact mask-normalized loss**

```python
def _changed_pixel_mae_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    changed_masks: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape or predictions.ndim != 4:
        raise ValueError("decoded predictions and targets must share [B, C, H, W]")
    if changed_masks.shape != (
        predictions.shape[0],
        1,
        predictions.shape[2],
        predictions.shape[3],
    ):
        raise ValueError("changed_masks must have shape [B, 1, H, W]")
    if not torch.all((changed_masks == 0) | (changed_masks == 1)):
        raise ValueError("changed_masks must be binary")
    expanded = changed_masks.expand_as(predictions)
    error_sum = torch.sum(torch.abs(predictions - targets) * expanded)
    value_count = torch.sum(expanded)
    if int(value_count) == 0:
        return error_sum
    return error_sum / value_count
```

- [x] **Step 4: Add lazy image-supervision tensors and combined objective**

Build target `uint8` frames and boolean changed masks from
`LatentWindowArrays.target_frame_indices` and `last_frame_indices`. In each
batch, convert target frames to float NCHW, denormalize the prediction with
torch tensors made from the checkpoint normalizer, decode it without
`torch.no_grad()`, and compute:

```python
latent_mse = torch.mean(torch.square(prediction - target_latent))
decoded_changed_mae = _changed_pixel_mae_loss(
    decoder.decode(prediction * latent_std + latent_mean),
    target_frames,
    changed_masks,
)
loss = latent_mse + changed_pixel_loss_weight * decoded_changed_mae
```

Only the dynamics parameters belong to the optimizer. Set the decoder to eval
mode and set all decoder parameters to `requires_grad_(False)` so gradients
flow through decoder operations into predicted latents but never accumulate on
the autoencoder.

- [x] **Step 5: Test validation, frozen decoder, and weight-zero compatibility**

Add tests that:

1. reject `-1`, `nan`, and `inf`;
2. reject `weight > 0` without both decoder and dataset;
3. confirm decoder parameters are unchanged after training;
4. confirm two seed-identical `weight=0` runs have identical histories and
   model weights whether optional decoder inputs are supplied or omitted.

- [x] **Step 6: Run the focused tests**

Expected: all `VisualLatentDynamicsTrainingTest` tests pass.

### Task 2: Add a frozen-source dynamics runner

**Files:**
- Create: `src/world_model_lab/train_visual_dynamics_objective.py`
- Create: `tests/test_train_visual_dynamics_objective.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: a visual dataset and an existing visual latent checkpoint.
- Produces: `run_frozen_decoder_dynamics_training(...) -> dict[str, Any]`.
- Produces CLI: `world-model-train-visual-dynamics-objective`.

- [x] **Step 1: Write failing source-reuse tests**

Create a tiny source checkpoint with `run_visual_latent_training`, then call:

```python
summary = run_frozen_decoder_dynamics_training(
    data_path=data_path,
    source_checkpoint_path=source_path,
    output_path=output_path,
    preview_path=preview_path,
    changed_pixel_loss_weight=0.1,
    dynamics_epochs=1,
    dynamics_batch_size=8,
)
```

Assert:

```python
self.assertTrue(output_path.is_file())
self.assertTrue(preview_path.is_file())
self.assertEqual(
    loaded.training_config["source_checkpoint_sha256"],
    sha256_file(source_path),
)
self.assertTrue(loaded.training_config["autoencoder_frozen"])
self.assertEqual(
    loaded.training_config["dynamics_changed_pixel_loss_weight"],
    0.1,
)
for name, tensor in source.autoencoder.state_dict().items():
    torch.testing.assert_close(
        loaded.autoencoder.state_dict()[name],
        tensor,
        rtol=0.0,
        atol=0.0,
    )
```

- [x] **Step 2: Run the new test module and verify import failure**

Expected: failure because the runner module does not exist.

- [x] **Step 3: Implement source verification and common-data reconstruction**

The runner must:

```python
source = load_visual_latent_checkpoint(source_checkpoint_path)
dataset = load_visual_dataset(data_path)
dataset_sha256 = sha256_file(data_path)
if source.dataset_metadata.get("sha256") != dataset_sha256:
    raise ValueError("source checkpoint dataset SHA-256 does not match data")
```

Then reuse:

```python
splits = source.split_episode_ids
autoencoder = source.autoencoder
latent_normalizer = source.latent_normalizer
action_normalizer = source.action_normalizer
latent_frames = encode_all_frames(autoencoder, dataset["frames"], batch_size=...)
window_arrays = {
    name: build_latent_window_arrays(
        dataset,
        build_visual_window_index(dataset, splits[name]),
        latent_frames,
    )
    for name in ("train", "validation", "test")
}
```

Infer latent layout, spatial channels, dynamics architecture, hidden size,
optimizer parameters, seed, and batch sizes from the source checkpoint unless
the runner has an explicit training-budget argument.

- [x] **Step 4: Train only fresh dynamics and publish a normal checkpoint**

Call `train_latent_dynamics` with the frozen source decoder, visual dataset,
and auxiliary weight. Preserve the source autoencoder history and test metrics
by reconstructing a `PhaseTrainingResult` around the loaded autoencoder.
Record:

```python
{
    "source_checkpoint": str(source_path.resolve()),
    "source_checkpoint_sha256": sha256_file(source_path),
    "autoencoder_frozen": True,
    "dynamics_loss": (
        "normalized_latent_mse_plus_changed_pixel_mae"
    ),
    "dynamics_changed_pixel_loss_weight": weight,
}
```

Use the existing checkpoint writer and prediction plotter. If plotting raises,
delete the newly written output checkpoint and re-raise.

- [x] **Step 5: Add negative tests**

Cover:

- mismatched dataset SHA-256;
- global source checkpoint rejection, because this controlled experiment is
  specifically the spatial CNN baseline;
- ConvGRU source rejection;
- output collision before training;
- invalid auxiliary weight;
- preview failure checkpoint rollback.

- [x] **Step 6: Register and test the CLI**

Add:

```toml
world-model-train-visual-dynamics-objective = "world_model_lab.train_visual_dynamics_objective:main"
```

The CLI arguments are `--data`, `--source-checkpoint`, `--output`, `--preview`,
`--changed-pixel-loss-weight`, `--dynamics-epochs`, and
`--dynamics-batch-size`. Argument errors must exit through `parser.error`.

### Task 3: Document the contract

**Files:**
- Modify: `README.md`
- Create: `docs/experiments/2026-07-17-spatial-dynamics-objective-alignment.md`

**Interfaces:**
- Consumes: the new CLI and held-out metrics.
- Produces: reproducible command, exact objective, controlled variables,
  results, interpretation, and decision.

- [x] **Step 1: Add the training command to README**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python -m \
  world_model_lab.train_visual_dynamics_objective \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8.pt \
  --output artifacts/visual_latent_spatial8_objective_w01.pt \
  --preview artifacts/visual_latent_spatial8_objective_w01_predictions.png \
  --changed-pixel-loss-weight 0.1 \
  --dynamics-epochs 50 \
  --dynamics-batch-size 256
```

- [x] **Step 2: Explain the training-only target mask boundary**

Document that the target-derived mask is supervision only. It is not available
to the model at inference time, is not a physical-state label, and does not
change the evaluation mask definition.

- [x] **Step 3: Predetermine experiment gates**

Record before running:

```text
primary: changed-pixel MAE < 0.314072
stability: normalized latent MSE <= 0.0339449  # no more than 10% worse
representation: Oracle metrics and autoencoder weights exactly unchanged
```

### Task 4: Execute and verify the controlled experiment

**Files:**
- Generate ignored: `artifacts/visual_latent_spatial8_objective_w01.pt`
- Generate ignored: `artifacts/visual_latent_spatial8_objective_w01_predictions.png`
- Modify: `docs/experiments/2026-07-17-spatial-dynamics-objective-alignment.md`

**Interfaces:**
- Consumes: `data/visual_episodes.npz` and
  `artifacts/visual_latent_spatial8.pt`.
- Produces: one reloadable candidate checkpoint, preview, and recorded
  decision.

- [x] **Step 1: Run focused and full verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_train_visual_latent_model \
  tests.test_train_visual_dynamics_objective

PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v

PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

git diff --check
```

Expected: zero failures and zero command errors.

- [x] **Step 2: Run the 50-epoch candidate**

Use the README command with `weight=0.1`.

- [x] **Step 3: Reload and independently recompute held-out metrics**

Load the new checkpoint, verify both dataset and source hashes, encode frames,
rebuild the checkpoint's test windows, and call `evaluate_latent_dynamics`.
Require every recomputed metric to match the stored value.

- [x] **Step 4: Compare against all fixed baselines**

Report candidate versus:

- CNN latent-only World: latent MSE `0.0308590`, changed MAE `0.314072`;
- decoded-last: changed MAE `0.359824`;
- Oracle: changed MAE `0.266102`;
- raw copy-last: changed MAE `0.601716`.

Also run the existing action and history ablations on the candidate.

- [x] **Step 5: Record the decision without result shopping**

If both gates pass, retain the aligned objective for the next multi-step
experiment. If changed MAE does not improve, reject `weight=0.1` and diagnose
gradient scale or mask sparsity before trying another value. If changed MAE
improves but latent MSE violates the stability gate, do not adopt the
candidate; the next experiment may lower the predeclared weight.

- [x] **Step 6: Final diff and artifact verification**

Run the full test suite, `compileall`, `git diff --check`, inspect
`git status --short`, and verify the source and candidate checkpoint hashes
recorded in the experiment document.
