# Visual Object Residual Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one learned object-only foreground/alpha decoder on top of the
frozen spatial autoencoder, preserve the source background path exactly, and
train H5 dynamics only if the enhanced decoder passes all representation
gates.

**Architecture:** Keep the source encoder and RGB decoder byte-equal and add a
three-stage transposed-convolution head from `[B, 8, 8, 8]` latents to three
foreground RGB channels plus one alpha logit. Composite
`base * (1 - alpha) + foreground * alpha`; train only the new head with
full-frame reconstruction, object-foreground reconstruction, and a balanced
object/background mask loss.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, unittest.

## Global Constraints

- Use `data/visual_episodes.npz` unchanged, with SHA-256
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`.
- Use `artifacts/visual_latent_spatial8_objective_w01.pt` as source.
- Keep source encoder, base decoder, source dynamics, latent/action
  normalizers, and train/validation/test episode IDs exactly unchanged.
- Keep latent shape `[B, 8, 8, 8]`, object-head hidden channels `16`, batch
  size `128`, learning rate `1e-3`, 20 epochs, seed `0`.
- Initialize alpha probability to `0.01`.
- Use fixed loss:
  `full MSE + 1.0 * object foreground MSE + 0.01 * balanced mask BCE`.
- Derive the training mask only from exact renderer `CAR_COLOR` or
  `HEADING_COLOR`; the true mask is not an inference input.
- Train exactly one residual-decoder candidate; do not sweep weights,
  channels, alpha initialization, seeds, or epoch budgets.
- Preserve legacy checkpoint loading by defaulting the new model config to no
  residual head.
- Refuse to overwrite checkpoints, previews, or non-empty diagnostics.
- Apply the same four locked representation gates:
  object MSE `< 0.1580440933868`, full MSE `<= 0.0016618691395`,
  background MSE `<= 0.0006603598343`, and H5 oracle cumulative changed-pixel
  MAE `< 0.1996767445825`.
- Train H5 only if all four representation gates pass.
- If H5 runs, compare residual H5 against residual H1 so both sides use the
  identical enhanced decoder.
- Do not connect any candidate to MPC unless every registered gate passes.

## File Structure

- Modify `src/world_model_lab/visual_latent_model.py`: optional foreground and
  alpha residual decoder while preserving legacy behavior.
- Modify `src/world_model_lab/train_visual_latent_model.py`: persist and load
  residual-decoder model configuration.
- Create `src/world_model_lab/visual_object_residual.py`: objective,
  frozen-base training, and alpha-mask evaluation.
- Create `src/world_model_lab/train_visual_object_residual.py`: controlled
  source-checkpoint runner and CLI.
- Modify `pyproject.toml`: register the residual-decoder trainer.
- Modify `tests/test_visual_latent_model.py`: blend, shape, and legacy tests.
- Modify `tests/test_train_visual_latent_model.py`: checkpoint round-trip and
  legacy model-config tests.
- Create `tests/test_visual_object_residual.py`: loss and frozen-gradient
  tests.
- Create `tests/test_train_visual_object_residual.py`: runner, invariants,
  publication, and CLI tests.
- Create
  `docs/experiments/2026-07-19-visual-object-residual-decoder.md`: protocol,
  results, artifacts, and decision.

---

### Task 1: Add an optional object residual decoder

**Files:**
- Modify: `src/world_model_lab/visual_latent_model.py`
- Modify: `tests/test_visual_latent_model.py`

**Interfaces:**
- Extends:
  `SpatialConvAutoencoder(..., object_residual_decoder=False,
  object_head_channels=None, object_initial_alpha=0.01)`.
- Produces:
  `decode_components(latents) -> (base, foreground, mask_logits, composite)`.

- [x] **Step 1: Write failing model tests**

Require legacy construction to have no object-head parameters and produce the
same decode path. For an enabled head, require exact tensor shapes and:

```python
expected = base * (1.0 - torch.sigmoid(mask_logits))
expected += foreground * torch.sigmoid(mask_logits)
torch.testing.assert_close(composite, expected)
```

Reject an enabled head with non-positive channels or alpha outside `(0, 1)`.

- [x] **Step 2: Run the focused test and confirm signature failures**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_model.SpatialConvAutoencoderTest -v
```

- [x] **Step 3: Implement latent-grid validation and alpha blending**

Use one shared latent-shape helper. The object head is:

```python
nn.Sequential(
    nn.ConvTranspose2d(latent_channels, 32, 4, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(32, 16, 4, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(16, 4, 4, 2, 1),
)
```

Interpret channels `0:3` through sigmoid as foreground and channel `3:4` as
alpha logits. Initialize the final alpha bias to
`log(0.01 / 0.99)`.

- [x] **Step 4: Run focused model tests**

Expected: legacy and residual model tests pass.

---

### Task 2: Persist residual models without breaking old checkpoints

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Adds model config:
  `object_residual_decoder: bool` and `object_head_channels: int`.
- Defaults absent fields to `False` and `0`.

- [x] **Step 1: Write failing round-trip and legacy tests**

Save an enabled spatial autoencoder, reload it, and require byte-equal
predictions and state tensors. Remove both new config keys from a legacy
payload and require it to load as a plain spatial autoencoder.

- [x] **Step 2: Run checkpoint tests and confirm load failure**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_visual_latent_model.VisualLatentCheckpointTest -v
```

- [x] **Step 3: Thread model configuration through factory/save/load**

Reject residual settings for global latents. Require a strict boolean flag,
positive head channels when enabled, and zero head channels when disabled.

- [x] **Step 4: Run focused checkpoint tests**

Expected: all visual checkpoint tests pass.

---

### Task 3: Train only the new head

**Files:**
- Create: `src/world_model_lab/visual_object_residual.py`
- Create: `tests/test_visual_object_residual.py`

**Interfaces:**
- Produces:
  `initialize_object_residual_autoencoder(source, *, head_channels,
  initial_alpha, seed) -> SpatialConvAutoencoder`.
- Produces:
  `object_residual_objective(model, images, masks, *,
  foreground_weight, mask_weight) -> (loss, components)`.
- Produces:
  `train_object_residual_decoder(...) -> PhaseTrainingResult`.
- Produces:
  `evaluate_object_residual_mask(...) -> dict[str, float | int]`.

- [x] **Step 1: Write failing objective and gradient tests**

For one object and one background pixel, require:

```text
total = full composite MSE
      + foreground_weight * object-only foreground MSE
      + mask_weight * (object BCE + background BCE)
```

Backpropagate and require finite gradients only on
`object_decoder_convolutions`; encoder/base-decoder parameters must be frozen
and byte-equal before/after one optimizer step.

- [x] **Step 2: Run tests and confirm missing module**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_object_residual -v
```

- [x] **Step 3: Implement deterministic source copying and freezing**

Instantiate the enabled model inside `torch.random.fork_rng`, seed it, copy
`encoder_convolutions` and `decoder_convolutions` state dicts from the source,
then set their parameters `requires_grad=False`.

- [x] **Step 4: Implement fixed objective and best-validation restore**

Use `VisualObjectFrameDataset`, Adam over only trainable head parameters, the
seeded DataLoader generator, and deep-copy the minimum validation total state.
Reject empty object/background regions and non-finite losses.

- [x] **Step 5: Implement alpha diagnostics**

Report threshold-0.5 intersection-over-union, precision, recall, mean alpha
inside the object, and mean alpha in the background on held-out episodes.

- [x] **Step 6: Run focused residual tests**

Expected: all objective, freezing, determinism, and metric tests pass.

---

### Task 4: Add the controlled residual training runner

**Files:**
- Create: `src/world_model_lab/train_visual_object_residual.py`
- Create: `tests/test_train_visual_object_residual.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  `run_visual_object_residual_training(...) -> dict[str, Any]`.
- Registers:
  `world-model-train-visual-object-residual`.

- [x] **Step 1: Write failing runner and CLI tests**

Run one epoch on a physical-colour fixture. Require source/candidate encoder,
base decoder, dynamics, normalizers, and split IDs to be exact; require only
object-head state to differ. Check the fixed config and mask metrics.

- [x] **Step 2: Run focused tests and confirm missing runner**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_visual_object_residual -v
```

- [x] **Step 3: Implement source validation and candidate publication**

Require a plain spatial CNN source with matching dataset SHA-256. Reuse source
dynamics and metadata, train only the object head, compute standard and alpha
metrics, and atomically publish checkpoint plus preview with rollback.

- [x] **Step 4: Record exact protocol metadata**

Store head channels, alpha initialization, three loss weights, exact mask
definition, frozen component flags, source digest, and
`dynamics_reused_unmodified=True`.

- [x] **Step 5: Run focused runner tests**

Expected: complete checkpoint/preview, strict invariants, CLI help, collision
handling, and rollback tests pass.

---

### Task 5: Train once and apply representation gates

**Files:**
- Modify:
  `docs/experiments/2026-07-19-visual-object-residual-decoder.md`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_residual.pt`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_residual_predictions.png`
- Generated but not committed:
  `artifacts/diagnostics/visual-object-residual/`

- [x] **Step 1: Run the full suite before training**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests
```

- [x] **Step 2: Train the single registered candidate**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_object_residual \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --output artifacts/visual_latent_spatial8_object_residual.pt \
  --preview artifacts/visual_latent_spatial8_object_residual_predictions.png \
  --head-channels 16 \
  --initial-alpha 0.01 \
  --epochs 20 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --foreground-loss-weight 1.0 \
  --mask-loss-weight 0.01
```

- [x] **Step 3: Run the existing matched representation diagnostic**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_autoencoder \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --candidate-checkpoint artifacts/visual_latent_spatial8_object_residual.pt \
  --output-dir artifacts/diagnostics/visual-object-residual \
  --horizons 1 5 10 \
  --windows-per-episode 8 \
  --decision-horizon 5
```

- [x] **Step 4: Apply all four gates**

Stop before H5 on any failure. Record alpha metrics as diagnostic evidence,
not additional post-hoc gates.

---

### Task 6: Conditionally train H5 and run matched counterfactual evaluation

**Files:**
- Modify:
  `docs/experiments/2026-07-19-visual-object-residual-decoder.md`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_residual_h5.pt`
- Generated but not committed:
  `artifacts/visual_latent_spatial8_object_residual_h5_predictions.png`
- Generated but not committed:
  `artifacts/diagnostics/visual-object-residual-h5-counterfactual/`

- [x] **Step 1: Evaluate the representation promotion condition**

The candidate failed two representation gates, so H5 was not run.

- [x] **Step 2: Evaluate the counterfactual diagnostic condition**

No residual H5 checkpoint exists, so the conditional diagnostic was not run.

- [x] **Step 3: Record the conditional H5 decision**

The current default is retained; no H5 or MPC promotion was attempted.

---

### Task 7: Verify, document, and commit

**Files:**
- Modify all files above.

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

- [x] **Step 2: Verify repository boundaries**

Stage only source, tests, plan, experiment document, and `pyproject.toml`.
Leave data, checkpoints, previews, diagnostics, caches, and `.DS_Store`
untracked or ignored.

- [x] **Step 3: Commit the controlled result**

Commit the implementation and recorded gate decision together.
