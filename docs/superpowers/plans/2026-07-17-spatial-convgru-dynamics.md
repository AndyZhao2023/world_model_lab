# Spatial ConvGRU Dynamics Implementation Plan

> **For Codex:** Implement this plan with test-first changes and fresh verification.

**Goal:** Compare the existing channel-concatenation spatial CNN against a
ConvGRU that consumes the same four normalized spatial latents and aligned
actions one time step at a time.

**Controlled experiment:** Reuse `artifacts/visual_latent_spatial8.pt` without
retraining its autoencoder. Preserve its dataset hash, episode splits, latent
normalizer, action normalizer, encoder, and decoder. Train only the replacement
dynamics model for 50 epochs with seed 0, batch size 256, learning rate 1e-3,
and 40 hidden channels.

**Architecture:** Each of the four flattened latent vectors is reshaped to
`[B, C, 8, 8]`. Its aligned 2-D action is broadcast over the same grid and
concatenated to the latent channels. A convolutional GRU cell updates one
hidden grid per step. A final 3x3 convolution predicts a residual that is added
to the last context latent and flattened back to `[B, C*8*8]`.

## Task 1: Specify ConvGRU behavior with failing tests

**Files:**
- Modify: `tests/test_visual_latent_model.py`

1. Test the exact output shape for a small batch.
2. Zero every parameter and verify that the residual path returns the final
   context latent exactly.
3. Reject non-positive channels, fewer than two context frames, and malformed
   context/action shapes.
4. Run only these tests and confirm they fail because the class is absent.

## Task 2: Implement the recurrent spatial dynamics module

**Files:**
- Modify: `src/world_model_lab/visual_latent_model.py`

1. Add `SpatialLatentDynamicsConvGRU`.
2. Use separate 3x3 convolutions for update/reset gates, candidate state, and
   output residual.
3. Align the three history actions and one current action with the four context
   latent frames.
4. Keep `latent_dim`, `hidden_size`, and `context_frames` attributes compatible
   with existing training/checkpoint code.
5. Run the focused model tests.

## Task 3: Add architecture selection and checkpoint compatibility

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_train_visual_latent_model.py`

1. Add `cnn` and `convgru` as validated spatial dynamics architectures.
2. Pass the selection through `_make_dynamics`, `train_latent_dynamics`,
   `run_visual_latent_training`, training metadata, and
   `--spatial-dynamics-architecture`.
3. Save the concrete architecture in `model_config`.
4. Load old spatial checkpoints that omit the field as `cnn`; global
   checkpoints remain MLP-based.
5. Test ConvGRU checkpoint round-trip, exact prediction preservation, tiny
   end-to-end training, CLI help, and legacy default behavior.

## Task 4: Run the frozen-autoencoder comparison

**Files:**
- Create: `artifacts/visual_latent_spatial8_convgru.pt`
- Create: `artifacts/visual_latent_spatial8_convgru_predictions.png`

1. Load `artifacts/visual_latent_spatial8.pt`.
2. Verify its recorded dataset SHA-256 against the current dataset.
3. Encode frames with the frozen autoencoder and rebuild windows using the
   recorded episode splits.
4. Reuse the recorded normalizers without refitting.
5. Train only `SpatialLatentDynamicsConvGRU` with 40 hidden channels for 50
   epochs.
6. Evaluate using normalized latent MSE, full-frame pixel MSE, changed-pixel
   MAE, action ablations, and history ablations.
7. Save a reloadable checkpoint and prediction preview.

## Task 5: Record evidence and verify

**Files:**
- Create: `docs/experiments/2026-07-17-spatial-convgru-dynamics.md`
- Modify: `docs/experiments/2026-07-17-spatial-history-diagnostics.md`
- Modify: `README.md`

1. Report parameter counts and the exact controlled protocol.
2. Compare ConvGRU against the existing CNN, decoded-last baseline, and Oracle.
3. State whether explicit recurrent ordering improves dynamics and whether
   action/history ablations support the interpretation.
4. Run the focused tests, full test suite, `compileall`, and `git diff --check`.
5. Reload the new checkpoint and reproduce its stored evaluation metrics.
