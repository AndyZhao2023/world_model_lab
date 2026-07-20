# Visual object-slot local-write decoder implementation plan

> **Execution:** Implement this plan inline with test-first changes. Train
> exactly the single candidate registered in
> `docs/experiments/2026-07-20-visual-object-slot-decoder.md`.

**Goal:** Add a frozen-base object-slot decoder whose learned writes are
structurally limited to an 11x11 window, then decide it using six
pre-registered representation gates.

**Architecture:** Extend `SpatialConvAutoencoder` with a mutually exclusive
object-slot mode. A spatial-softmax centre and attention-pooled heading form a
four-value physical slot. A small MLP turns that slot into a local RGBA patch,
which is bilinearly placed and hard-clamped to an 11x11 support before
compositing over the frozen base decoder.

**Tech stack:** Python 3.10+, PyTorch, NumPy, unittest, existing
world-model-lab checkpoint and diagnostic utilities.

---

### Task 1: Specify slot targets and local support with failing tests

**Files:**

- Create: `tests/test_visual_object_slot.py`
- Create: `src/world_model_lab/visual_object_slot.py`

**Steps:**

1. Add failing tests for physical state to
   `[image_cx, image_cy, sin(theta), cos(theta)]` conversion, including
   letterboxed y coordinates and angle wrap.
2. Add a failing test that a slot-enabled model produces finite
   `[B, 4]` states and `[B, 3/1, 64, 64]` placed components.
3. Add a failing test proving alpha is exactly zero outside the 11x11 support
   for both integer and fractional predicted centres.
4. Run:

   ```bash
   PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
     /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
     -m unittest tests.test_visual_object_slot -v
   ```

   Confirm failure because the object-slot API does not exist.
5. Implement only the target conversion, spatial-softmax slot head, patch MLP,
   differentiable placement, and hard support needed by those tests.
6. Re-run the focused test and confirm it passes.

### Task 2: Extend checkpoint compatibility test-first

**Files:**

- Modify: `src/world_model_lab/visual_latent_model.py`
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_train_visual_latent_model.py`

**Steps:**

1. Add failing checkpoint tests for:
   - slot config round-trip;
   - exact state-dict round-trip;
   - legacy checkpoints with no slot keys;
   - rejection of simultaneous dense residual and object-slot modes;
   - rejection of invalid patch or hidden sizes.
2. Run the focused checkpoint tests and confirm the new cases fail.
3. Add model config fields:

   ```text
   object_slot_decoder
   object_slot_patch_size
   object_slot_hidden_size
   ```

4. Keep missing fields backward compatible as slot-disabled zero values.
5. Re-run focused tests and the existing residual-decoder tests.

### Task 3: Implement the declared objective and deterministic trainer

**Files:**

- Modify: `src/world_model_lab/visual_object_slot.py`
- Modify: `tests/test_visual_object_slot.py`

**Steps:**

1. Add failing tests that the total objective exactly equals:

   ```text
   full_mse
   + 1.0 * foreground_object_mse
   + 0.01 * balanced_alpha_bce
   + 1.0 * centre_mse
   + 0.1 * heading_mse
   ```

2. Add failing tests proving encoder/base decoder parameters are frozen,
   slot parameters receive gradients, source tensors remain bit-identical,
   invalid masks/states/weights fail closed, and one-epoch toy training is
   deterministic.
3. Implement `VisualObjectSlotFrameDataset`,
   `initialize_object_slot_autoencoder`, `object_slot_objective`,
   `train_object_slot_decoder`, and held-out slot/alpha evaluation.
4. Re-run `tests.test_visual_object_slot` until all tests pass.

### Task 4: Add the production training entry point

**Files:**

- Create: `src/world_model_lab/train_visual_object_slot.py`
- Create: `tests/test_train_visual_object_slot.py`
- Modify: `pyproject.toml`

**Steps:**

1. Add failing integration tests for source/dataset digest validation,
   output preflight, frozen source encoder/base/dynamics invariants, checkpoint
   reload, JSON summary fields, and CLI `--help`.
2. Implement:

   ```text
   world-model-train-visual-object-slot
   ```

   with registered defaults:

   ```text
   source      artifacts/visual_latent_spatial8_objective_w01.pt
   output      artifacts/visual_latent_spatial8_object_slot.pt
   preview     artifacts/visual_latent_spatial8_object_slot_predictions.png
   patch       11
   hidden      64
   alpha       0.01
   epochs      20
   batch       128
   lr          0.001
   weights     foreground=1, mask=0.01, centre=1, heading=0.1
   ridge       0.001
   ```

3. Reuse the source dynamics and normalizers unchanged, compute the source
   ridge-probe and candidate direct-state held-out metrics, save the candidate
   through the existing atomic checkpoint path, and generate a deterministic
   preview.
4. Run both new test modules and existing latent/residual checkpoint tests.

### Task 5: Run the single candidate and representation diagnostic

**Files:**

- Modify:
  `docs/experiments/2026-07-20-visual-object-slot-decoder.md`
- Generated and ignored:
  `artifacts/visual_latent_spatial8_object_slot.pt`
- Generated and ignored:
  `artifacts/visual_latent_spatial8_object_slot_predictions.png`
- Generated and ignored:
  `artifacts/diagnostics/visual-object-slot/`

**Steps:**

1. Record SHA-256 of the registered data and source checkpoint and abort on
   mismatch.
2. Train exactly once:

   ```bash
   PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
     /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
     -m world_model_lab.train_visual_object_slot \
     --data data/visual_episodes.npz \
     --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
     --output artifacts/visual_latent_spatial8_object_slot.pt \
     --preview artifacts/visual_latent_spatial8_object_slot_predictions.png \
     --patch-size 11 --hidden-size 64 --initial-alpha 0.01 \
     --epochs 20 --batch-size 128 --learning-rate 0.001 \
     --foreground-loss-weight 1.0 --mask-loss-weight 0.01 \
     --centre-loss-weight 1.0 --heading-loss-weight 0.1 \
     --source-probe-ridge 0.001
   ```

3. Run the existing matched representation diagnostic:

   ```bash
   PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
     /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
     -m world_model_lab.diagnose_visual_autoencoder \
     --data data/visual_episodes.npz \
     --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
     --candidate-checkpoint artifacts/visual_latent_spatial8_object_slot.pt \
     --output-dir artifacts/diagnostics/visual-object-slot
   ```

4. Combine the diagnostic's four reconstruction gates with the training
   summary's two direct-state gates. Stop before H5 if any gate fails.
5. Update the experiment document from `Registered` to `Completed`, including
   exact metrics, six PASS/FAIL rows, artifact hashes, and the strict decision.

### Task 6: Verify the repository and hand off

**Files:**

- Modify: `README.md` only if its experiment/CLI index requires the new command.

**Steps:**

1. Run all tests:

   ```bash
   PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
     /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
     -m unittest discover -s tests -v
   ```

2. Run compile and CLI checks:

   ```bash
   PYTHONPATH=src \
     /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
     -m compileall -q src tests

   PYTHONPATH=src \
     /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
     -m world_model_lab.train_visual_object_slot --help
   ```

3. Inspect `git diff --check`, `git status --short`, and the final diff.
4. Commit only source, tests, `pyproject.toml`, and documentation. Do not stage
   `.DS_Store`, `data/`, or `artifacts/`.

## Self-review

- The support change is structural: alpha is exactly zero outside the local
  window; it is not another dense-alpha scalar sweep.
- Simulator state and masks are supervision only and are absent from
  inference.
- The source encoder, base decoder, dynamics, normalizers, and splits remain
  frozen and directly comparable.
- The patch size and all gates were fixed before candidate training.
- Existing checkpoints remain loadable.
- Failure of any representation gate prevents H5 training and MPC work.
