# Visual Recursive Rollout Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Train a fresh spatial CNN dynamics model with the existing one-step frozen-decoder objective plus an H5 differentiable free-rollout objective, then compare it with the current objective-aligned H1 checkpoint under the existing H1/H5/H10 diagnostic.

**Architecture:** Add episode-safe dense rollout arrays beside the existing one-step latent arrays. A focused recursive-training module will normalize one true four-frame context, recursively append each predicted latent while shifting the aligned action history, and backpropagate latent plus decoded changed-pixel losses through all five steps. A new runner will reuse the source autoencoder, normalizers, split, architecture, seed, optimizer settings, and one-step dataset while training fresh dynamics with one additional rollout term.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, unittest.

## Global Constraints

- Use `artifacts/visual_latent_spatial8_objective_w01.pt` as the source checkpoint.
- Reinitialize spatial CNN dynamics with the source seed; do not fine-tune source dynamics weights.
- Keep the source spatial autoencoder weights exactly frozen and byte-equal after reload.
- Keep latent/action normalizers and train/validation/test episode IDs exactly unchanged.
- Keep dynamics hidden channels, learning rate, batch size, epoch budget, and seed unchanged unless explicitly overridden by the CLI for tests.
- Keep the one-step objective `normalized latent MSE + 0.1 * decoded changed-pixel MAE`.
- Add H5 free-rollout loss with weight `1.0`; each rollout step uses the same latent plus adjacent true-frame changed-pixel objective.
- Build every valid training and validation H5 start; never cross an episode or pad a short episode.
- Optimize `one_step_loss + rollout_loss_weight * mean(H5 step losses)`.
- Choose the best checkpoint by the matching total validation objective.
- Evaluate the candidate with the existing episode-equal H1/H5/H10 visual rollout diagnostic.
- Do not connect the candidate to MPC regardless of this experiment's result.
- Refuse to overwrite checkpoint, preview, or non-empty diagnostic output paths.

## Pre-Registered Gates

Reference values from the current objective-aligned checkpoint under the H10-capable, episode-equal diagnostic:

```text
H1 free normalized latent MSE         0.0362509983
H1 free cumulative changed-pixel MAE  0.3034423844
H5 free normalized latent MSE         0.3129600827
H5 free cumulative changed-pixel MAE  0.2759844311
H10 free normalized latent MSE        0.8571245613
H10 free cumulative changed-pixel MAE 0.2991328294
```

The H5 candidate passes only if:

```text
primary H5:
    free normalized latent MSE < 0.3129600827
    free cumulative changed-pixel MAE < 0.2759844311

H1 stability:
    free normalized latent MSE <= 0.0398760981  # at most 10% worse
    free cumulative changed-pixel MAE <= 0.3186145036  # at most 5% worse

representation invariance:
    source and candidate autoencoder tensors are exactly equal
    source and candidate normalizers and split IDs are exactly equal
```

H10 is a secondary generalization result because the candidate is trained only through H5. If either primary H5 gate fails, retain the current objective-aligned checkpoint and record a valid negative experiment before changing horizon or rollout weight.

---

### Task 1: Build episode-safe H-step latent rollout arrays

**Files:**
- Modify: `src/world_model_lab/visual_latent_data.py`
- Modify: `tests/test_visual_latent_data.py`

**Interfaces:**
- Consumes: validated visual dataset, selected episode IDs, encoded frame latents, positive rollout horizon.
- Produces: `LatentRolloutArrays`.
- Produces: `build_latent_rollout_arrays(dataset, selected_episode_ids, latent_frames, *, horizon)`.

- [x] **Step 1: Write the failing alignment test**

Use transition lengths `(10, 8, 7)`, selected IDs `[10, 11, 12]`, and `horizon=5`. For episode 10, valid current-action steps are `3, 4, 5`; episode 11 has step `3`; episode 12 is ineligible.

```python
arrays = build_latent_rollout_arrays(
    visual,
    np.asarray([10, 11, 12], dtype=np.int64),
    latents,
    horizon=5,
)

np.testing.assert_array_equal(arrays.episode_ids, [10, 10, 10, 11])
np.testing.assert_array_equal(arrays.start_step_ids, [3, 4, 5, 3])
self.assertEqual(arrays.context_latents.shape, (4, 4, latent_dim))
self.assertEqual(arrays.history_actions.shape, (4, 3, 2))
self.assertEqual(arrays.rollout_actions.shape, (4, 5, 2))
self.assertEqual(arrays.target_latents.shape, (4, 5, latent_dim))
self.assertEqual(arrays.target_frame_indices.shape, (4, 5))
```

Verify the first sample exactly:

```python
np.testing.assert_array_equal(arrays.context_latents[0], latents[0:4])
np.testing.assert_array_equal(arrays.history_actions[0], visual["actions"][0:3])
np.testing.assert_array_equal(arrays.rollout_actions[0], visual["actions"][3:8])
np.testing.assert_array_equal(arrays.target_latents[0], latents[4:9])
np.testing.assert_array_equal(arrays.target_frame_indices[0], np.arange(4, 9))
```

- [x] **Step 2: Run the focused test and verify the missing import**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_data
```

Expected: import failure for `LatentRolloutArrays` or `build_latent_rollout_arrays`.

- [x] **Step 3: Implement immutable rollout arrays**

Add:

```python
@dataclass(frozen=True)
class LatentRolloutArrays:
    context_latents: np.ndarray
    history_actions: np.ndarray
    rollout_actions: np.ndarray
    target_latents: np.ndarray
    initial_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    episode_ids: np.ndarray
    start_step_ids: np.ndarray

    @property
    def count(self) -> int:
        return int(self.context_latents.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.rollout_actions.shape[1])
```

For each selected episode with `L` transitions, enumerate:

```python
for start_step in range(CONTEXT_FRAMES - 1, L - horizon + 1):
    context = frames[start_step - 3 : start_step + 1]
    history_actions = actions[start_step - 3 : start_step]
    rollout_actions = actions[start_step : start_step + horizon]
    targets = frames[start_step + 1 : start_step + horizon + 1]
```

Resolve these slices through canonical frame and transition offsets. Require finite arrays, `int64` index vectors, one shared positive horizon, and read-only owned copies. Reject an empty selected ID vector, duplicates, non-integer IDs, missing IDs, non-positive horizon, malformed latents, and the case where every selected episode is too short.

- [x] **Step 4: Run focused data tests**

Expected: all existing one-step tests and new rollout-array tests pass.

---

### Task 2: Add differentiable visual free-rollout loss

**Files:**
- Create: `src/world_model_lab/visual_recursive_training.py`
- Create: `tests/test_visual_recursive_training.py`

**Interfaces:**
- Consumes: one-step `LatentWindowArrays`, H-step `LatentRolloutArrays`, source normalizers, frozen decoder, and spatial CNN dimensions.
- Produces: `recursive_normalized_latents(...) -> torch.Tensor`.
- Produces: `recursive_rollout_objective(...) -> torch.Tensor`.
- Produces: `RecursiveDynamicsTrainingResult`.
- Produces: `train_recursive_latent_dynamics(...)`.

- [x] **Step 1: Write failing recursive-state tests**

Use identity normalizers and a toy dynamics module:

```python
class CurrentActionDynamics(torch.nn.Module):
    def forward(self, context, history, current):
        return context[:, -1] + current[:, :1]
```

With last context latent `1` and future action first components `2, 3, 4`, require differentiable predictions `3, 6, 10`. Assert the second prediction uses the first prediction rather than the true first target.

- [x] **Step 2: Write failing rollout-loss tests**

For a two-step batch whose true normalized targets equal the recursive toy predictions, require zero latent rollout loss when `changed_pixel_loss_weight=0`. Change only the second target and require a positive loss. Backpropagate a non-zero case and assert a learnable dynamics parameter receives a finite gradient.

- [x] **Step 3: Implement differentiable recursive prediction**

```python
def recursive_normalized_latents(model, context, history, actions):
    predictions = []
    for step in range(actions.shape[1]):
        prediction = model(context, history, actions[:, step])
        predictions.append(prediction)
        context = torch.cat((context[:, 1:], prediction[:, None]), dim=1)
        history = torch.cat(
            (history[:, 1:], actions[:, step : step + 1]),
            dim=1,
        )
    return torch.stack(predictions, dim=1)
```

Do not use `torch.no_grad()` or convert through NumPy.

- [x] **Step 4: Implement the per-step recursive objective**

Normalize target latents and actions before batching. Derive each step's changed mask from adjacent true frames:

```text
step 1: true initial frame versus true target 1
step h: true target h-1 versus true target h
```

For each step:

```python
latent_mse = mean((prediction - target) ** 2)
decoded = decoder.decode(prediction * latent_std + latent_mean)
changed_mae = changed_pixel_mae(decoded, target_frame, changed_mask)
step_loss = latent_mse + changed_pixel_loss_weight * changed_mae
```

Return the arithmetic mean across H steps. The decoder remains in the autograd graph, but all decoder parameters must have `requires_grad=False`.

- [x] **Step 5: Implement deterministic combined training**

Train fresh spatial CNN dynamics with:

```text
total = one_step_objective + rollout_loss_weight * H-step objective
```

Use all one-step train windows and all H-step train windows. Pair deterministic shuffled rollout batches with one-step batches using an independent `seed + 1` generator and modulo wraparound. Validate one-step and rollout datasets independently, choose the best model by total validation loss, and record total, one-step, and rollout histories in `RecursiveDynamicsTrainingResult`.

- [x] **Step 6: Run focused recursive-training tests**

Expected: recursion, loss, gradient, validation, and deterministic training tests pass.

---

### Task 3: Build the H5 training runner and checkpoint contract

**Files:**
- Create: `src/world_model_lab/train_visual_dynamics_recursive.py`
- Create: `tests/test_train_visual_dynamics_recursive.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: source objective-aligned checkpoint and visual dataset.
- Produces: `run_recursive_dynamics_training(...) -> dict[str, Any]`.
- Produces CLI: `world-model-train-visual-dynamics-recursive`.

- [x] **Step 1: Write failing end-to-end runner tests**

Create ten fixture episodes with eight transitions so every split has H5 windows. Save a spatial CNN source checkpoint, run one epoch, and verify:

```python
self.assertTrue(output_path.is_file())
self.assertTrue(preview_path.is_file())
self.assertEqual(loaded.training_config["dynamics_rollout_horizon"], 5)
self.assertEqual(loaded.training_config["dynamics_rollout_loss_weight"], 1.0)
self.assertTrue(loaded.training_config["dynamics_reinitialized"])
```

Require exact equality for source/candidate autoencoder tensors, normalizers, and split IDs. Verify both one-step and rollout split counts in the returned summary.

- [x] **Step 2: Write failing input and rollback tests**

Reject:

- dataset SHA mismatch;
- global or ConvGRU source checkpoint;
- `rollout_horizon <= 1`;
- negative or non-finite rollout weight;
- missing H5 train or validation windows;
- existing output/preview;
- preview publication failure, with rollback of the newly written checkpoint.

- [x] **Step 3: Implement the controlled runner**

Load and validate the source exactly like `run_frozen_decoder_dynamics_training`. Encode all frames once. Build:

```python
one_step_arrays[name] = build_latent_window_arrays(...)
rollout_arrays[name] = build_latent_rollout_arrays(
    dataset,
    source.split_episode_ids[name],
    latent_frames,
    horizon=rollout_horizon,
)
```

Train with the source seed and hyperparameters, evaluate held-out one-step metrics through `evaluate_latent_dynamics`, save the source autoencoder history unchanged, and publish a one-step preview. Add checkpoint metadata:

```python
{
    "source_checkpoint": ...,
    "source_checkpoint_sha256": ...,
    "autoencoder_frozen": True,
    "dynamics_reinitialized": True,
    "dynamics_loss": "one_step_plus_recursive_rollout",
    "dynamics_changed_pixel_loss_weight": 0.1,
    "dynamics_rollout_horizon": 5,
    "dynamics_rollout_loss_weight": 1.0,
}
```

- [x] **Step 4: Register and test the CLI**

Add:

```toml
world-model-train-visual-dynamics-recursive = "world_model_lab.train_visual_dynamics_recursive:main"
```

CLI arguments:

```text
--data
--source-checkpoint
--output
--preview
--changed-pixel-loss-weight
--rollout-horizon
--rollout-loss-weight
--dynamics-epochs
--dynamics-batch-size
```

- [x] **Step 5: Run focused runner tests**

Expected: all recursive runner and existing objective runner tests pass.

---

### Task 4: Execute and document the H5 controlled experiment

**Files:**
- Modify: `README.md`
- Create: `docs/experiments/2026-07-18-visual-recursive-rollout-training.md`
- Generate ignored: `artifacts/visual_latent_spatial8_objective_w01_h5.pt`
- Generate ignored: `artifacts/visual_latent_spatial8_objective_w01_h5_predictions.png`
- Generate ignored: `artifacts/diagnostics/visual-rollout-h5-comparison/`

**Interfaces:**
- Consumes: current objective-aligned checkpoint.
- Produces: H5 candidate checkpoint, preview, reproducible diagnostics, and pass/fail decision.

- [x] **Step 1: Record the pre-registered protocol before training**

Write the source digests, exact loss formula, fixed hyperparameters, reference metrics, and gates from this plan into the experiment document before running the candidate.

- [x] **Step 2: Run full pre-training verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v

PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

git diff --check
```

- [x] **Step 3: Train the H5 candidate**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_dynamics_recursive \
  --data data/visual_episodes.npz \
  --source-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01.pt \
  --output \
    artifacts/visual_latent_spatial8_objective_w01_h5.pt \
  --preview \
    artifacts/visual_latent_spatial8_objective_w01_h5_predictions.png \
  --changed-pixel-loss-weight 0.1 \
  --rollout-horizon 5 \
  --rollout-loss-weight 1.0 \
  --dynamics-epochs 50 \
  --dynamics-batch-size 256
```

- [x] **Step 4: Run the existing controlled diagnostic**

Treat the current objective-aligned H1 checkpoint as `baseline` and the new H5 checkpoint as `aligned` in the stable schema:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_rollout \
  --data data/visual_episodes.npz \
  --baseline-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01.pt \
  --aligned-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01_h5.pt \
  --output-dir artifacts/diagnostics/visual-rollout-h5-comparison \
  --horizons 1 5 10 \
  --windows-per-episode 8 \
  --counterfactual-seeds 0 1 2 3 4 5 6 7 8 9
```

- [x] **Step 5: Verify representation invariance and reproducibility**

Reload both checkpoints. Require exact autoencoder tensor equality, exact normalizer/split equality, dataset SHA equality, and finite stored metrics. Rerun diagnostics into a separate temporary directory and require byte-identical `manifest.json`, `metrics.json`, and PNG.

- [x] **Step 6: Apply the pre-registered gates**

Record:

- H1/H5/H10 teacher-forced and free latent MSE;
- H1/H5/H10 cumulative changed-pixel MAE;
- H1 and H5 pass/fail gates;
- H10 secondary result;
- action sensitivity changes;
- whether the current objective-aligned checkpoint or H5 candidate remains preferred.

- [x] **Step 7: Update README and run final verification**

Document the command, controlled result, limitations, and next decision. Rerun the full test suite, `compileall`, `git diff --check`, and staged diff checks.

- [x] **Step 8: Commit source, tests, plan, and experiment record**

Do not stage `.DS_Store`, data, checkpoints, previews, or ignored diagnostic artifacts.
