# Spatial Dynamics Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the trained spatial dynamics model improves on reusing the last encoded frame and whether its predictions depend on aligned actions.

**Architecture:** Extend the existing held-out evaluator without changing model weights, training, checkpoint format, or prior metric definitions. Decode the last context latent as a representation-matched no-dynamics baseline, then evaluate the same frozen model with action inputs set to their training mean and with complete four-action rows deterministically permuted across test windows.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, unittest.

## Global Constraints

- Preserve every existing metric name and numeric calculation.
- Do not retrain or mutate the spatial checkpoint.
- Do not read physical states, rewards, dones, or labels.
- Define zero normalized action as the training-set action mean, not raw physical zero.
- Shuffle each window's three historical actions and current action together using `np.random.default_rng(action_shuffle_seed).permutation`.
- Use action shuffle seed `0` for the recorded experiment.
- Keep raw-frame `copy_last_*` metrics distinct from decoded-last-latent metrics.

---

### Task 1: Representation-matched no-dynamics baseline

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Test: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Produces: `decoded_last_latent_pixel_mse`, `decoded_last_latent_pixel_mae`, and `decoded_last_latent_changed_pixel_mae` in `evaluate_latent_dynamics`.

- [ ] **Step 1: Write the failing numeric test**

Use a scalar decoder where the last normalized context latent is `0.25` and the target frame is all ones. Assert:

```python
assert metrics["decoded_last_latent_pixel_mse"] == 0.75 ** 2
assert metrics["decoded_last_latent_pixel_mae"] == 0.75
assert metrics["decoded_last_latent_changed_pixel_mae"] == 0.75
```

- [ ] **Step 2: Run the focused test and confirm key failure**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_visual_latent_model.VisualLatentDynamicsTrainingTest -v
```

Expected: failure because the decoded-last metric keys are absent.

- [ ] **Step 3: Decode the final context latent**

Inside each evaluation batch, normalize `arrays.context_latents[start:stop, -1]`, decode it with `_decode_normalized_latents`, and accumulate errors against the same target frame and changed-pixel mask used by world, Oracle, and raw copy-last metrics.

- [ ] **Step 4: Verify the numeric test passes**

Expected: the three decoded-last metrics exactly match the hand-calculated values.

---

### Task 2: Mean-action and shuffled-action ablations

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Test: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Produces: `_arrays_with_replaced_actions(arrays, history_actions, current_actions) -> LatentWindowArrays`.
- Extends: `evaluate_latent_dynamics(..., action_shuffle_seed=0, include_action_ablations=True)`.
- Produces: normalized latent MSE, pixel MSE, and changed-pixel MAE under `mean_action_ablation_*` and `shuffled_action_ablation_*` prefixes.

- [ ] **Step 1: Write failing deterministic-ablation tests**

Create at least four windows and an action-sensitive test dynamics module. Assert that:

```python
first = evaluate_latent_dynamics(..., action_shuffle_seed=7)
second = evaluate_latent_dynamics(..., action_shuffle_seed=7)
assert first == second
assert first["mean_action_ablation_normalized_latent_mse"] != first["normalized_latent_mse"]
assert first["shuffled_action_ablation_normalized_latent_mse"] != first["normalized_latent_mse"]
```

Also assert negative shuffle seeds are rejected.

- [ ] **Step 2: Verify the focused tests fail**

Expected: failure because action ablation parameters and keys do not exist.

- [ ] **Step 3: Build immutable action variants**

For mean-action evaluation, fill both action arrays from `action_normalizer.mean`. For shuffled evaluation, apply one deterministic row permutation to both arrays:

```python
permutation = np.random.default_rng(action_shuffle_seed).permutation(arrays.count)
history = arrays.history_actions[permutation]
current = arrays.current_actions[permutation]
```

Reuse every non-action field unchanged through a fresh `LatentWindowArrays` instance.

- [ ] **Step 4: Evaluate variants without nested ablations**

Call `evaluate_latent_dynamics(..., include_action_ablations=False)` for each variant, then copy only:

```python
normalized_latent_mse
pixel_mse
changed_pixel_mae
```

under stable ablation prefixes. Existing calls keep the default seed `0` and receive all diagnostics.

- [ ] **Step 5: Run all visual latent training tests**

Expected: old metrics remain unchanged and new diagnostics are deterministic and finite.

---

### Task 3: Existing-checkpoint experiment and verification

**Files:**
- Create: `docs/experiments/2026-07-17-spatial-dynamics-diagnostics.md`
- Modify: `docs/experiments/2026-07-17-spatial-latent-world-model.md`

**Interfaces:**
- Consumes: `artifacts/visual_latent_spatial8.pt` and `data/visual_episodes.npz`.
- Produces: diagnostic comparison without checkpoint mutation or retraining.

- [ ] **Step 1: Reconstruct held-out latent windows**

Load the checkpoint and visual dataset, verify the dataset SHA-256, encode all frames with the frozen autoencoder, rebuild only the checkpoint's test episode windows, and call `evaluate_latent_dynamics` with batch size `256` and shuffle seed `0`.

- [ ] **Step 2: Record the decision table**

Compare world, decoded-last, mean-action, shuffled-action, Oracle, and raw copy-last changed-pixel MAE. Interpret action dependence only from observed deltas; do not assume a small delta means actions are universally unnecessary.

- [ ] **Step 3: Run full verification**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests

git diff --check
```

Expected: compilation and all tests pass, with no whitespace errors.

