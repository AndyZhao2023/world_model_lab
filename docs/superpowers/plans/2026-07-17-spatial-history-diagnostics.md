# Spatial History Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the frozen spatial dynamics model uses ordered visual motion history beyond the current latent frame.

**Architecture:** Extend the held-out evaluator with two context-only variants while preserving recorded actions. `repeat_last_context` replaces all four latent frames with the final context latent; `reverse_history_context` reverses only the first three latent frames and keeps the final latent as the residual anchor.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, unittest.

## Global Constraints

- Do not retrain or mutate the checkpoint.
- Preserve every existing metric and action ablation.
- Keep the final context latent at index `-1` for both history variants.
- Keep recorded actions unchanged so only visual context changes.
- Do not read states, rewards, dones, or labels.
- Report normalized latent MSE, full-frame MSE, and changed-pixel MAE.

---

### Task 1: History-sensitive regression tests

**Files:**
- Modify: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Expects: `repeat_last_context_*` and `reverse_history_context_*` metrics from `evaluate_latent_dynamics`.

- [ ] **Step 1: Write a failing toy-dynamics test**

Use a dynamics module whose prediction is:

```python
return context[:, -1] + context[:, -2] - context[:, -3]
```

Construct chronological context `[0.1, 0.2, 0.4, 0.7]` and target `0.9`.
Assert recorded normalized latent MSE is zero, while repeat-last and
reverse-history MSE are positive.

- [ ] **Step 2: Confirm the metric keys are absent**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_visual_latent_model.VisualLatentDynamicsTrainingTest -v
```

Expected: failure because the two context-ablation metric prefixes do not exist.

---

### Task 2: Context variants in the evaluator

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Test: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Produces: `_arrays_with_replaced_context(arrays, context_latents) -> LatentWindowArrays`.
- Extends: `evaluate_latent_dynamics(..., include_context_ablations=True)`.

- [ ] **Step 1: Implement immutable context replacement**

Reuse targets, actions, frame indices, episode IDs, and step IDs through a new
`LatentWindowArrays`; replace only `context_latents`.

- [ ] **Step 2: Build both context variants**

```python
repeat_last = np.repeat(arrays.context_latents[:, -1:, :], 4, axis=1)
reverse_history = arrays.context_latents.copy()
reverse_history[:, :-1] = arrays.context_latents[:, :-1][:, ::-1, :]
```

- [ ] **Step 3: Evaluate without nested ablations**

For each variant call:

```python
evaluate_latent_dynamics(
    ...,
    include_action_ablations=False,
    include_context_ablations=False,
)
```

Copy only `normalized_latent_mse`, `pixel_mse`, and `changed_pixel_mae` under
the two stable prefixes. Existing action-ablation recursive calls must also
disable context ablations.

- [ ] **Step 4: Run all visual latent training tests**

Expected: old metrics, action diagnostics, and new history diagnostics all pass.

---

### Task 3: Frozen-checkpoint experiment

**Files:**
- Create: `docs/experiments/2026-07-17-spatial-history-diagnostics.md`
- Modify: `docs/experiments/2026-07-17-spatial-dynamics-diagnostics.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `artifacts/visual_latent_spatial8.pt`.
- Produces: one directly comparable history-ablation table.

- [ ] **Step 1: Rebuild checkpoint test windows and evaluate**

Use the checkpoint's test episode IDs, latent/action normalizers, batch size
`256`, and action shuffle seed `0`. Do not retrain.

- [ ] **Step 2: Interpret the causal boundary**

If repeat-last degrades, the model uses information older than the current
frame. If reverse-history degrades, ordered history matters. Because actions
remain recorded while visual history changes, degradation may include
cross-modal inconsistency and must not be attributed solely to frame order.

- [ ] **Step 3: Run complete verification**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests

git diff --check
```

Expected: compilation and all tests pass with no whitespace errors.
