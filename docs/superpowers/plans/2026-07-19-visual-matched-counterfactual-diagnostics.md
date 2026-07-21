# Visual Matched Counterfactual Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Execute complete replacement action rows from identical held-out visual histories in `CarEnv`, then compare the current H1 source and H5 recursive candidate against the resulting matched counterfactual latent and frame targets.

**Architecture:** Reuse the existing H10-capable, episode-equal visual rollout window selector and Sattolo action-row permutations. A focused simulator module will execute each donated action row from the recipient window's exact aligned physical state and return immutable targets plus a post-terminal validity mask. A separate diagnostic runner will encode those true frames with the shared frozen autoencoder, evaluate both compatible checkpoints under the exact applied actions, aggregate direct and action-effect errors by episode and seed, and atomically publish a manifest, metrics, and plot.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Pillow renderer, Matplotlib, unittest.

## Global Constraints

- Use `data/visual_episodes.npz` with SHA-256 `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`.
- Compare source `artifacts/visual_latent_spatial8_objective_w01.pt` with SHA-256 `5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369`.
- Compare candidate `artifacts/visual_latent_spatial8_objective_w01_h5.pt` with SHA-256 `4764d89d7399694574ad91a6703c90872d9c0f7c1b6e7743572dc58d3ff0e424`.
- Require exact autoencoder, latent/action normalizer, test split, dataset SHA-256, latent layout, and CNN architecture equality.
- Reuse horizons `1`, `5`, and `10`, at most eight evenly spaced H10-capable windows per test episode, and seeds `0..9`.
- For each seed, use the existing no-fixed-point Sattolo permutation to donate one complete current-plus-future action row to each recipient window.
- Recreate each branch with `CarEnv(initial_state=aligned_state_at_anchor)` and the schema-v1 default scene.
- Feed the environment's actual clipped actions to both models; retain requested actions only for provenance.
- Count the terminal transition as valid and mask every later step. Never silently drop a terminal branch or use an invalid target in a metric.
- Aggregate valid windows inside each episode, then episodes equally, then seeds by mean and sample standard deviation.
- Do not update model weights, refit normalizers, change splits, sweep action seeds, or connect either model to MPC.
- Refuse to overwrite a non-empty output directory and publish all diagnostic files atomically.

---

### Task 1: Build deterministic matched simulator branches

**Files:**
- Create: `src/world_model_lab/visual_counterfactual_data.py`
- Create: `tests/test_visual_counterfactual_data.py`

**Interfaces:**
- Consumes: validated visual dataset, selected `VisualRolloutWindow` objects, and one no-fixed-point donor permutation.
- Produces: immutable `MatchedCounterfactualBatch`.
- Produces: `build_matched_counterfactual_batch(dataset, windows, donor_window_indices)`.

- [x] **Step 1: Write the failing exact-branch test**

Build two physically valid windows with a two-step horizon. Give the first
window a left-steering action row and the second a right-steering row. Use
donor permutation `[1, 0]`. Independently step:

```python
manual = CarEnv(initial_state=dataset["states"][first.initial_frame_index])
expected = [
    manual.step(*windows[1].future_actions[step])[0]
    for step in range(2)
]
```

Require:

```python
np.testing.assert_allclose(batch.true_states[0], expected)
np.testing.assert_array_equal(batch.donor_window_indices, [1, 0])
np.testing.assert_array_equal(batch.valid_steps, True)
np.testing.assert_array_equal(
    batch.applied_actions,
    batch.requested_actions,
)
```

Also require every stored NumPy array to be read-only.

- [x] **Step 2: Run the focused test and verify the missing import**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_counterfactual_data
```

Expected: import failure for `MatchedCounterfactualBatch` or
`build_matched_counterfactual_batch`.

- [x] **Step 3: Implement the immutable branch contract**

Create:

```python
@dataclass(frozen=True)
class MatchedCounterfactualBatch:
    recipient_episode_ids: np.ndarray       # [N], int64
    recipient_start_steps: np.ndarray       # [N], int64
    donor_window_indices: np.ndarray        # [N], int64 permutation
    requested_actions: np.ndarray           # [N, H, 2], float64
    applied_actions: np.ndarray             # [N, H, 2], float64
    true_states: np.ndarray                 # [N, H, 4], float64
    true_frames: np.ndarray                 # [N, H, 64, 64, 3], uint8
    valid_steps: np.ndarray                 # [N, H], bool
    terminal_steps: np.ndarray              # [N], int64; -1 or 1..H
    terminal_reasons: np.ndarray            # [N], unicode
```

Require one shared positive horizon, a complete in-range donor permutation,
no fixed points, finite actions, finite valid states, zero-filled invalid
frames, and terminal metadata aligned with the validity mask.

- [x] **Step 4: Implement deterministic branch execution**

For recipient `i`, take:

```python
initial_state = dataset["states"][window.initial_frame_index]
requested = windows[donor_window_indices[i]].future_actions
env = CarEnv(initial_state=initial_state)
```

Clip every requested action with the environment limits to form the complete
applied row. Step until terminal. Render each valid returned state with
`render_observation(state, scene=scene_from_env(env))`. Mark the terminal
transition valid, record `goal`, `collision`, `out_of_bounds`, or
`time_limit`, and leave later targets invalid.

- [x] **Step 5: Add boundary tests and run the focused suite**

Reject empty windows, inconsistent horizons, malformed/out-of-range/duplicate
permutations, fixed points, invalid frame indices, and non-finite actions.
Create a branch that leaves the world on step one and require:

```python
np.testing.assert_array_equal(batch.valid_steps[0], [True, False])
self.assertEqual(batch.terminal_steps[0], 1)
self.assertEqual(batch.terminal_reasons[0], "out_of_bounds")
```

Expected: all branch tests pass.

---

### Task 2: Add masked matched-accuracy and action-effect metrics

**Files:**
- Create: `src/world_model_lab/diagnose_visual_counterfactual.py`
- Create: `tests/test_diagnose_visual_counterfactual.py`

**Interfaces:**
- Consumes: one `MatchedCounterfactualBatch`, factual predictions/targets, matched counterfactual predictions/targets, episode IDs, and one seed.
- Produces: `summarize_matched_counterfactual_predictions(...)`.
- Produces: `aggregate_counterfactual_seed_records(...)`.
- Produces: source/candidate comparison and pre-registered decision records.

- [x] **Step 1: Write failing masked aggregation tests**

Use three windows with episode IDs `[10, 10, 11]`, two rollout steps, and a
mask where one episode-10 window terminates after step one. Set scalar latent
errors so episode 10 has two valid windows and episode 11 has one. Require the
metric to average the two episode-10 windows first, then average episode 10
and 11 equally. Require step-two coverage to exclude only the invalid branch.

- [x] **Step 2: Write failing action-effect tests**

For each valid element calculate:

```text
predicted effect = predicted counterfactual - predicted factual
true effect      = true counterfactual - true factual
effect error     = MSE(predicted effect, true effect)
```

Use a toy case where direct counterfactual error is non-zero but the predicted
and true effects are identical, and require
`normalized_latent_effect_mse == 0`.

- [x] **Step 3: Implement direct and effect curves**

Produce these per-step values:

```text
normalized_latent_mse
pixel_mse
transition_changed_pixel_mae
cumulative_changed_pixel_mae
normalized_latent_effect_mse
pixel_effect_mse
```

Apply the validity mask before each window/episode aggregation. For changed
pixels, compare true adjacent counterfactual frames for transition masks and
the common initial frame for cumulative masks. Sum changed-pixel numerator
and denominator inside each episode before averaging episodes.

- [x] **Step 4: Implement seed aggregation and comparison**

Each model/step/metric record is:

```python
{
    "mean": float,
    "sample_std": float,
}
```

The comparison record is candidate minus source:

```python
{
    "absolute": candidate - source,
    "relative_percent": 100 * (candidate - source) / source,
}
```

Reject missing steps, non-finite values, or inconsistent seeds. For an exact
zero source reference, report zero relative change when the candidate is also
zero and use the existing epsilon fallback otherwise.

- [x] **Step 5: Encode the pre-registered decision**

At decision horizon H5 require:

```text
candidate matched normalized latent MSE < source
candidate matched cumulative changed-pixel MAE < source
candidate normalized latent effect MSE < source
```

At H1 require:

```text
candidate matched normalized latent MSE <= 1.10 * source
candidate matched cumulative changed-pixel MAE <= 1.05 * source
```

Return every gate with source, limit, candidate, and pass/fail. The candidate
passes only if all five gates pass.

- [x] **Step 6: Run focused metric tests**

Expected: direct accuracy, action effect, masks, episode weighting, seed
aggregation, comparison, and decision tests pass.

---

### Task 3: Build the atomic matched counterfactual diagnostic runner

**Files:**
- Modify: `src/world_model_lab/diagnose_visual_counterfactual.py`
- Modify: `tests/test_diagnose_visual_counterfactual.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: visual dataset, compatible source/candidate checkpoints, horizons, window cap, seeds, and decision horizon.
- Produces: `run_visual_counterfactual_diagnostics(...) -> dict[str, object]`.
- Produces CLI: `world-model-diagnose-visual-counterfactual`.

- [x] **Step 1: Write the failing end-to-end runner test**

Create ten deterministic, physically valid six-transition episodes. Save two
identical spatial CNN checkpoints. Run horizons `(1, 2)`, two windows per
episode, seeds `(0, 1)`, and decision horizon `2`. Require:

```text
manifest.json
metrics.json
matched_counterfactual_comparison.png
```

Require zero candidate-minus-source deltas, every Sattolo permutation to have
no fixed points, a complete coverage record, oracle reconstruction metrics,
and a failed strict-improvement decision for identical models.

- [x] **Step 2: Write failing publication and compatibility tests**

Reject incompatible autoencoder weights, normalizers, test splits, dataset
hashes, non-spatial CNN checkpoints, missing decision horizon, invalid seeds,
and a non-empty output directory. Patch plotting to raise and require the
staging directory to be removed without publishing a partial output.

- [x] **Step 3: Implement one shared target-generation pass**

Reuse:

```python
select_visual_rollout_windows(...)
_sattolo_permutation(...)
rollout_normalized_latents(...)
_decode_normalized_rollout_latents(...)
_validate_visual_checkpoint_pair(...)
```

Encode valid true branch frames once with the shared baseline autoencoder.
Fill invalid target positions with the latent normalizer mean so their
normalized value is zero, and exclude them with `valid_steps`.

- [x] **Step 4: Evaluate both models on exact applied actions**

Compute factual predictions once per model. For every seed, generate one
shared matched branch batch, then evaluate source and candidate using
`batch.applied_actions`. Aggregate model metrics, shared oracle decoder
metrics, terminal coverage, clipping counts, and paired comparisons.

- [x] **Step 5: Publish deterministic artifacts atomically**

Write:

```text
manifest.json
metrics.json
matched_counterfactual_comparison.png
```

The manifest records dataset/checkpoint hashes, selected windows, seeds,
donor permutations, environment reconstruction semantics, terminal masking,
aggregation, clipping count, and gates. Stage in a sibling temporary
directory, rename only after every file succeeds, and remove staging on any
exception.

- [x] **Step 6: Register and test the CLI**

Add:

```toml
world-model-diagnose-visual-counterfactual = "world_model_lab.diagnose_visual_counterfactual:main"
```

CLI arguments:

```text
--data
--source-checkpoint
--candidate-checkpoint
--output-dir
--horizons
--windows-per-episode
--counterfactual-seeds
--decision-horizon
```

- [x] **Step 7: Run focused runner tests**

Expected: all matched counterfactual data and diagnostic tests pass.

---

### Task 4: Execute and document the controlled experiment

**Files:**
- Modify: `README.md`
- Create: `docs/experiments/2026-07-19-visual-matched-counterfactual-diagnostics.md`
- Generate ignored: `artifacts/diagnostics/visual-matched-counterfactual-h5-comparison/`

**Interfaces:**
- Consumes: the fixed visual dataset and H1/H5 checkpoints.
- Produces: reproducible matched counterfactual evidence and a model decision.

- [x] **Step 1: Record the protocol and gates before model execution**

Write the fixed hashes, action donation semantics, simulator reconstruction,
terminal mask, metrics, aggregation, and five decision gates into the
experiment document. Leave results marked pending.

- [x] **Step 2: Run pre-experiment verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests

PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

git diff --check
```

- [x] **Step 3: Execute the registered diagnostic once**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_counterfactual \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --candidate-checkpoint artifacts/visual_latent_spatial8_objective_w01_h5.pt \
  --output-dir \
    artifacts/diagnostics/visual-matched-counterfactual-h5-comparison \
  --horizons 1 5 10 \
  --windows-per-episode 8 \
  --counterfactual-seeds 0 1 2 3 4 5 6 7 8 9 \
  --decision-horizon 5
```

- [x] **Step 4: Verify reproducibility independently**

Rerun into a new temporary directory. Require byte-identical manifest,
metrics, and PNG files. Require finite JSON numbers, exact source/candidate
representation equality, and matching dataset/checkpoint hashes.

- [x] **Step 5: Apply the pre-registered gates**

Record H1/H5/H10 direct matched errors, action-effect errors, oracle decoder
metrics, valid coverage, terminal reasons, action clipping count, all five
gate results, and whether source or candidate remains preferred.

- [x] **Step 6: Update README and run final verification**

Document what matched counterfactual accuracy adds beyond sensitivity, the
command, result, limitations, and next decision. Run the full test suite,
`compileall`, `git diff --check`, and staged diff checks.

- [x] **Step 7: Commit only source, tests, plan, and documentation**

Do not stage `.DS_Store`, datasets, checkpoints, plots, metrics, manifests, or
other ignored artifacts.
