# Visual Multi-Step Rollout Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the latent-only and objective-aligned spatial CNN checkpoints under teacher-forced and recursive visual rollouts through horizon 10, then measure how much replacing only future action sequences changes each model's imagined trajectory.

**Architecture:** Build episode-safe visual rollout windows containing four initial latents, three aligned history actions, ten future actions, and ten true target latents/frames. Evaluate teacher forcing from true rolling contexts and free rollout from recursively predicted normalized latents. Generate counterfactual rollouts by applying deterministic no-fixed-point permutations to complete future action sequences while preserving each window's real visual context and past actions.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, unittest.

## Global Constraints

- Evaluate existing checkpoints only; do not mutate or retrain either model.
- Require identical spatial autoencoder weights, latent/action normalizers, test episode IDs, dataset SHA-256, latent dimensions, and CNN dynamics architecture.
- Use test episodes recorded in the checkpoints and reject any missing episode.
- Use `max_horizon=10`, snapshots `1, 5, 10`, and at most `8` evenly spaced windows per eligible episode.
- A valid start has three past transitions plus ten future transitions inside one episode.
- Aggregate window metrics inside each episode first, then average episodes equally.
- Report both teacher forcing and free rollout so compounding error is distinguishable from local one-step error.
- Report full-frame MSE, normalized latent MSE, adjacent-transition changed-pixel MAE, and cumulative changed-pixel MAE.
- Derive changed-pixel masks only from true RGB frames; do not read physical states, rewards, dones, or object labels.
- Counterfactual sequences replace only current and future actions; the initial context and three history actions remain recorded.
- Counterfactual divergence measures sensitivity, not correctness.
- Use Sattolo permutations for seeds `0` through `9` so every future action row comes from a different rollout window while internal action order stays intact.
- Publish `manifest.json`, `metrics.json`, and `visual_rollout_comparison.png` as one staged no-clobber directory bundle.

---

### Task 1: Define visual rollout windows and recursive state updates

**Files:**
- Create: `src/world_model_lab/diagnose_visual_rollout.py`
- Create: `tests/test_diagnose_visual_rollout.py`

**Interfaces:**
- Produces `VisualRolloutWindow`.
- Produces `VisualRolloutSelection`.
- Produces `select_visual_rollout_windows(...)`.
- Produces `rollout_normalized_latents(...)`.
- Produces `teacher_forced_normalized_latents(...)`.

- [x] **Step 1: Write failing episode-alignment tests**

Use a two-episode fixture with transition lengths `15` and `8`,
`max_horizon=5`, and `windows_per_episode=2`. Verify that the first episode
selects local current-action steps `3` and `10`, while the second selects
steps `3` and `3` only once after de-duplication:

```python
selection = select_visual_rollout_windows(
    dataset=visual,
    latent_frames=latents,
    selected_episode_ids=np.asarray([10, 11], dtype=np.int64),
    max_horizon=5,
    windows_per_episode=2,
)

self.assertEqual(
    [(window.episode_id, window.start_step) for window in selection.windows],
    [(10, 3), (10, 10), (11, 3)],
)
```

For every window assert:

```python
self.assertEqual(window.context_latents.shape, (4, latent_dim))
self.assertEqual(window.history_actions.shape, (3, 2))
self.assertEqual(window.future_actions.shape, (5, 2))
self.assertEqual(window.target_latents.shape, (5, latent_dim))
self.assertEqual(window.target_frame_indices.shape, (5,))
```

Also verify exact slices against `frame_offsets` and `transition_offsets`, and
that an episode with fewer than `3 + max_horizon` transitions is listed in
`skipped_episode_ids`.

- [x] **Step 2: Run the new tests and verify import failure**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_diagnose_visual_rollout
```

Expected: failure because `diagnose_visual_rollout` does not exist.

- [x] **Step 3: Implement immutable aligned windows**

For episode transition length `L`, valid current-action steps satisfy:

```python
first_step = 3
last_step = L - max_horizon
valid_count = last_step - first_step + 1
```

Select relative indices with:

```python
def _evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    if limit == 1:
        return np.asarray([0], dtype=np.int64)
    return np.rint(np.linspace(0, count - 1, limit)).astype(np.int64)
```

Resolve one start step `t` as:

```python
context_latents = latent_frames[frame_start + t - 3 : frame_start + t + 1]
history_actions = actions[action_start + t - 3 : action_start + t]
future_actions = actions[action_start + t : action_start + t + max_horizon]
target_latents = latent_frames[
    frame_start + t + 1 : frame_start + t + max_horizon + 1
]
```

- [x] **Step 4: Write failing recursion tests**

Use identity normalizers and a toy dynamics module that returns:

```python
context[:, -1] + current_action[:, :1]
```

With initial last latent `1` and future actions `2, 3, 4`, free rollout must
produce normalized predictions `3, 6, 10`. Teacher forcing must use each true
four-latent context instead of feeding those predictions back.

- [x] **Step 5: Implement batched teacher and free rollout**

Free rollout:

```python
prediction = dynamics(context, history, future[:, step])
predictions.append(prediction)
context = torch.cat((context[:, 1:], prediction[:, None, :]), dim=1)
history = torch.cat(
    (history[:, 1:], future[:, step : step + 1]),
    dim=1,
)
```

Teacher forcing constructs the true latent sequence by concatenating initial
context and targets, then uses `sequence[:, step : step + 4]` with the matching
three history actions at every step.

### Task 2: Add rollout metrics and counterfactual action sensitivity

**Files:**
- Modify: `src/world_model_lab/diagnose_visual_rollout.py`
- Modify: `tests/test_diagnose_visual_rollout.py`

**Interfaces:**
- Produces `evaluate_visual_rollout_model(...)`.
- Produces `_sattolo_permutation(count, seed)`.
- Produces `evaluate_counterfactual_sensitivity(...)`.

- [x] **Step 1: Test deterministic no-fixed-point action permutations**

```python
first = _sattolo_permutation(12, seed=7)
second = _sattolo_permutation(12, seed=7)
np.testing.assert_array_equal(first, second)
self.assertTrue(np.all(first != np.arange(12)))
self.assertEqual(sorted(first.tolist()), list(range(12)))
```

Reject `count < 2` and negative seeds.

- [x] **Step 2: Test episode-equal metric aggregation**

Construct two episodes where the first has two low-error windows and the
second has one high-error window. Assert the final metric is the mean of the
two episode means rather than the mean of all three windows.

For horizon `h`, define:

```text
latent MSE:
    mean over latent dimensions

full-frame MSE:
    mean over RGB values

transition changed mask:
    true frame[t+h-1] != true frame[t+h]

cumulative changed mask:
    true initial frame[t] != true target frame[t+h]
```

Changed-pixel errors are summed within each episode and divided by that
episode's changed RGB-value count before averaging eligible episodes.

- [x] **Step 3: Implement model rollout evaluation**

Return one record per dense step `1..max_horizon`:

```python
{
    "episodes": 18,
    "windows": 130,
    "teacher_forcing": {
        "normalized_latent_mse": ...,
        "pixel_mse": ...,
        "transition_changed_pixel_mae": ...,
        "cumulative_changed_pixel_mae": ...,
    },
    "free_rollout": {
        "normalized_latent_mse": ...,
        "pixel_mse": ...,
        "transition_changed_pixel_mae": ...,
        "cumulative_changed_pixel_mae": ...,
    },
}
```

- [x] **Step 4: Implement counterfactual sensitivity**

For each seed, permute complete `[H, 2]` future action rows, run a second free
rollout, and compare it with the recorded-action free rollout:

```text
normalized latent RMS
decoded full-frame pixel MSE
decoded full-frame pixel MAE
```

Aggregate windows inside episodes before averaging episodes. Across seeds,
report mean and sample standard deviation for every dense horizon.

- [x] **Step 5: Run focused tests**

Expected: all selection, recursion, aggregation, and permutation tests pass.

### Task 3: Build the comparison runner and atomic artifact bundle

**Files:**
- Modify: `src/world_model_lab/diagnose_visual_rollout.py`
- Modify: `tests/test_diagnose_visual_rollout.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces `run_visual_rollout_diagnostics(...) -> dict[str, Any]`.
- Produces CLI `world-model-diagnose-visual-rollout`.

- [x] **Step 1: Write failing compatibility tests**

Create two tiny checkpoint paths from the same payload and verify the runner
accepts them. Then independently alter each of these and require a named
`ValueError` before evaluation:

- autoencoder state dict;
- latent normalizer;
- action normalizer;
- test split;
- dataset SHA-256;
- dynamics architecture.

- [x] **Step 2: Implement strict checkpoint compatibility**

Require:

```python
isinstance(autoencoder, SpatialConvAutoencoder)
isinstance(dynamics, SpatialLatentDynamicsCNN)
torch.equal(left_autoencoder_tensor, right_autoencoder_tensor)
np.array_equal(left_normalizer.mean, right_normalizer.mean)
np.array_equal(left_normalizer.std, right_normalizer.std)
np.array_equal(left_test_ids, right_test_ids)
```

Verify the current dataset file hash against both checkpoints.

- [x] **Step 3: Build metrics and comparison payloads**

`metrics.json` schema version 1 contains:

```text
models.baseline.steps[1..10]
models.aligned.steps[1..10]
models.<name>.counterfactual.steps[1..10]
comparison.aligned_minus_baseline[1..10]
snapshots[1, 5, 10]
```

The comparison reports finite absolute and relative deltas for recorded free
rollout metrics plus aligned-to-baseline counterfactual sensitivity ratios.

- [x] **Step 4: Plot four diagnostic panels**

Create `visual_rollout_comparison.png`:

1. normalized latent MSE, teacher versus free;
2. cumulative changed-pixel MAE, teacher versus free;
3. counterfactual normalized latent RMS with seed standard deviation;
4. counterfactual decoded pixel MSE with seed standard deviation.

Use one color per model and dashed/solid line styles for teacher/free.

- [x] **Step 5: Publish one staged bundle**

Refuse a non-empty output directory. Write `manifest.json`, `metrics.json`,
and the PNG into a sibling temporary directory; rename it to the final output
only after all three files succeed. Remove staging on any exception.

- [x] **Step 6: Register and test the CLI**

Add:

```toml
world-model-diagnose-visual-rollout = "world_model_lab.diagnose_visual_rollout:main"
```

CLI arguments:

```text
--data
--baseline-checkpoint
--aligned-checkpoint
--output-dir
--horizons
--windows-per-episode
--counterfactual-seeds
```

### Task 4: Execute and document the controlled comparison

**Files:**
- Modify: `README.md`
- Create: `docs/experiments/2026-07-17-visual-multistep-rollout-diagnostics.md`
- Generate ignored: `artifacts/diagnostics/visual-rollout-objective-comparison/`

**Interfaces:**
- Consumes the latent-only and objective-aligned checkpoints.
- Produces a reproducible metrics bundle and next-stage decision.

- [x] **Step 1: Record protocol before execution**

Use:

```text
baseline: artifacts/visual_latent_spatial8.pt
aligned:  artifacts/visual_latent_spatial8_objective_w01.pt
horizons: 1, 5, 10
windows per eligible episode: 8
counterfactual seeds: 0..9
```

Questions fixed before the run:

1. Does objective alignment reduce free-rollout cumulative changed-pixel MAE
   at horizons 1, 5, and 10?
2. How quickly does free rollout separate from teacher forcing?
3. Does counterfactual action divergence grow with horizon?
4. Is the aligned model more action-sensitive than the latent-only model?

- [x] **Step 2: Run full pre-experiment verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v

PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

git diff --check
```

- [x] **Step 3: Execute the diagnostic**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_rollout \
  --data data/visual_episodes.npz \
  --baseline-checkpoint artifacts/visual_latent_spatial8.pt \
  --aligned-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --output-dir \
    artifacts/diagnostics/visual-rollout-objective-comparison \
  --horizons 1 5 10 \
  --windows-per-episode 8 \
  --counterfactual-seeds 0 1 2 3 4 5 6 7 8 9
```

- [x] **Step 4: Reload the JSON and verify exact reproducibility**

Re-run the evaluator in memory and require the regenerated metrics payload to
equal the stored `metrics.json`. Verify all three artifact SHA-256 values.

- [x] **Step 5: Record bounded conclusions**

State separately:

- recorded-action rollout accuracy;
- teacher/free compounding gap;
- counterfactual sensitivity;
- why sensitivity does not prove counterfactual correctness;
- whether visual multi-step quality is sufficient for MPC.

- [x] **Step 6: Run final verification and commit**

Run the full test suite, `compileall`, staged diff checks, and checkpoint
compatibility checks. Commit source, tests, plan, README, and experiment
document without adding ignored artifacts or `.DS_Store`.
