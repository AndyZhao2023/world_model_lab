# Multi-Seed Stability Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Track every checkbox and stop on unexplained test failures.

**Goal:** Build a reproducible paired H1/H10 experiment that fixes the episode split, varies five training seeds, aggregates dense rollout errors, and determines whether the observed heading tradeoff is stable.

**Architecture:** Extend the existing training boundary with a backward-compatible `split_seed`, then add one orchestration module that calls `run_training()` and `run_diagnostics()` for every paired run. Keep aggregation and plotting as pure helpers, verify split IDs and dataset hashes before publishing top-level results, and store all generated output under ignored `artifacts/` paths.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, CSV/JSON, argparse, unittest.

---

## Task 1: Decouple the episode split seed from the training seed

**Files:**

- Modify: `src/world_model_lab/train_world_model.py`
- Modify: `tests/test_train_world_model.py`

- [ ] **Step 1: Add regression tests for explicit and legacy split behavior**

Append two tests to `TrainWorldModelTest`. Use `make_sequence_dynamics(episodes=10, steps=12)`, save the arrays to a temporary NPZ, and train one epoch with `hidden_size=8`, `batch_size=32`, and `rollout_horizon=1`.

```python
def _save_sequence_dataset(path: Path) -> None:
    states, actions, next_states, episode_ids, step_ids = (
        make_sequence_dynamics(episodes=10, steps=12)
    )
    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
    )

def test_explicit_split_seed_keeps_episode_split_fixed_across_training_seeds(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_path = root / "transitions.npz"
        _save_sequence_dataset(data_path)
        checkpoints = []
        for seed in (3, 9):
            checkpoint = root / f"seed-{seed}.pt"
            run_training(
                data_path=data_path,
                output_path=checkpoint,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                seed=seed,
                split_seed=5,
            )
            checkpoints.append(load_checkpoint(checkpoint))

    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(
            checkpoints[0].split_episode_ids[name],
            checkpoints[1].split_episode_ids[name],
        )
    self.assertEqual(checkpoints[0].training_config["split_seed"], 5)
    self.assertEqual(checkpoints[1].training_config["split_seed"], 5)

def test_omitted_split_seed_preserves_seed_based_split(self):
    # Train once with split_seed omitted and once with split_seed=seed.
    # Assert all three split arrays are identical and both summaries report
    # the same effective split seed.
```

The second test must use the same training seed for both runs and assert `summary["split_seed"] == seed` for both. It protects checkpoint compatibility and the existing CLI behavior.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_train_world_model.TrainWorldModelTest.test_explicit_split_seed_keeps_episode_split_fixed_across_training_seeds \
  tests.test_train_world_model.TrainWorldModelTest.test_omitted_split_seed_preserves_seed_based_split
```

Expected: both tests fail because `run_training()` does not accept `split_seed` and the summary/checkpoint does not expose it.

- [ ] **Step 3: Implement the minimal backward-compatible seed boundary**

Change the signature and compute the effective split seed once:

```python
def run_training(
    *,
    data_path: Path | str,
    output_path: Path | str,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    split_seed: int | None = None,
    rollout_horizon: int = 1,
    rollout_loss_weight: float = 1.0,
) -> dict[str, Any]:
    effective_split_seed = seed if split_seed is None else split_seed
    splits = split_episode_ids(episode_ids, seed=effective_split_seed)
```

Add `"split_seed": effective_split_seed` to both `training_config` and the returned summary. Keep `seed` as the only seed passed to `train_model()`.

Add the CLI flag and forward it:

```python
parser.add_argument("--split-seed", type=int)
summary = run_training(
    data_path=args.data,
    output_path=args.output,
    hidden_size=args.hidden_size,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    seed=args.seed,
    split_seed=args.split_seed,
    rollout_horizon=args.rollout_horizon,
    rollout_loss_weight=args.rollout_loss_weight,
)
```

- [ ] **Step 4: Run focused and affected tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_train_world_model
```

Expected: PASS. Existing callers that omit `split_seed` retain their previous split.

- [ ] **Step 5: Commit**

```bash
git add src/world_model_lab/train_world_model.py tests/test_train_world_model.py
git commit -m "feat: separate training and split seeds"
```

## Task 2: Add deterministic paired-curve aggregation and plotting

**Files:**

- Create: `src/world_model_lab/multiseed_experiment.py`
- Create: `tests/test_multiseed_experiment.py`

- [ ] **Step 1: Write synthetic aggregation tests**

Create a tiny diagnostics fixture with two seeds and two rollout steps. Match the existing metrics schema exactly:

```python
def make_metrics(position, heading, velocity, total):
    return {
        "schema_version": 2,
        "rollout": {
            "curves": {
                "steps": [1, 2],
                "free_rollout": {
                    "physical": {
                        "position": position,
                        "heading_degrees": heading,
                        "velocity": velocity,
                    },
                    "normalized_mse": {"total": total},
                },
            }
        },
    }
```

Build records with these position curves: seed 0 H1 `[1.0, 2.0]`, H10 `[0.8, 2.5]`; seed 1 H1 `[1.2, 2.2]`, H10 `[1.0, 2.3]`. Use heading curves where seed 0 first regresses at step 2 and seed 1 never regresses: seed 0 H1 `[1.0, 2.0]`, H10 `[0.5, 2.5]`; seed 1 H1 `[1.0, 2.0]`, H10 `[0.5, 1.5]`. Supply finite two-step velocity and normalized-total curves as well. Assert:

```python
summary = build_experiment_summary(records, snapshot_horizons=(1, 2))

position = summary["metrics"]["position"]
np.testing.assert_allclose(position["h1"]["mean"], [1.1, 2.1])
np.testing.assert_allclose(position["h1"]["std"], [np.sqrt(0.02), np.sqrt(0.02)])
np.testing.assert_allclose(position["paired_delta"]["mean"], [-0.2, 0.3])
np.testing.assert_allclose(
    position["paired_delta"]["std"], [0.0, np.sqrt(0.08)], atol=1e-15
)
self.assertEqual(position["paired_delta"]["improved_seed_count"][1], 0)
self.assertEqual(position["paired_delta"]["worse_seed_count"][1], 2)
self.assertEqual(summary["per_seed"]["0"]["first_heading_regression_step"], 2)
self.assertIsNone(summary["per_seed"]["1"]["first_heading_regression_step"])
```

Use tolerance `1e-12`: a paired delta below `-1e-12` is improved, above `1e-12` is worse, otherwise equal. Require at least two unique, non-negative training seeds and use sample standard deviation (`ddof=1`). Also test rejection of duplicate seeds and inconsistent step arrays.

Add output tests:

```python
write_summary_csv(summary, csv_path)
self.assertEqual(len(csv_path.read_text().splitlines()), 17)  # header + 2*4*2

plot_multiseed_comparison(summary, plot_path)
self.assertEqual(plot_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
```

- [ ] **Step 2: Run the new test module and confirm RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_multiseed_experiment
```

Expected: import failure because `world_model_lab.multiseed_experiment` does not exist.

- [ ] **Step 3: Implement pure metric extraction and statistics helpers**

Define these constants and helpers in `multiseed_experiment.py`:

```python
METRIC_NAMES = ("position", "heading_degrees", "velocity", "normalized_total")
COMPARISON_TOLERANCE = 1e-12

def _metric_curve(metrics: dict[str, Any], name: str) -> np.ndarray:
    curves = metrics["rollout"]["curves"]["free_rollout"]
    if name == "normalized_total":
        values = curves["normalized_mse"]["total"]
    else:
        values = curves["physical"][name]
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"invalid curve for metric: {name}")
    return result

def _series_statistics(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=1).tolist(),
    }
```

Each input record also contains relative `h1_metrics_path` and `h10_metrics_path` strings so the per-seed section can preserve provenance. `build_experiment_summary(records, snapshot_horizons, split_seed)` must:

1. sort records by seed;
2. validate unique non-negative seeds and identical `steps` arrays;
3. stack H1 and H10 curves for all four metrics;
4. calculate H1/H10 mean and sample std;
5. calculate paired `delta_h10_minus_h1` mean/std and store per-step `improved_seed_count`, `worse_seed_count`, and `equal_seed_count` inside `paired_delta`;
6. record sparse snapshots only for requested horizons present in `steps`;
7. record each pair's relative metrics paths, sparse paired values, and first step with heading delta greater than the tolerance.

The returned dictionary must be JSON-safe and contain `schema_version`, `seeds`, `split_seed`, `steps`, `metrics`, `sparse_horizons`, and `per_seed`. Serialize with `allow_nan=False` so non-finite values fail visibly.

- [ ] **Step 4: Implement stable CSV and a 2x2 comparison figure**

`write_summary_csv()` writes this exact column order:

```text
seed,horizon,metric,h1,h10,delta_h10_minus_h1
```

Rows are sorted by seed, then horizon, then metric order in `METRIC_NAMES`. `plot_multiseed_comparison()` creates four panels in metric order, with an H1 and H10 mean curve plus a `mean ± 1 std` band. Label physical units, use a linear y-axis anchored at zero, and save one PNG through a non-interactive Matplotlib backend used by the tests.

- [ ] **Step 5: Run the focused tests and commit**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_multiseed_experiment
```

Expected: PASS.

```bash
git add src/world_model_lab/multiseed_experiment.py tests/test_multiseed_experiment.py
git commit -m "feat: aggregate multiseed rollout metrics"
```

## Task 3: Add experiment orchestration and the CLI contract

**Files:**

- Modify: `src/world_model_lab/multiseed_experiment.py`
- Modify: `tests/test_multiseed_experiment.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write validation and end-to-end contract tests**

Add a helper that saves deterministic sequence dynamics with ten episodes and twelve transitions per episode. Add focused unit tests for:

- rejecting fewer than two seeds;
- rejecting duplicate or negative seeds;
- rejecting a missing, non-file dataset path;
- rejecting an output directory that already contains any entry;
- rejecting checkpoint records whose `train`, `validation`, or `test` episode IDs differ;
- rejecting diagnostics manifests whose dataset SHA-256 values differ.

Then add an end-to-end test with seeds `(0, 1)`, one epoch, `hidden_size=8`, `batch_size=32`, and diagnostic horizons `(1, 2)`. Keep the production comparison fixed at H1 versus H10; the synthetic episodes have twelve transitions so horizon-10 training windows exist.

```python
summary = run_multiseed_experiment(
    data_path=data_path,
    output_dir=output_dir,
    seeds=(0, 1),
    split_seed=0,
    rollout_loss_weight=1.0,
    hidden_size=8,
    epochs=1,
    batch_size=32,
    diagnostic_horizons=(1, 2),
    windows_per_episode=2,
    xy_bins=2,
    feature_bins=2,
    min_bin_count=1,
)
```

Assert the exact top-level filenames:

```python
self.assertEqual(
    {path.name for path in output_dir.iterdir()},
    {
        "experiment_manifest.json",
        "summary.json",
        "summary.csv",
        "multiseed_comparison.png",
        "runs",
    },
)
```

For every `seed_{0,1}/{h1,h10}`, assert one `world_model.pt` and these five diagnostics files: `metrics.json`, `manifest.json`, `overview.png`, `rollout_errors.png`, and `rollout_loss_components.png`. Load all four checkpoints and assert identical train/validation/test episode ID arrays. Assert the manifest has the input dataset SHA-256, the exact seed list, split seed, training configuration, H1/H10 configurations, diagnostics configuration, and run-relative paths.

- [ ] **Step 2: Run the orchestration tests and confirm RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_multiseed_experiment
```

Expected: failures because `run_multiseed_experiment()` and validation helpers are not yet implemented.

- [ ] **Step 3: Implement configuration validation and invariant checks**

Add a small JSON writer and this public boundary:

```python
def run_multiseed_experiment(
    *,
    data_path: Path | str,
    output_dir: Path | str,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    split_seed: int = 0,
    hidden_size: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    rollout_loss_weight: float = 1.0,
    diagnostic_horizons: Iterable[int] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    xy_bins: int = 12,
    feature_bins: int = 8,
    min_bin_count: int = 5,
) -> dict[str, Any]:
    data = Path(data_path)
    output = Path(output_dir)
```

Normalize iterables to tuples exactly once. Validate that the dataset is a regular file; seeds contain at least two unique non-negative integers; the split seed is non-negative; hidden size, epochs, batch size, learning rate, window/bin counts, and diagnostic horizons are positive; rollout loss weight is finite and non-negative; diagnostic horizons are non-empty, unique, and strictly increasing; and the output directory is absent or empty. There is no implicit overwrite or resume.

After each run, load its checkpoint with `load_checkpoint()` and its diagnostics manifest/metrics JSON. Before writing the four top-level output files, verify that all runs have equal split ID arrays for all three split names and the same dataset SHA-256. Raise `ValueError` on any invariant violation so a partial run never looks complete.

- [ ] **Step 4: Orchestrate paired runs only through existing APIs**

For each seed, create:

```text
runs/seed_<seed>/h1/world_model.pt
runs/seed_<seed>/h1/diagnostics/*
runs/seed_<seed>/h10/world_model.pt
runs/seed_<seed>/h10/diagnostics/*
```

Call `run_training()` twice with the same `seed` and `split_seed`:

```python
run_training(
    data_path=data,
    output_path=h1_checkpoint,
    hidden_size=hidden_size,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    seed=seed,
    split_seed=split_seed,
    rollout_horizon=1,
    rollout_loss_weight=0.0,
)
run_training(
    data_path=data,
    output_path=h10_checkpoint,
    hidden_size=hidden_size,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    seed=seed,
    split_seed=split_seed,
    rollout_horizon=10,
    rollout_loss_weight=rollout_loss_weight,
)
```

Immediately run `run_diagnostics()` against each checkpoint. Do not reproduce model training, rollout, normalization, or error formulas in the experiment runner. Build each aggregation record with `seed`, loaded H1/H10 metrics, and both metrics paths relative to the experiment directory.

After invariant checks, call `build_experiment_summary()`, then write `summary.json`, `summary.csv`, and `multiseed_comparison.png`. Write `experiment_manifest.json` with schema version, resolved dataset path/hash, seeds, split seed, shared training configuration, both model configurations, diagnostics configuration, and relative paths for every run. Return the compact contract from the design: output paths, seeds, split seed, longest horizon metrics, and first heading-regression step per seed.

- [ ] **Step 5: Add a stable command-line interface**

Add an argparse CLI with these principal flags:

```python
parser.add_argument("--data", type=Path, default=Path("data/transitions.npz"))
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/experiments/h1-vs-h10-seeds-0-4"),
)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--split-seed", type=int, default=0)
parser.add_argument("--hidden-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
parser.add_argument(
    "--diagnostic-horizons",
    type=int,
    nargs="+",
    default=[1, 5, 10, 20, 50],
)
```

Expose all remaining diagnostic knobs already accepted by `run_diagnostics()`. Convert `FileNotFoundError` and `ValueError` with `parser.error(str(error))`, and print the returned summary as sorted, indented JSON.

Register the command in `pyproject.toml`:

```toml
world-model-multiseed = "world_model_lab.multiseed_experiment:main"
```

- [ ] **Step 6: Run focused, CLI, and affected tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_multiseed_experiment \
  tests.test_train_world_model \
  tests.test_diagnose_model \
  tests.test_diagnostics
PYTHONPATH=src .venv/bin/python -m world_model_lab.multiseed_experiment --help
```

Expected: all tests PASS and help lists `--seeds`, `--split-seed`, and `--diagnostic-horizons`.

- [ ] **Step 7: Commit**

```bash
git add src/world_model_lab/multiseed_experiment.py tests/test_multiseed_experiment.py pyproject.toml
git commit -m "feat: run paired multiseed experiments"
```

## Task 4: Document, verify, and run the fixed five-seed protocol

**Files:**

- Modify: `README.md`
- Generate (ignored): `artifacts/experiments/h1-vs-h10-seeds-0-4/**`

- [ ] **Step 1: Document the experimental question and command**

Add a “Multi-seed stability experiment” section explaining:

- H1 is one-step training with effective rollout weight `0.0`;
- H10 uses rollout horizon `10` and rollout loss weight `1.0`;
- both variants share the same episode split (`split_seed=0`) and five training seeds;
- paired delta is `H10 - H1`, so negative means H10 is better;
- the output directory must be new or empty.

Include this reproducible command:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/world-model-multiseed \
  --data data/transitions.npz \
  --output-dir artifacts/experiments/h1-vs-h10-seeds-0-4 \
  --seeds 0 1 2 3 4 \
  --split-seed 0 \
  --epochs 100 \
  --diagnostic-horizons 1 5 10 20 50
```

Document `experiment_manifest.json`, `summary.json`, `summary.csv`, the 2x2 plot, and the per-run checkpoint/diagnostics bundle.

- [ ] **Step 2: Run the complete test suite before the expensive experiment**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests PASS. Stop and diagnose any failure before launching the five-seed run.

- [ ] **Step 3: Run the approved five-seed experiment exactly once**

Run the documented command from the repository root. Do not tune hyperparameters after seeing intermediate seeds; the ten trainings are one paired protocol.

Expected top-level output:

```text
artifacts/experiments/h1-vs-h10-seeds-0-4/
  experiment_manifest.json
  summary.json
  summary.csv
  multiseed_comparison.png
  runs/seed_0/{h1,h10}/{world_model.pt,diagnostics/}
  runs/seed_1/{h1,h10}/{world_model.pt,diagnostics/}
  runs/seed_2/{h1,h10}/{world_model.pt,diagnostics/}
  runs/seed_3/{h1,h10}/{world_model.pt,diagnostics/}
  runs/seed_4/{h1,h10}/{world_model.pt,diagnostics/}
```

- [ ] **Step 4: Validate artifacts and classify the result**

Run:

```bash
.venv/bin/python -m json.tool artifacts/experiments/h1-vs-h10-seeds-0-4/experiment_manifest.json >/dev/null
.venv/bin/python -m json.tool artifacts/experiments/h1-vs-h10-seeds-0-4/summary.json >/dev/null
git check-ignore artifacts/experiments/h1-vs-h10-seeds-0-4/summary.json
git diff --check
```

Read `summary.json` and report, for horizon 50:

- mean ± sample std for H1 and H10;
- paired delta mean ± sample std;
- improved/worse/equal counts for position, heading, velocity, and normalized total;
- every seed’s first heading-regression step.

Apply these predeclared labels without changing thresholds:

- **stable tradeoff:** at least 4/5 seeds improve position and velocity, while at least 4/5 worsen heading at horizon 50;
- **stable improvement:** at least 4/5 improve position, heading, and velocity;
- **seed-sensitive:** signs vary across seeds and neither stable rule applies;
- **mixed:** any remaining pattern.

- [ ] **Step 5: Commit documentation only**

Generated experiment artifacts remain ignored. Commit the README after its command and output contract have been verified:

```bash
git add README.md
git commit -m "docs: explain multiseed stability experiment"
```

- [ ] **Step 6: Final verification**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: the full suite passes, no whitespace errors exist, and the worktree is clean. The branch should contain four implementation/documentation commits after the design and plan commits.
