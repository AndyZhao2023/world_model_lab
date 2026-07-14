# Multi-Seed Stability Experiment Design

## Goal

Add a reproducible experiment runner that determines whether the observed H1
versus H10 rollout tradeoff is stable across training randomness. The runner
must hold the episode split fixed, vary only model/training randomness, train
paired H1 and H10 models for five seeds, run the same held-out diagnostics for
every checkpoint, and aggregate the paired results into machine-readable and
visual artifacts.

The immediate scientific question is:

> Does H10 consistently improve long-horizon position and velocity while
> degrading heading, or was the seed-0 result a random fluctuation?

## Scope

This stage adds:

- an explicit training/data-split seed boundary;
- a reusable multi-seed experiment module and CLI;
- paired H1/H10 training for seeds `0, 1, 2, 3, 4` by default;
- the existing schema-version-2 held-out diagnostics for every checkpoint;
- dense step-curve aggregation across seeds;
- sparse horizon summaries and paired per-seed CSV output;
- a four-panel mean-plus-standard-deviation comparison plot;
- experiment configuration and dataset fingerprinting;
- deterministic unit and end-to-end tests;
- README documentation and one real five-seed experiment run.

This stage does not change the MLP architecture, state representation,
one-step loss, free-rollout loss formula, optimizer, checkpoint format,
dataset contents, action source, diagnostic metric definitions, MPC, PPO,
rewards, or termination prediction. It does not add a general YAML experiment
framework, parallel/distributed training, hyperparameter search, automatic
resume, or statistical significance testing.

## Why Split and Training Seeds Must Be Separate

The current `seed` parameter controls two different sources of variation:

1. `split_episode_ids(..., seed=seed)` determines the train, validation, and
   test episodes;
2. PyTorch initialization and batch permutations use the same seed during
   optimization.

Changing that single value therefore changes both the learned model and the
held-out population. Such a run measures end-to-end pipeline variability, but
it cannot isolate whether H10's behavior comes from its training objective or
from a different test split.

The experiment introduces an optional `split_seed` at the training boundary:

```python
def run_training(
    *,
    ...,
    seed: int = 0,
    split_seed: int | None = None,
    rollout_horizon: int = 1,
    ...,
) -> dict[str, Any]:
```

The effective split seed is:

```python
effective_split_seed = seed if split_seed is None else split_seed
```

This preserves existing behavior for all callers that do not supply
`split_seed`. The training CLI adds optional `--split-seed`; omitting it keeps
the current coupled-seed behavior. New checkpoints record the effective value
as `training_config["split_seed"]`, and the returned training summary exposes
`split_seed` as well.

The multi-seed runner always passes the fixed experiment `split_seed` and uses
each value from `seeds` only as the training seed. H1 and H10 within a pair use
the same training seed and split seed.

## Compared Configurations

The experiment is deliberately fixed to the current scientific comparison:

| Name | Rollout horizon | Rollout loss weight |
|---|---:|---:|
| `h1` | 1 | 0.0 effective weight |
| `h10` | 10 | 1.0 by default |

Both configurations share:

- dataset;
- fixed episode split;
- training seed within each pair;
- MLP hidden size;
- epochs;
- batch size;
- learning rate;
- diagnostic horizons and window-selection protocol.

The runner accepts `rollout_loss_weight` so the exact H10 experiment remains
recorded and repeatable, but it does not expose arbitrary model configurations
or horizons. A broader experiment matrix belongs to a later scope.

## Architecture

The feature adds one orchestration module and reuses the current boundaries:

```text
multiseed_experiment.py
    -> validate experiment protocol and empty output directory
    -> for each training seed
        -> run_training(... split_seed=fixed, horizon=1)
        -> run_diagnostics(... same held-out protocol)
        -> run_training(... split_seed=fixed, horizon=10)
        -> run_diagnostics(... same held-out protocol)
        -> verify both checkpoints use the same split IDs
    -> load schema-v2 metrics from every run
    -> aggregate paired dense curves and sparse horizons
    -> write experiment_manifest.json
    -> write summary.json
    -> write summary.csv
    -> write multiseed_comparison.png
```

`train_world_model.py` remains responsible for data splitting, normalization,
training, checkpointing, and per-checkpoint metrics. `diagnose_model.py`
remains responsible for held-out evaluation and its five-file bundle. The new
module only orchestrates those APIs and aggregates their output; it must not
copy training or diagnostic math.

## Public Python Interface and CLI

The reusable entry point is:

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
```

The package registers:

```text
world-model-multiseed = world_model_lab.multiseed_experiment:main
```

Default command:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/world-model-multiseed \
  --data data/transitions.npz \
  --output-dir artifacts/experiments/h1-vs-h10-seeds-0-4 \
  --split-seed 0 \
  --seeds 0 1 2 3 4 \
  --epochs 100
```

The CLI exposes one flag for every function parameter. `--seeds` and
`--diagnostic-horizons` use one-or-more integer values. It catches
`FileNotFoundError` and `ValueError` and reports them through
`argparse.ArgumentParser.error`, matching the existing commands.

## Input Validation

The runner rejects the experiment before starting training when:

- fewer than two seeds are supplied;
- a seed is negative;
- seeds contain duplicates;
- `split_seed` is negative;
- epochs, hidden size, batch size, learning rate, or bin/window counts are not
  positive;
- rollout loss weight is non-finite or negative;
- diagnostic horizons are empty, non-positive, duplicated, or unsorted;
- the data path is missing;
- the output directory exists and is not empty.

An existing empty output directory is allowed. Requiring a fresh directory
prevents old seeds or old protocol results from being silently mixed into a
new summary. Automatic resume and overwrite are intentionally excluded.

If a training or diagnostic run fails after work begins, the exception
propagates and completed per-run artifacts remain available for inspection.
The runner does not write any top-level manifest, summary, CSV, or comparison
plot until all paired runs and invariant checks succeed. A rerun uses a new or
manually emptied output directory.

## Artifact Layout

The complete output is:

```text
artifacts/experiments/h1-vs-h10-seeds-0-4/
├── experiment_manifest.json
├── summary.json
├── summary.csv
├── multiseed_comparison.png
└── runs/
    ├── seed_0/
    │   ├── h1/
    │   │   ├── world_model.pt
    │   │   └── diagnostics/
    │   │       ├── metrics.json
    │   │       ├── manifest.json
    │   │       ├── overview.png
    │   │       ├── rollout_errors.png
    │   │       └── rollout_loss_components.png
    │   └── h10/
    │       └── ... same files ...
    └── seed_1 ... seed_4/
```

All paths stored inside top-level experiment artifacts are relative to the
experiment output directory, except the resolved dataset path. The entire
tree lives below the already ignored `artifacts/` directory.

## Experiment Manifest

`experiment_manifest.json` uses schema version 1 and contains only protocol
and provenance, not aggregate results:

```json
{
  "schema_version": 1,
  "dataset": {
    "path": "/absolute/path/data/transitions.npz",
    "sha256": "..."
  },
  "training": {
    "seeds": [0, 1, 2, 3, 4],
    "split_seed": 0,
    "hidden_size": 128,
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "models": {
      "h1": {"rollout_horizon": 1, "rollout_loss_weight": 0.0},
      "h10": {"rollout_horizon": 10, "rollout_loss_weight": 1.0}
    }
  },
  "diagnostics": {
    "horizons": [1, 5, 10, 20, 50],
    "windows_per_episode": 8,
    "xy_bins": 12,
    "feature_bins": 8,
    "min_bin_count": 5
  },
  "runs": [
    {
      "seed": 0,
      "model": "h1",
      "checkpoint": "runs/seed_0/h1/world_model.pt",
      "metrics": "runs/seed_0/h1/diagnostics/metrics.json",
      "manifest": "runs/seed_0/h1/diagnostics/manifest.json"
    }
  ]
}
```

The `runs` array is ordered by seed and then `h1`, `h10`. Before writing this
manifest, the runner verifies that every diagnostic manifest contains the
same dataset hash and the same train, validation, and test split IDs. Because
the diagnostic manifest currently stores test IDs but not train/validation
IDs, the runner loads each checkpoint and compares all three
`split_episode_ids` arrays directly.

## Aggregated Summary

`summary.json` uses schema version 1. The aggregation unit is a paired training
seed, not an individual rollout window or episode. It contains:

```text
schema_version
seeds
split_seed
steps
metrics
  position
  heading_degrees
  velocity
  normalized_total
    h1.mean
    h1.std
    h10.mean
    h10.std
    paired_delta.mean
    paired_delta.std
    paired_delta.improved_seed_count
    paired_delta.worse_seed_count
    paired_delta.equal_seed_count
sparse_horizons
per_seed
```

For every dense step and metric, let `x_i` be the value for training seed `i`.
The aggregate mean is the arithmetic mean across seeds, and standard deviation
is the sample standard deviation with `ddof=1`. At least two seeds are required
so this value is defined.

The paired difference is always:

```text
delta = H10 - H1
```

All tracked metrics are errors, so lower is better:

- `delta < -1e-12` counts as improved;
- `delta > 1e-12` counts as worse;
- otherwise it counts as equal.

`normalized_total` comes directly from each report's
`rollout.step_curves.<mode>.normalized_mse.total`. Physical metrics come from
`rollout.step_curves.<mode>.physical`. This experiment compares the
`free_rollout` mode only.

`sparse_horizons` contains an aggregate snapshot for each configured
diagnostic horizon and each of the four metrics. Its values are selected from
the dense arrays at index `horizon - 1`, ensuring one numerical source rather
than separately aggregating sparse diagnostic summaries.

`per_seed` stores, for each seed:

- relative H1 and H10 metrics paths;
- the first dense step where H10 heading is worse than H1 by more than
  `1e-12`, or `null` if none exists;
- paired values at each configured sparse horizon for all four metrics.

All numeric values must be finite and the file must pass
`json.dumps(..., allow_nan=False)`.

## CSV Output

`summary.csv` is a tidy paired table with columns:

```text
seed,horizon,metric,h1,h10,delta_h10_minus_h1
```

For each seed, configured horizon, and metric, one row is written. Model names
do not become separate rows because the paired columns make the comparison
explicit. Rows are ordered by seed, horizon, and metric in this order:
`position`, `heading_degrees`, `velocity`, `normalized_total`.

## Comparison Plot

`multiseed_comparison.png` contains a 2-by-2 grid:

1. position error in metres;
2. heading error in degrees;
3. velocity error in metres per second;
4. normalized total MSE.

Every panel plots dense step `1..max_horizon` curves for H1 and H10. The line
is the across-seed mean and the translucent band is mean plus or minus one
sample standard deviation. The y-axis is linear and anchored at zero. H1 uses
blue and H10 uses orange/red consistently with existing diagnostic plots.

The plot consumes `summary.json`-style aggregate values and performs no model
inference or metric aggregation.

## Returned CLI Summary

`run_multiseed_experiment()` returns a compact JSON-safe mapping printed by
the CLI:

```text
output_dir
manifest
summary
csv
plot
seeds
split_seed
longest_horizon
longest_horizon_metrics
  position
  heading_degrees
  velocity
  normalized_total
    h1_mean
    h10_mean
    paired_delta_mean
    improved_seed_count
    worse_seed_count
first_heading_regression_steps
```

This console summary answers the main question without duplicating every
dense value.

## Testing Strategy

Tests remain deterministic and use temporary directories.

`tests/test_train_world_model.py` verifies:

- different training seeds with the same explicit split seed produce
  identical train/validation/test episode IDs;
- omitting `split_seed` produces the same split as explicitly setting it to
  the training seed;
- checkpoint training config and returned summary contain the effective split
  seed.

`tests/test_multiseed_experiment.py` verifies:

- validation rejects duplicate/negative/insufficient seeds, invalid protocol
  values, and non-empty output directories before training;
- synthetic dense curves produce exact paired means, sample standard
  deviations, deltas, improvement counts, and first heading regression steps;
- CSV row order and values match the JSON summary;
- the comparison plot writes a valid PNG with four panels;
- a small real end-to-end run with two training seeds, one epoch, a small
  hidden size, and a deterministic synthetic dataset writes all top-level and
  per-run artifacts;
- the end-to-end run's H1 and H10 checkpoints share all split IDs and the
  output JSON is finite.

The complete existing unit suite runs after focused tests. The real experiment
then trains seeds `0..4` for 100 epochs and runs diagnostics at horizons
`1, 5, 10, 20, 50` without changing the protocol after observing results.

## Documentation

README adds a `多随机种子稳定性实验` section that explains:

- why the split seed is fixed;
- why H1 and H10 results are paired by training seed;
- the default command and artifact layout;
- that `delta = H10 - H1` and negative error delta means improvement;
- that mean plus or minus standard deviation describes variability but is not
  a formal significance test;
- how to decide whether the heading tradeoff is systematic or seed-sensitive.

## Acceptance Criteria

The feature is complete when:

- existing training callers retain their original split behavior;
- explicit `split_seed=0` yields identical episode splits for all ten runs;
- five paired H1/H10 seeds complete under the unchanged training and
  diagnostic protocol;
- every run produces a valid checkpoint and five-file diagnostic bundle;
- dense aggregate arrays have exactly 50 finite values;
- sample means, standard deviations, paired deltas, counts, and first heading
  regression steps are reproducible;
- the JSON, CSV, and PNG outputs agree at sparse horizons;
- focused and full tests pass;
- generated experiment artifacts remain ignored by Git;
- the experiment conclusion is reported as stable, seed-sensitive, or mixed
  based on paired evidence rather than a single run.
