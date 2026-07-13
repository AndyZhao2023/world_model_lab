# Model Diagnostics Lab Design

## Goal

Turn the existing state-space world model into a reproducible diagnostic
benchmark. The benchmark must explain where the model is accurate, where it
fails, and how recursive prediction compounds one-step error without changing
the model architecture, training loop, or checkpoint format.

## Scope

This stage evaluates the existing checkpoint and dataset. It does not retrain
models, add representation ablations, introduce experiment-tracking services,
or implement MPC/PPO.

The first benchmark covers:

- dataset coverage on the checkpoint test split;
- teacher-forced one-step prediction errors;
- recursive free-rollout errors from deterministic windows;
- fixed-window horizon comparisons with episode-balanced aggregation;
- error slices over XY position, velocity, steering, and acceleration;
- machine-readable metrics, visual reports, and an input manifest.

## Architecture

The implementation is split by responsibility:

- `diagnostics.py` contains pure NumPy-oriented data selection, prediction,
  aggregation, and binning functions. It does not read files or create plots.
- `diagnostic_plots.py` renders diagnostic result structures to PNG files. It
  does not load checkpoints or recompute metrics.
- `diagnose_model.py` is the orchestration and CLI layer. It loads the dataset
  and checkpoint, calls the diagnostic core, writes JSON files, and invokes the
  plot layer.

The existing `evaluate_rollout.py` remains available for inspecting one
representative episode. The new benchmark may reuse its validated rollout
construction semantics, but the benchmark API and report generation stay in
the new focused modules.

## Evaluation Population

Errors are evaluated only on episode IDs stored in the checkpoint's `test`
split. The command fails if a test episode is missing, steps are not contiguous
from zero, or adjacent transitions do not form a continuous trajectory.

Coverage plots compare train and test occupancy so distribution differences are
visible, but train transitions never contribute to model-error metrics.

## One-Step Evaluation

Teacher-forced prediction receives the recorded true state and action for every
test transition:

```text
(true state_t, recorded action_t) -> model -> predicted state_t+1
```

The benchmark compares the prediction with the recorded `next_state`. Heading
differences are wrapped to `[-pi, pi)` before absolute error statistics are
computed.

Each error family reports `count`, `mean`, `median`, `p90`, and `max`:

- position error in metres: Euclidean norm over X and Y;
- heading error in degrees;
- velocity error in metres per second.

## Fixed-Window Rollout Protocol

The CLI receives an ordered positive horizon list and uses its maximum as the
required window length. For each eligible test episode, the benchmark selects
up to `windows_per_episode` start indices evenly across all valid starts. This
selection is deterministic and includes both boundary starts when more than
one window is requested.

Every selected window contains the same maximum number of actions. Therefore
all requested horizons are evaluated on exactly the same windows and episode
set.

Two prediction modes are computed for every window:

1. Teacher forcing predicts each next state independently from the recorded
   state at that step.
2. Free rollout receives the true initial state once, then recursively feeds
   each predicted state back into the model while applying the recorded action
   sequence.

The difference between their errors exposes compounding error. Both modes use
identical windows and actions.

Metrics are aggregated in two levels: first average windows within each
episode, then average the per-episode values. This macro average prevents long
episodes from dominating because they provide more possible windows.

For a requested horizon `h`, the teacher-forcing prediction uses the recorded
state and action at window offset `h - 1` and compares the predicted next state
with offset `h`. The free-rollout prediction starts at window offset zero and
recursively applies all `h` actions. Horizon `mean`, `median`, `p90`, and `max`
are computed over per-episode window means; the report separately records the
episode and window counts.

The report records eligible episode IDs, selected start indices, skipped short
episode IDs, `windows_per_episode`, and the evaluated horizons.

## Error Slices

One-step test errors are grouped by the current transition's:

- XY position on a fixed rectangular grid;
- velocity;
- steering;
- acceleration.

XY, velocity, and action bin edges come from the observed full-dataset ranges
and are stored in the report, making unchanged dataset runs directly
comparable. The final edge includes the observed maximum. Each cell or
one-dimensional bin includes its sample count and the three error summaries.

The plot layer masks error cells with fewer than `min_bin_count` samples. Their
JSON entries retain `count` but store error summaries as `null`, so sparse
regions are not mistaken for reliable low-error regions.

## Output Bundle

The default output directory is `artifacts/diagnostics/baseline/`:

```text
artifacts/diagnostics/baseline/
├── metrics.json
├── manifest.json
├── overview.png
└── rollout_errors.png
```

`metrics.json` contains one-step summaries, error slices, rollout windows, and
teacher-forcing/free-rollout horizon metrics. JSON values must use standard
finite numbers; an empty or masked bin is represented with `null`, never NaN.

`manifest.json` records:

- SHA-256 and resolved path for the dataset and checkpoint;
- training configuration and test episode IDs from the checkpoint;
- diagnostic parameters and output schema version.

`overview.png` shows train/test XY coverage, XY position-error heatmap, and
velocity/steering/acceleration error slices with sample counts.

`rollout_errors.png` compares teacher-forcing and free-rollout position,
heading, and velocity errors over the requested horizons.

## CLI

The module is exposed both through Python and a console script:

```bash
.venv/bin/python -m world_model_lab.diagnose_model \
  --data data/transitions.npz \
  --checkpoint artifacts/world_model.pt \
  --output-dir artifacts/diagnostics/baseline \
  --horizons 1 5 10 20 50 \
  --windows-per-episode 8 \
  --xy-bins 12 \
  --feature-bins 8 \
  --min-bin-count 5
```

The installed command name is `world-model-diagnose`. Invalid horizons, bin
counts, missing arrays, missing checkpoint splits, non-contiguous trajectories,
and the absence of any episode long enough for the maximum horizon produce a
clear error and a non-zero exit.

## Testing

Tests use small synthetic trajectories and deterministic constant-delta models.
They cover:

- deterministic, evenly spaced rollout-window selection;
- identical window membership at every horizon;
- episode-balanced aggregation rather than transition-weighted aggregation;
- teacher-forcing and recursive free-rollout semantics;
- wrapped heading and physical-unit error statistics;
- bin counts, masked sparse bins, and JSON-safe `null` values;
- creation and signatures of both PNG reports;
- SHA-256 manifest entries and complete output-bundle creation;
- CLI validation for malformed datasets and impossible maximum horizons.

The existing 35 tests must continue to pass. A real-data smoke run against
`data/transitions.npz` and `artifacts/world_model.pt` must produce all four
bundle files with finite top-level metrics.

## Success Criteria

The stage is complete when one deterministic command generates the same
evaluation population, metrics, and plots for unchanged dataset/checkpoint
inputs; all horizon curves use one fixed window population; and the output
identifies both high-error regions and low-coverage regions without conflating
the two.
