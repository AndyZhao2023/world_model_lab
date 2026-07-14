# Ensemble Uncertainty Diagnostics Design

## Goal

Add a reproducible diagnostic that combines multiple independently trained H10
world-model checkpoints into one ensemble and tests whether model disagreement
is useful as an estimate of prediction risk.

The experiment must answer two separate questions on the held-out test split:

1. Does the ensemble mean predict better than a typical individual member?
2. When ensemble members disagree more, is the ensemble mean actually more
   likely to be wrong?

The second question is the prerequisite for using uncertainty as a future MPC
penalty. This stage measures that prerequisite; it does not implement MPC.

## Scope

This stage adds:

- loading and validation for two or more compatible checkpoints;
- batched ensemble next-state prediction in physical units;
- circular aggregation for heading;
- physical-unit disagreement for position, heading, and velocity;
- held-out one-step error-versus-disagreement calibration;
- held-out free-rollout comparison of the ensemble mean and its members;
- JSON metrics, a provenance manifest, CSV calibration rows, and plots;
- a package CLI, tests, and README instructions;
- one real run using the five H10 checkpoints from the existing multi-seed
  experiment protocol.

This stage does not retrain the members, bootstrap episodes, change the MLP,
predict aleatoric variance, change checkpoint format, alter existing diagnostic
metrics, implement an uncertainty loss, connect the ensemble to MPC or PPO, or
claim that disagreement is calibrated before the experiment demonstrates it.

## Why Reuse the Existing H10 Members

The current multi-seed experiment already trains H10 models with:

- identical data;
- identical train, validation, and test episode IDs;
- identical model and optimization hyperparameters;
- different PyTorch initialization and batch-order seeds.

Those checkpoints isolate training randomness and are sufficient for the first
epistemic-uncertainty experiment. Episode bootstrap is a reasonable follow-up
if different seeds alone produce too little member diversity, but adding it now
would mix a training-pipeline change into the uncertainty measurement.

Only H10 members are used in a given ensemble. H1 and H10 checkpoints must not
be mixed because their different objectives would make disagreement partly a
training-objective comparison instead of uncertainty within one model family.

## Architecture

The feature adds two focused modules:

```text
ensemble.py
    -> validate compatible LoadedWorldModel members
    -> predict every member's next state
    -> aggregate x, y, velocity by arithmetic mean
    -> aggregate heading by circular mean
    -> compute per-sample member disagreement

diagnose_ensemble.py
    -> load dataset and checkpoints
    -> select the common held-out test split
    -> run one-step calibration
    -> run independent member free rollouts
    -> aggregate member and ensemble errors
    -> write manifest, metrics, CSV, and plots
```

`train_world_model.py` remains responsible for checkpoint loading and member
inference. `diagnostics.py` remains responsible for test-window selection and
physical error definitions. The new modules reuse those boundaries rather than
copying training, splitting, or error math.

## Checkpoint Compatibility Invariant

The loader requires at least two checkpoints. Every checkpoint must have the
same:

- `split_episode_ids` arrays for train, validation, and test;
- `training_config["split_seed"]`;
- `training_config["rollout_horizon"]`, which must equal `10`;
- effective `training_config["rollout_loss_weight"]`;
- model input/output contract and hidden size;
- input and target normalizer means and standard deviations.

Training seeds must be present, non-negative, and unique. Validation failures
name the incompatible field before any metrics are written. Checkpoint order is
normalized by training seed so artifacts are reproducible when CLI paths are
supplied in a different order.

The normalizer equality requirement is expected because all members use the
same training episodes. It also gives normalized disagreement one unambiguous
physical scale. A later bootstrap ensemble may relax this invariant by storing
an explicit ensemble-level reference normalizer.

## Ensemble Prediction

For a batch of `N` state/action pairs and `M` members, prediction returns:

```python
@dataclass(frozen=True)
class EnsemblePrediction:
    member_next_states: np.ndarray  # [M, N, 4]
    mean_next_states: np.ndarray    # [N, 4]
    disagreement: dict[str, np.ndarray]
```

Each member uses its own checkpoint inference path. Arithmetic components are:

```text
mean_x        = mean(member_x)
mean_y        = mean(member_y)
mean_velocity = mean(member_velocity)
```

Heading uses the circular mean:

```text
mean_heading = atan2(mean(sin(member_heading)),
                     mean(cos(member_heading)))
```

This avoids the incorrect arithmetic result near `-pi/pi`. If headings are
`179 degrees` and `-179 degrees`, the mean must be near `180 degrees`, not zero.

Per-sample disagreement contains:

- `position`: root-mean-square Euclidean distance of member XY predictions
  from the ensemble mean, in metres;
- `heading_degrees`: root-mean-square wrapped angular distance from the
  circular mean, in degrees;
- `velocity`: root-mean-square distance from mean velocity, in metres/second;
- `normalized_total`: root mean square over all member/component deviations
  after division by the common target-delta standard deviation.

The normalized heading deviation is wrapped before scaling. All arrays must be
finite. The ensemble mean and disagreement are copies that callers cannot use
to mutate internal member state.

## One-Step Calibration

One-step calibration evaluates every transition from the common checkpoint
test episodes:

```text
true state_t + recorded action_t
                 |
                 v
       M next-state predictions
                 |
        circular ensemble mean
                 |
        compare with true state_t+1
```

For position, heading, velocity, and normalized total, the diagnostic records:

- ensemble error distribution;
- member error distributions and their across-member mean;
- ensemble improvement relative to mean member error;
- Pearson correlation between disagreement and ensemble error;
- equal-count calibration bins ordered by disagreement;
- lowest-bin and highest-bin mean error and their risk ratio.

Correlation is descriptive evidence, not a significance test. Constant
disagreement produces a JSON `null` correlation rather than NaN. Calibration
bins retain sample count, disagreement mean, and error mean; no empty or
non-finite bin values are serialized.

## Free-Rollout Evaluation

The diagnostic reuses deterministic held-out windows selected by
`select_rollout_windows`. Every member begins from the same real initial state
and rolls forward independently under the recorded action sequence:

```text
member state_t -> member model -> member state_t+1 -> same member model -> ...
```

The ensemble mean is computed across member states separately at every rollout
step. The mean is not fed back into each member because that would collapse
member trajectories and underestimate future disagreement.

The diagnostic computes dense curves for every step from `1` through the
largest requested horizon and stores the requested horizons as named snapshots.
At every dense step, the output records:

- ensemble mean position, heading, velocity, and normalized-total error;
- mean, minimum, and maximum individual-member error;
- ensemble disagreement at that horizon;
- disagreement/error correlation across rollout windows;
- the number of eligible episodes and windows.

As in the existing diagnostics, errors are averaged within each episode before
episodes are weighted equally for aggregate accuracy. Correlation uses window
samples because it tests whether uncertainty ranks individual predictions by
risk; the artifact explicitly records both aggregation units.

## Public Interface and CLI

The reusable diagnostic entry point is:

```python
def run_ensemble_diagnostics(
    *,
    data_path: Path | str,
    checkpoint_paths: Iterable[Path | str],
    output_dir: Path | str,
    horizons: Iterable[int] = (1, 5, 10, 20, 50),
    windows_per_episode: int = 8,
    calibration_bins: int = 10,
) -> dict[str, Any]:
```

The package registers:

```text
world-model-diagnose-ensemble = world_model_lab.diagnose_ensemble:main
```

Example using the existing multi-seed H10 outputs:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/world-model-diagnose-ensemble \
  --data data/transitions.npz \
  --checkpoints \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_0/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_1/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_2/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_3/h10/world_model.pt \
    artifacts/experiments/h1-vs-h10-seeds-0-4/runs/seed_4/h10/world_model.pt \
  --output-dir artifacts/diagnostics/h10-ensemble-seeds-0-4
```

The CLI catches file and value errors through `argparse` and prints the returned
artifact summary as sorted, indented JSON, matching existing commands.

## Artifact Contract

The output directory must be absent or empty. A successful run writes:

```text
h10-ensemble-seeds-0-4/
├── manifest.json
├── metrics.json
├── one_step_calibration.csv
├── one_step_calibration.png
└── rollout_uncertainty.png
```

`manifest.json` schema version 1 stores:

- resolved dataset path and SHA-256;
- sorted checkpoint paths and SHA-256 values;
- member training seeds and shared configuration;
- test episode IDs;
- horizon, window, and calibration-bin settings.

`metrics.json` schema version 1 stores the one-step and rollout results. JSON is
written with `allow_nan=False`. Both plots are derived only from `metrics.json`
compatible records so plotting can be tested independently of model inference.

The one-step plot has four panels. Each panel plots mean observed error against
mean disagreement for the calibration bins, with sample counts available in
the CSV. The rollout plot has four panels showing ensemble error, mean-member
error, and ensemble disagreement over horizon. Because error and disagreement
can use different units, disagreement uses a secondary y-axis and must be
labelled explicitly.

Top-level files are written only after checkpoint validation and all numerical
evaluation succeed. If evaluation fails, the incomplete directory contains no
apparently valid manifest or metrics bundle.

## Error Handling

The runner rejects:

- fewer than two checkpoints;
- duplicate, missing, or non-file checkpoint paths;
- incompatible checkpoint metadata or normalizers;
- a missing dataset or required dataset arrays;
- dataset episode IDs that do not contain the checkpoint test split;
- non-positive, duplicated, or unsorted horizons;
- non-positive `windows_per_episode` or `calibration_bins`;
- an existing non-empty output directory;
- non-finite predictions, errors, or disagreements;
- requested horizons longer than every selected test episode.

## Testing Strategy

Tests use small deterministic `LoadedWorldModel` fixtures and a temporary NPZ
dataset. They cover:

- arithmetic mean and physical disagreement for x, y, and velocity;
- circular heading mean and disagreement across the `-pi/pi` boundary;
- checkpoint sorting and every compatibility failure;
- one-step calibration bins, correlations, and constant-disagreement handling;
- independent recursive member rollouts rather than shared mean-state rollout;
- equal episode weighting for aggregate accuracy;
- complete JSON-safe artifact output and plot creation;
- CLI help, argument-error behavior, and package script registration;
- the full existing test suite as regression coverage.

Each production behavior is introduced through a failing test before the
minimal implementation. The final real run is an experiment verification step,
not a replacement for automated tests.

## Success Criteria

Implementation is complete when:

1. all existing and new tests pass;
2. two or more compatible H10 checkpoints can be evaluated without retraining;
3. the result quantifies ensemble-versus-member accuracy at horizons
   `1, 5, 10, 20, 50`;
4. one-step and rollout artifacts quantify disagreement-versus-error behavior;
5. heading aggregation remains correct at the angular wrap boundary;
6. the five-member real run produces the complete artifact bundle;
7. the conclusion states whether disagreement is useful on this dataset rather
   than assuming that it is.
