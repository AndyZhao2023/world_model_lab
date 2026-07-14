# Episode Bootstrap Ensemble Design

## Goal

Add reproducible episode-level bootstrap training for H10 world-model ensemble
members and compare its long-horizon uncertainty signal against the existing
seed-only H10 ensemble.

The experiment answers one focused question:

> Does training each member on a different episode bootstrap sample make
> ensemble disagreement track held-out rollout error more reliably at H10 and
> longer horizons?

The implementation must preserve the current model architecture, held-out
split, normalization scale, H10 objective, and diagnostic definitions so the
training sample is the only intentional experimental variable.

## Scope

This stage adds:

- deterministic sampling of training episode IDs with replacement;
- consistent expansion of one-step transitions and H10 sequence windows from
  the same episode draw counts;
- a shared normalizer fitted on the complete unbootstrapped training split;
- bootstrap provenance in every checkpoint;
- a focused five-member H10 bootstrap experiment runner;
- comparison of bootstrap and existing seed-only ensemble diagnostics;
- JSON, CSV, plot, manifest, tests, and README instructions;
- one real run with training seeds `0` through `4` and fixed `split_seed=0`.

This stage does not change `WorldModelMLP`, retrain the H1 baseline, bootstrap
validation or test data, estimate aleatoric variance, change the H10 loss,
connect uncertainty to MPC/PPO, add distributed training, or declare bootstrap
successful merely because member disagreement becomes larger.

## Experimental Invariants

The seed-only and bootstrap ensembles must share:

- the exact input dataset and SHA-256;
- `split_seed` and train/validation/test episode arrays;
- model input/output contract and hidden size;
- optimizer hyperparameters and epoch count;
- `rollout_horizon=10` and rollout loss weight;
- the input and target normalizer statistics fitted on the complete training
  split;
- training seeds, with one baseline member and one bootstrap member per seed;
- held-out diagnostic horizons, window count, calibration bins, and error
  definitions.

Only the bootstrap ensemble changes which training episodes contribute to the
loss and how many times each contributes. The existing seed-only checkpoints
are supplied as inputs and are not retrained during this experiment.

## Bootstrap Contract

Given `N` unique training episode IDs, one member draws exactly `N` IDs with
replacement using `numpy.random.default_rng(bootstrap_seed)`. Draw order is
retained because it defines the deterministic order of expanded training data.

For example:

```text
train IDs       = [10, 11, 12, 13]
bootstrap draws = [12, 10, 12, 13]
draw counts     = {10: 1, 11: 0, 12: 2, 13: 1}
```

Episode `12` contributes its complete transition sequence twice, episode `11`
does not contribute to that member, and validation/test data remain unchanged.
The number of sampled episodes equals the original train episode count, while
the total transition count may vary because episode lengths may differ. This
is expected episode-level bootstrap behavior.

The sampler rejects:

- non-vector, empty, non-integer, negative, or duplicate source episode IDs;
- boolean, negative, or non-integer bootstrap seeds;
- any sampled ID not present in the original train split.

Sampling is a pure function. Identical source IDs and seed produce identical
draws and counts across repeated runs.

## Architecture

The feature adds one focused data module and one experiment orchestrator:

```text
bootstrap.py
    -> validate unique training episode IDs
    -> draw N episode IDs with replacement
    -> expand dataset transition indices in draw order
    -> summarize deterministic draw counts

train_world_model.py
    -> split episodes exactly as before
    -> fit normalizers from the complete train split
    -> optionally replace loss inputs with bootstrap-expanded inputs
    -> build H10 windows from the same bootstrap episode draws
    -> keep validation/test inputs and windows unchanged
    -> save bootstrap provenance in checkpoint metadata

bootstrap_experiment.py
    -> validate existing seed-only H10 baseline checkpoints
    -> train five bootstrap H10 members
    -> run existing ensemble diagnostics for both groups
    -> compare error and disagreement/error correlation
    -> write the experiment artifact bundle
```

`ensemble.py` remains responsible for compatible member loading and ensemble
prediction. `diagnose_ensemble.py` remains responsible for held-out one-step
and rollout diagnostics. Bootstrap code reuses both rather than introducing a
second uncertainty definition.

## Training Integration

`run_training()` gains an optional `bootstrap_seed: int | None = None`.

- `None` preserves current behavior byte-for-byte at the data-contract level.
- An integer enables episode bootstrap after the split is created.

The full training mask is always constructed first. Input and target
normalizers are fitted from that full mask. `train_model()` accepts explicit
normalizers for this path; when omitted, it retains its current behavior of
fitting them from the supplied training arrays.

When bootstrap is enabled:

1. Draw episode IDs from `splits["train"]`.
2. Expand transition indices by concatenating all transitions for each draw in
   draw order. A repeated episode repeats its transitions.
3. Pass the expanded one-step inputs and targets to `train_model()` together
   with the full-train normalizers.
4. Pass the same drawn episode ID vector to `build_sequence_windows()`. That
   existing function already iterates supplied IDs in order, so repeated IDs
   repeat every eligible H10 window.
5. Build validation windows from the original unique validation IDs.
6. Evaluate validation and test metrics on the original held-out masks.

The one-step and rollout terms therefore apply the same episode multiplicity.
An episode shorter than H10 can still contribute one-step transitions but has
no H10 windows, matching existing sequence-window behavior. The checkpoint
reports both expanded transition count and expanded H10-window count.

## Checkpoint Provenance

Bootstrap checkpoints retain the existing checkpoint format and add training
configuration fields:

```json
{
  "bootstrap_seed": 3,
  "bootstrap_episode_draws": 200,
  "bootstrap_unique_episodes": 126,
  "bootstrap_episode_counts": {
    "4": 2,
    "7": 1
  }
}
```

The counts mapping includes every original train episode, including zero-count
episodes, uses decimal string keys for JSON stability, and its values sum to
`bootstrap_episode_draws`. Seed plus source split and counts make the sample
auditable and reproducible.

Non-bootstrap checkpoints continue omitting these fields so existing loaders
remain compatible. Ensemble compatibility continues requiring equal split and
normalizer arrays. It does not require equal bootstrap counts because member
data diversity is intentional.

## Experiment Runner

The focused runner accepts:

- dataset path;
- exactly one existing seed-only H10 baseline checkpoint per training seed;
- output directory;
- training seeds, defaulting to `0 1 2 3 4`;
- fixed split seed and current H10 training hyperparameters;
- ensemble diagnostic horizons, windows per episode, and calibration bins.

The first implementation and the real five-member run train serially for
deterministic orchestration and simple failure reporting. Parallel scheduling
is a later performance optimization and is not part of the bootstrap data
contract.

The runner sorts members by training seed and validates that every baseline
seed has exactly one bootstrap counterpart. It rejects any mismatch in dataset
hash, split arrays, normalizers, hidden size, H10 objective, or diagnostic
protocol before producing a comparison.

## Diagnostic Comparison

The runner invokes the existing ensemble diagnostic for:

```text
baseline_diagnostics/
bootstrap_diagnostics/
```

For each of `position`, `heading_degrees`, `velocity`, and
`normalized_total`, the comparison records:

- one-step ensemble error mean;
- one-step disagreement/error Pearson correlation;
- rollout ensemble error at each requested horizon;
- rollout disagreement mean at each requested horizon;
- rollout disagreement/error Pearson correlation at each requested horizon;
- bootstrap minus baseline deltas for every finite comparable value.

If either correlation is JSON `null`, its delta is also `null`. No NaN or
infinity is serialized. Positive correlation delta means bootstrap improves
risk ranking; negative ensemble-error delta means bootstrap improves mean
prediction accuracy. These signs are documented explicitly because the two
metrics use different notions of improvement.

The experiment does not use a pass/fail threshold for scientific results.
Bootstrap may increase diversity without improving calibration, and that must
be reported as a valid negative result.

## Artifact Bundle

The output directory must be absent or empty. A successful run contains:

```text
experiment_manifest.json
comparison.json
comparison.csv
comparison.png
baseline_diagnostics/
bootstrap_diagnostics/
runs/
  seed_0/world_model.pt
  seed_1/world_model.pt
  seed_2/world_model.pt
  seed_3/world_model.pt
  seed_4/world_model.pt
```

The manifest records dataset path/hash, baseline checkpoint paths/hashes,
bootstrap checkpoint paths/hashes, seeds, split seed, H10 configuration,
bootstrap protocol, and diagnostic configuration. Paths inside the output
bundle are relative; external dataset and baseline paths are absolute.

`comparison.csv` uses one row per evaluation kind, metric, and horizon. A
one-step row has an empty horizon field. `comparison.png` contains paired
baseline/bootstrap curves for ensemble error and disagreement/error
correlation; null correlations are rendered as gaps.

## Validation and Failure Handling

Validation occurs before expensive training whenever possible:

- dataset and baseline checkpoint paths must be regular files;
- output directory must be absent or empty;
- seeds must be unique non-negative integers and match baseline checkpoint
  seeds exactly;
- all protocol counts and horizons must be positive integers;
- learning rate and rollout loss weight must be finite and valid;
- baseline members must satisfy existing H10 ensemble compatibility;
- the dataset must contain every checkpoint split episode;
- the full training split must contain enough data to fit all normalizer
  dimensions;
- bootstrap-expanded one-step data and H10 windows must be non-empty.

Each bootstrap seed writes to a disjoint directory. If training or diagnostics
fails, the runner raises the original error and does not write final comparison
or manifest files. Existing partial run directories remain available for
inspection; the output directory cannot be reused without explicit cleanup,
matching the current multi-seed experiment convention.

## Test Strategy

Development follows red-green-refactor.

Pure bootstrap tests cover:

- deterministic exact draws for a fixed seed;
- draw length equal to source episode count;
- repeated draws repeating complete transition index groups;
- zero-count episodes retained in provenance;
- invalid IDs and seeds rejected by name.

Training integration tests cover:

- legacy `bootstrap_seed=None` behavior;
- identical full-train normalizers for baseline and bootstrap checkpoints;
- identical train/validation/test split arrays;
- validation/test evaluation using unique held-out transitions;
- identical episode multiplicity in expanded transitions and H10 windows;
- complete checkpoint bootstrap provenance and count sums;
- deterministic repeatability for the same seeds.

Experiment tests cover:

- a two-member tiny end-to-end H10 run;
- complete manifest, JSON, CSV, PNG, diagnostics, and checkpoint outputs;
- seed/config/split/normalizer mismatch rejection;
- JSON-safe null correlations and null deltas;
- documented delta signs and stable row ordering;
- no final comparison files after a training or diagnostic failure.

Final verification runs the focused tests, the full unit-test suite,
`compileall`, and `git diff --check`, followed by the real five-member H10
experiment.

## Success Criteria

Engineering success requires:

- deterministic and auditable episode bootstrap sampling;
- no validation/test leakage;
- common normalization and compatible ensemble checkpoints;
- complete reproducible artifacts;
- all tests and static verification passing.

Scientific evidence is considered encouraging only if:

- H1 ensemble error does not regress materially;
- H10 and longer disagreement/error correlation becomes more positive for the
  main physical metrics;
- calibration improvement is not explained solely by a large increase in
  ensemble prediction error;
- the result is consistent across more than one horizon and metric.

If long-horizon correlation remains weak or negative, the conclusion is that
episode bootstrap alone does not solve common-mode rollout drift. The next
candidate is an explicit probabilistic world model or a richer source of model
diversity, not an MPC uncertainty penalty.

## Risks

- Unequal episode lengths make expanded transition counts vary across members.
  This is intrinsic to episode bootstrap and is surfaced in checkpoint
  metadata rather than silently normalized away.
- Some members may omit training episodes that cover rare states. That can
  improve epistemic diversity but also degrade mean accuracy; both calibration
  and error are reported.
- Sharing the full-train normalizer slightly couples members through summary
  statistics. This is intentional so disagreement uses one common scale and
  the experiment isolates loss-data diversity.
- Five members provide limited correlation evidence. The experiment is a
  diagnostic learning step, not a production uncertainty guarantee.
