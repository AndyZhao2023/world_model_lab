# Rollout Step Diagnostics Design

## Goal

Extend the held-out world-model diagnostic bundle with dense per-step error
curves and normalized loss-component curves. The new evidence must explain
when recursive predictions begin to diverge and which predicted state
component contributes to the rollout objective, without changing the model,
training loop, dataset split, or fixed-horizon benchmark.

## Motivation

The current diagnostic report evaluates teacher forcing and free rollout only
at configured horizons such as 1, 5, 10, 20, and 50. Those points show the
final effect of compounding error, but they do not reveal whether an error
began early and grew gradually or appeared only after a longer recursive
rollout.

The current physical metrics also combine `x` and `y` into position error and
do not expose how the four state dimensions contribute to the normalized
rollout MSE used during training. This prevents the horizon-1 and horizon-10
experiment from explaining why long-horizon position and velocity improved
while heading became worse.

## Scope

This stage adds:

- dense step curves from step 1 through the configured maximum horizon;
- physical-unit curves for position, heading, and velocity;
- normalized squared-error curves for `x`, `y`, `heading`, and `velocity`;
- a normalized total-MSE curve matching the four-component rollout objective;
- a dedicated component-loss plot;
- backward-compatible plotting for existing schema-version-1 metric files;
- documentation and deterministic regression tests;
- a controlled rerun of the horizon-1 and horizon-10 diagnostic bundles.

This stage does not change `WorldModelMLP`, checkpoint contents, training loss,
optimizer behavior, sequence sampling, dataset splits, diagnostic CLI options,
MPC, PPO, rewards, or termination prediction. It does not introduce a general
metric-plugin system, confidence bands, or per-latent-dimension analysis.

## Architecture

Dense step diagnostics remain inside the existing held-out diagnostic path:

```text
diagnose_model.py
    -> build_diagnostic_metrics()
        -> select the existing fixed max-horizon test windows
        -> compute each teacher-forced and free-rollout trajectory once
        -> summarize every step using episode-macro aggregation
        -> preserve the configured sparse-horizon summaries
    -> metrics.json
    -> rollout_errors.png
    -> rollout_loss_components.png
```

The existing `window_predictions` values already contain predictions for every
step up to the maximum horizon. Dense summaries reuse those arrays and must not
invoke the model again. This keeps diagnostic inference cost unchanged apart
from small NumPy aggregation and plotting work.

## Error Definitions

For a prediction at rollout step `k`, define physical state error as:

\[
e_k =
[\hat{x}_k-x_k,
 \hat{y}_k-y_k,
 \operatorname{wrap}(\hat{\theta}_k-\theta_k),
 \hat{v}_k-v_k].
\]

Heading wrapping uses the existing half-open `[-pi, pi)` convention.

### Physical Metrics

The dense physical curves retain the current user-facing units:

\[
\operatorname{position}_k =
\sqrt{e_{k,x}^2 + e_{k,y}^2},
\]

\[
\operatorname{heading\_degrees}_k =
|e_{k,\theta}| \cdot 180 / \pi,
\]

\[
\operatorname{velocity}_k = |e_{k,v}|.
\]

### Normalized Loss Components

To match the multi-step training objective, divide each physical state error
by the checkpoint's target-delta standard deviation and square it:

\[
\ell_{k,d} =
\left(\frac{e_{k,d}}{\sigma_{\Delta s,d}}\right)^2,
\qquad
d \in \{x,y,\theta,v\}.
\]

The per-step total is the arithmetic mean of the four component values:

\[
\ell_{k,\mathrm{total}} =
\frac{1}{4}
\sum_d \ell_{k,d}.
\]

`target_std` must have shape `[4]` and contain only finite positive values.
This is already true for valid checkpoints, but the diagnostic boundary must
reject invalid normalizer data with a clear `ValueError` rather than emitting
NaN or infinity into JSON.

## Episode-Macro Aggregation

Every selected window has the same configured maximum length, so every dense
step uses the same eligible episodes and the same windows. For each mode,
step, and metric:

1. compute the value for every selected window;
2. average all windows belonging to one episode;
3. average those per-episode means with equal episode weight.

This preserves the current diagnostic invariant that long episodes do not
receive more weight merely because they supply more candidate windows.

The dense curves store macro means only. The existing sparse-horizon entries
continue to store distribution summaries such as count, standard deviation,
median, p95, and maximum. This avoids duplicating a large nested statistics
payload for all 50 steps while keeping detailed inspection at the configured
benchmark horizons.

## Metrics Schema Version 2

The top-level `schema_version` changes from `1` to `2`. All existing keys and
their meanings remain unchanged. A new `rollout.step_curves` value is added:

```json
{
  "schema_version": 2,
  "rollout": {
    "protocol": {
      "horizons": [1, 5, 10, 20, 50],
      "max_horizon": 50
    },
    "horizons": {},
    "step_curves": {
      "steps": [1, 2, 3, 4, 5],
      "aggregation": "episode_macro_mean",
      "teacher_forcing": {
        "physical": {
          "position": [0.0, 0.0, 0.0, 0.0, 0.0],
          "heading_degrees": [0.0, 0.0, 0.0, 0.0, 0.0],
          "velocity": [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "normalized_mse": {
          "x": [0.0, 0.0, 0.0, 0.0, 0.0],
          "y": [0.0, 0.0, 0.0, 0.0, 0.0],
          "heading": [0.0, 0.0, 0.0, 0.0, 0.0],
          "velocity": [0.0, 0.0, 0.0, 0.0, 0.0],
          "total": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
      },
      "free_rollout": {
        "physical": {
          "position": [0.0, 0.0, 0.0, 0.0, 0.0],
          "heading_degrees": [0.0, 0.0, 0.0, 0.0, 0.0],
          "velocity": [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "normalized_mse": {
          "x": [0.0, 0.0, 0.0, 0.0, 0.0],
          "y": [0.0, 0.0, 0.0, 0.0, 0.0],
          "heading": [0.0, 0.0, 0.0, 0.0, 0.0],
          "velocity": [0.0, 0.0, 0.0, 0.0, 0.0],
          "total": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
      }
    }
  }
}
```

The example shows five values for readability. Actual arrays always contain
exactly `max_horizon` finite floats and `steps` is exactly
`[1, 2, ..., max_horizon]`.

The names under `physical` and `normalized_mse` are grouped by semantic
purpose rather than placed in one flat namespace. A future latent world model
can add representation, reward, value, or task metrics without redefining the
meaning of the current physical-state curves. This stage does not implement
those future metrics.

## Computation Boundary

`diagnostics.py` owns all numerical work. It adds focused internal helpers for:

- wrapped four-dimensional state differences;
- normalized squared-error components;
- per-step, per-episode macro aggregation;
- constructing the JSON-safe `step_curves` value.

The existing sparse `rollout.horizons` summaries and the new dense curves must
be computed from the same cached teacher-forced and free-rollout predictions.
Their values at a configured horizon must therefore agree for the shared
physical metric means, within floating-point tolerance.

`diagnostic_plots.py` remains presentation-only. It must not recompute errors,
normalization, or aggregation from raw trajectories.

## Visualization

### Dense Physical Error Plot

`plot_rollout_errors()` uses schema-version-2 `step_curves` when available and
draws all steps from 1 through `max_horizon` for:

- position error in metres;
- heading error in degrees;
- velocity error in metres per second.

Teacher forcing remains blue and free rollout remains red. The curves are
dense lines; configured sparse horizons may be shown as markers but are not a
second data source.

For a schema-version-1 metrics mapping without `step_curves`, the function
falls back to the existing sparse-horizon behavior. This preserves the ability
to plot diagnostic JSON generated before this change.

### Component Loss Plot

A new `plot_rollout_loss_components()` writes
`rollout_loss_components.png`. It contains a 2-by-2 grid for normalized MSE of
`x`, `y`, `heading`, and `velocity`. Each panel compares teacher forcing and
free rollout over every step. The plot uses a non-negative linear y-axis by
default so zero values remain visible; it does not silently switch to log
scale when a component contains zero.

The total normalized-MSE curve remains in `metrics.json` for comparison and
automation, but is not added as a fifth panel. The four component panels are
the diagnostic target of this stage.

## Diagnostic Bundle and CLI

`run_diagnostics()` writes the new image beside the existing artifacts and
adds its path to the returned summary:

```text
metrics.json
manifest.json
overview.png
rollout_errors.png
rollout_loss_components.png
```

The existing CLI and flags remain unchanged. `--horizons` still defines sparse
benchmark reporting points, and its largest value defines the dense curve
length. For example, `--horizons 1 5 10 20 50` produces sparse summaries at
those five horizons and dense curves for every step from 1 through 50.

The manifest retains the configured horizons and other protocol values. No
separate dense-step CLI flag is added because a shorter curve can already be
requested by choosing a smaller maximum horizon.

## Compatibility and Error Handling

- Existing `rollout.protocol` and `rollout.horizons` structures do not change.
- Existing callers that ignore unknown JSON keys continue to work.
- Plotting accepts both schema version 1 and schema version 2 mappings.
- All generated values must pass `json.dumps(..., allow_nan=False)`.
- Missing `step_curves` is allowed only for the plotting fallback, not for a
  newly generated schema-version-2 report.
- Empty selections and impossible maximum horizons continue to use the
  existing validation errors.
- Invalid checkpoint target standard deviations fail before metric output is
  written.

## Testing Strategy

Tests remain deterministic and use small synthetic trajectories.

`tests/test_diagnostics.py` verifies:

- dense steps cover exactly 1 through the maximum horizon;
- free-rollout physical curves expose recursive compounding error;
- heading component MSE uses wrapped radians rather than raw angle difference;
- normalized component values use checkpoint `target_std` exactly;
- total MSE equals the mean of the four component curves;
- step curves preserve equal episode weighting;
- sparse-horizon physical means agree with the dense curve at shared steps;
- schema-version-2 output is finite JSON.

`tests/test_diagnostic_plots.py` verifies:

- dense physical curves are used when `step_curves` is present;
- schema-version-1 sparse mappings still produce a valid PNG;
- the new component plot writes a PNG and draws all four named components.

`tests/test_diagnose_model.py` verifies:

- the bundle contains `rollout_loss_components.png`;
- the returned summary exposes the new path;
- repeated runs remain byte-reproducible for JSON and manifest artifacts.

The complete unit suite must pass after focused tests. A real-data smoke run
regenerates horizon-1 and horizon-10 diagnostic bundles with the same dataset,
test split, horizons, and window-selection parameters used in the previous
controlled experiment. Generated artifacts remain ignored by Git.

## Acceptance Criteria

The change is complete when:

- a default diagnostic run stores 50 dense steps without additional model
  inference beyond the existing cached trajectories;
- every dense series has finite length-50 values and uses episode-macro means;
- physical curves and normalized MSE component curves are both available;
- the four component values explain the total normalized rollout objective;
- schema-version-1 plots still work;
- the diagnostic bundle includes both rollout plots;
- all focused and full tests pass;
- real horizon-1 and horizon-10 bundles identify the first step at which the
  heading free-rollout curve becomes worse, equal, or better than the baseline;
- no generated checkpoint, metrics, or image artifact is committed.
