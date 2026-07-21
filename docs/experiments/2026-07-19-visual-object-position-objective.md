# Visual object-position objective

## Status

Completed. The candidate passed five of seven pre-registered gates, so the
strict decision is negative and the H1 source remains the default.

## Question

Can direct simulator-state supervision of the car centre close the H5 matched
cumulative moving-pixel gap while preserving the recursive candidate's latent
trajectory and action-effect gains?

## Why this target

The existing H5 recursive candidate improves matched H5 latent MSE by `27.40%`,
latent action-effect MSE by `15.63%`, full pixel MSE by `10.48%`, and
transition changed-pixel MAE by `3.66%`. It still worsens cumulative
changed-pixel MAE by `0.093%`, so it is not promoted.

Two object-target feasibility checks were run before implementing or training
the new candidate:

- extracting car centre from the frozen decoder by renderer colour gave about
  `18 px` mean held-out error and was rejected;
- a ridge-linear probe from normalized frozen latents to normalized `(x, y)`
  gave validation mean centre error `0.2701734298` world units, approximately
  `1.70 px`, and was accepted.

The same linear probe's validation heading error was approximately `65.85°`.
Heading supervision is therefore excluded from this experiment.

## Controlled change

Source checkpoint:

```text
artifacts/visual_latent_spatial8_objective_w01.pt
```

Fixed source digests:

```text
2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba  data/visual_episodes.npz
5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369  artifacts/visual_latent_spatial8_objective_w01.pt
4764d89d7399694574ad91a6703c90872d9c0f7c1b6e7743572dc58d3ff0e424  artifacts/visual_latent_spatial8_objective_w01_h5.pt
```

The following remain fixed:

- visual data, spatial autoencoder, latent/action normalizers, and split IDs;
- `[B, 8, 8, 8]` spatial latent layout and 64-channel CNN dynamics;
- four-frame context and action alignment;
- fresh dynamics initialization from seed `0`;
- learning rate `1e-3`, batch size `256`, and 50 epochs;
- one-step changed-pixel loss weight `0.1`;
- H5 recursive loss weight `1.0`;
- held-out H1/H5/H10 matched simulator protocol.

Only one auxiliary term is added:

```text
one-step objective
  = normalized latent MSE
  + 0.1 * decoded changed-pixel MAE
  + 1.0 * frozen-probe normalized position MSE

H5 recursive objective
  = mean over steps 1..5 of:
      normalized latent MSE
      + 0.1 * decoded changed-pixel MAE
      + 1.0 * frozen-probe normalized position MSE

total loss
  = one-step objective + 1.0 * H5 recursive objective
```

World `x` and `y` are independently mapped to `[-1, 1]`. The affine probe is
fit once from source-normalized autoencoder latents and physical states for
training frames only, using ridge coefficient `1e-3`. Probe parameters and the
autoencoder are frozen while dynamics trains. At inference, the world model
still consumes only visual latent history and actions; simulator state is
privileged training supervision, not an inference input.

## Probe feasibility and limitations

The pre-implementation probe check gave:

| Split | normalized XY MSE | mean world error | approximate mean pixel error |
|---|---:|---:|---:|
| train | 0.0016450676 | 0.1659037215 | 1.0452 |
| validation | 0.0041914316 | 0.2701734298 | 1.7021 |
| test | 0.0043880831 | 0.2692668018 | 1.6964 |

This is a privileged-state auxiliary objective. It tests whether explicit
object location improves this lab's world model; it does not establish that
the same supervision is available in an uninstrumented real-world setting.
The fixed test population has already been used by previous experiments and
should be treated as a stable benchmark, not a never-observed final test set.

## Pre-registered decision gates

The new candidate is compared with the current H1 default under the existing
ten-seed matched simulator protocol. It passes only if all gates pass:

```text
H5 primary:
    matched normalized latent MSE strictly improves
    matched cumulative changed-pixel MAE strictly improves
    normalized latent action-effect MSE strictly improves
    matched normalized position MSE strictly improves

H1 stability:
    matched normalized latent MSE <= 1.10 * source
    matched cumulative changed-pixel MAE <= 1.05 * source

object-loss ablation:
    H5 matched normalized position MSE < 0.0969399159
    # strictly better than the old H5 recursive candidate
```

H10 remains secondary generalization evidence. The position metric uses the
training-only frozen probe on predicted latents and compares it with the real
simulator state. Existing terminal masks and episode-equal aggregation remain
unchanged.

Before the new candidate is trained, the extended diagnostic will be run on
the existing H1 and H5 checkpoints to record exact source position baselines.
No position-weight, ridge, seed, horizon, or rollout-weight sweep is allowed
inside this run. Missing any gate keeps the H1 source as default and prohibits
MPC integration.

## Locked extended baseline

The extended diagnostic was run before creating the new checkpoint, using the
same 18 episodes, 136 windows, ten Sattolo seeds, matched simulator branches,
and terminal masks as the previous experiment.

Frozen probe:

```text
ridge       0.001
fit split   7,096 training frames
SHA-256     c278d5a55265dd4cea257ac8f0e804d66e36707747ff9674bce130975f9ac4bc
validation normalized position MSE  0.0041914214
validation mean world error          0.2701734335
```

Matched object-position baselines:

| Horizon | Metric | H1 source | old H5 candidate | Relative change |
|---:|---|---:|---:|---:|
| 1 | normalized position MSE | 0.0119640599 | 0.0087164068 | -27.15% |
| 1 | mean world-position error | 0.5558522158 | 0.4787780501 | -13.87% |
| 1 | normalized position-effect MSE | 0.0005385105 | 0.0004363482 | -18.97% |
| 5 | normalized position MSE | 0.1468615915 | 0.0969399159 | -33.99% |
| 5 | mean world-position error | 1.8603952243 | 1.4618952287 | -21.42% |
| 5 | normalized position-effect MSE | 0.0135459489 | 0.0148971685 | +9.98% |
| 10 | normalized position MSE | 0.2464327093 | 0.1651733344 | -32.97% |
| 10 | mean world-position error | 2.5580205691 | 2.0891271084 | -18.33% |
| 10 | normalized position-effect MSE | 0.1206615872 | 0.1068241739 | -11.47% |

The old H5 candidate already improves direct centre prediction, so merely
beating H1 on that metric would not isolate the new objective. The additional
strict H5 ablation gate was therefore locked before training.

Baseline bundle digests:

```text
740fb894a22e1acbf755ed151983b85e1c3cf9b04d5577f82a4adb4acbb5015c  manifest.json
b60fdead069429c42afb58d52f04a176c81d59cbe8665557157c5f9e1fa41899  metrics.json
8ae3e258773d6b368a653618304285a1925b5a5cb41b49f8962eb1aec34932b7  matched_counterfactual_comparison.png
```

## Registered outputs

```text
artifacts/visual_latent_spatial8_objective_w01_h5_position_w1.pt
artifacts/visual_latent_spatial8_objective_w01_h5_position_w1_predictions.png
artifacts/diagnostics/visual-matched-counterfactual-h5-position-comparison/
```

## Results

Training used the registered 50 epochs and selected epoch `50`:

```text
initial train total loss      0.9076096460
final train total loss        0.1946402213
best validation total loss    0.2768245896
validation one-step loss      0.0799548253
validation H5 rollout loss    0.1968697643
```

The final five validation totals were:

```text
0.2773935432
0.2801128172
0.2779232700
0.2880015095
0.2768245896
```

Thus epoch 50 was the minimum under a fluctuating validation curve, not one
point on a monotonic descent. The fixed budget was not extended.

Exact invariance checks passed for source/candidate autoencoder tensors,
latent/action normalizers, split IDs, dataset SHA-256, and probe SHA-256.

### Matched results versus the H1 default

Lower is better in every metric:

| Horizon | Metric | H1 source | position candidate | Relative change |
|---:|---|---:|---:|---:|
| 1 | normalized latent MSE | 0.0369668000 | 0.0380781717 | +3.01% |
| 1 | cumulative changed-pixel MAE | 0.3065254702 | 0.3208943091 | +4.69% |
| 1 | normalized latent action-effect MSE | 0.0013556197 | 0.0010346598 | -23.68% |
| 1 | normalized position MSE | 0.0119640599 | 0.0029396776 | -75.43% |
| 1 | mean world-position error | 0.5558522158 | 0.2889952975 | -48.01% |
| 5 | normalized latent MSE | 0.3065971604 | 0.3008744463 | -1.87% |
| 5 | cumulative changed-pixel MAE | 0.2764975452 | 0.3062392396 | +10.76% |
| 5 | normalized latent action-effect MSE | 0.1209001768 | 0.1225172304 | +1.34% |
| 5 | normalized position MSE | 0.1468615915 | 0.0234369108 | -84.04% |
| 5 | mean world-position error | 1.8603952243 | 0.8093508062 | -56.50% |
| 5 | normalized position-effect MSE | 0.0135459489 | 0.0033372966 | -75.36% |
| 5 | pixel MSE | 0.0022084709 | 0.0021713761 | -1.68% |
| 5 | transition changed-pixel MAE | 0.3467591122 | 0.3578470739 | +3.20% |
| 10 | normalized latent MSE | 0.7903406148 | 0.6966450908 | -11.86% |
| 10 | cumulative changed-pixel MAE | 0.3044660457 | 0.3219439886 | +5.74% |
| 10 | normalized position MSE | 0.2464327093 | 0.0758211182 | -69.23% |
| 10 | mean world-position error | 2.5580205691 | 1.4289610277 | -44.14% |

### Controlled comparison with the old H5 candidate

At H5, the new object-position objective changed the old recursive candidate:

| Metric | old H5 | position candidate | Relative change |
|---|---:|---:|---:|
| normalized position MSE | 0.0969399159 | 0.0234369108 | -75.82% |
| mean world-position error | 1.4618952287 | 0.8093508062 | -44.64% |
| normalized position-effect MSE | 0.0148971685 | 0.0033372966 | -77.60% |
| normalized latent MSE | 0.2225983635 | 0.3008744463 | +35.16% |
| cumulative changed-pixel MAE | 0.2767559394 | 0.3062392396 | +10.65% |
| normalized latent action-effect MSE | 0.1020056328 | 0.1225172304 | +20.11% |
| pixel MSE | 0.0019769668 | 0.0021713761 | +9.83% |
| transition changed-pixel MAE | 0.3340672414 | 0.3578470739 | +7.12% |

This isolates the effect: the position objective strongly improves the
position subspace read by the frozen probe, but trades away broad latent and
decoded-image quality.

### Gates

| Gate | Limit/reference | Candidate | Result |
|---|---:|---:|---|
| H5 matched latent MSE strictly improves over H1 | 0.3065971604 | 0.3008744463 | PASS |
| H5 cumulative changed-pixel MAE strictly improves over H1 | 0.2764975452 | 0.3062392396 | **FAIL** |
| H5 latent action-effect MSE strictly improves over H1 | 0.1209001768 | 0.1225172304 | **FAIL** |
| H5 normalized position MSE strictly improves over H1 | 0.1468615915 | 0.0234369108 | PASS |
| H1 matched latent MSE <= 110% source | 0.0406634801 | 0.0380781717 | PASS |
| H1 cumulative changed-pixel MAE <= 105% source | 0.3218517437 | 0.3208943091 | PASS |
| H5 normalized position MSE improves over old H5 | 0.0969399159 | 0.0234369108 | PASS |

Coverage was unchanged: 1,360/1,360 valid branches through H5 and
1,216/1,360 through H10. There were no clipped action steps. Terminal totals
remained 49 collisions, five goals, and 143 out-of-bounds branches.

Independent diagnostic execution produced byte-identical manifest, metrics,
and plot files.

Published artifact digests:

```text
16aef392118c8e8781fb7dafa7749751da8d856fbf7a42fc8b02775c09d79846  artifacts/visual_latent_spatial8_objective_w01_h5_position_w1.pt
eb6ad689e404462ccc6461348606465895d87e473bf3eb3ab78709a945bf8610  artifacts/visual_latent_spatial8_objective_w01_h5_position_w1_predictions.png
7bcd4a525d3a16422e1ca31250c706a46ae212d29753743d0dff92aa61b9b10f  artifacts/diagnostics/visual-matched-counterfactual-h5-position-comparison/manifest.json
0e0ceaf66bc09d83c4bb667d77cf2fa53cac47645347b346d612052dad4dac3b  artifacts/diagnostics/visual-matched-counterfactual-h5-position-comparison/metrics.json
a4f62397497fa6dba4572d5496cc76fd7f7045bb53848ae26132892eb33ebfcf  artifacts/diagnostics/visual-matched-counterfactual-h5-position-comparison/matched_counterfactual_comparison.png
```

## Decision

Do not promote the object-position candidate. It learns a much more accurate
matched car-centre trajectory and position action effect, so the auxiliary
signal is effective in its own probe space. However, the frozen linear probe
can be satisfied by latent changes that the frozen RGB decoder does not turn
into accurate moving-car pixels. H5 cumulative changed-pixel MAE worsens
`10.76%` versus the H1 source and `10.65%` versus the old H5 candidate.

Do not reinterpret the improved probe metric as improved decoded visual
prediction, and do not connect this checkpoint to MPC under the registered
rule.

The next experiment should repair object alignment in the representation and
decoder before another dynamics-objective run. A minimal direction is a new
autoencoder trained with renderer-derived car mask/heatmap supervision plus
RGB reconstruction, with an oracle object-mask/centre gate checked before
freezing it and retraining H5 dynamics. A position-loss weight sweep is not
justified while the current decoder remains the broken boundary.
