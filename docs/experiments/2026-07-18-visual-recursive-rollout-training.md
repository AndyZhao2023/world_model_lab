# Visual recursive rollout training

## Status

Completed. The candidate passed three of four metric gates, so the
pre-registered decision is negative and the source checkpoint remains the
default.

## Question

Can an H5 differentiable free-rollout objective reduce recursive latent drift
and cumulative moving-object error while preserving the current
objective-aligned model's one-step quality?

## Controlled change

Source checkpoint:

```text
artifacts/visual_latent_spatial8_objective_w01.pt
```

The following remain fixed:

- visual dataset and dataset SHA-256;
- train, validation, and test episode IDs;
- spatial autoencoder weights and `[B, 8, 8, 8]` latent layout;
- latent and action normalizers;
- four-frame context and aligned action convention;
- spatial CNN dynamics with 64 hidden channels;
- training seed `0`, learning rate `1e-3`, batch size `256`, and 50 epochs;
- one-step changed-pixel weight `0.1`;
- held-out one-step and H1/H5/H10 diagnostic protocols.

The dynamics weights are freshly initialized with the source seed. They are
not fine-tuned from the source dynamics.

Only one training term is added:

```text
total loss
  = one-step objective
  + 1.0 * H5 recursive rollout objective

one-step objective
  = normalized latent MSE
  + 0.1 * decoded changed-pixel MAE

H5 recursive rollout objective
  = mean over steps 1..5 of:
      normalized latent MSE
      + 0.1 * decoded changed-pixel MAE
```

At every rollout step, the predicted latent is appended to the next four-frame
context, so later losses backpropagate through earlier predictions. Actions
shift forward with the context. Each changed-pixel mask compares adjacent true
frames for that step; it is supervision only and is not a model input.

## Dataset windows

Every valid start is retained without padding or crossing an episode:

| Split | Episodes | One-step windows | H5 windows |
|---|---:|---:|---:|
| Train | 200 | 6,330 | 5,657 |
| Validation | 25 | 803 | 713 |
| Test | 25 | 920 | 830 |

## Source digests

```text
2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba  data/visual_episodes.npz
5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369  artifacts/visual_latent_spatial8_objective_w01.pt
```

## Pre-registered gates

Reference metrics use only episodes long enough for H10, at most eight
evenly-spaced windows per episode, and equal episode weighting:

| Metric | Current H1-trained source |
|---|---:|
| H1 free normalized latent MSE | 0.0362509983 |
| H1 free cumulative changed-pixel MAE | 0.3034423844 |
| H5 free normalized latent MSE | 0.3129600827 |
| H5 free cumulative changed-pixel MAE | 0.2759844311 |
| H10 free normalized latent MSE | 0.8571245613 |
| H10 free cumulative changed-pixel MAE | 0.2991328294 |

The candidate passes only if both H5 primary gates pass:

```text
H5 free normalized latent MSE < 0.3129600827
H5 free cumulative changed-pixel MAE < 0.2759844311
```

and both H1 stability gates pass:

```text
H1 free normalized latent MSE <= 0.0398760981
H1 free cumulative changed-pixel MAE <= 0.3186145036
```

Representation invariance additionally requires exact source/candidate
autoencoder tensor equality and exact normalizer/split equality. H10 is a
secondary generalization result because training stops at H5.

If either H5 primary gate fails, the experiment is negative and the current
source remains preferred. No horizon or rollout-weight sweep will be started
inside this run.

## Reproduction commands

Training:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_dynamics_recursive \
  --data data/visual_episodes.npz \
  --source-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01.pt \
  --output artifacts/visual_latent_spatial8_objective_w01_h5.pt \
  --preview \
    artifacts/visual_latent_spatial8_objective_w01_h5_predictions.png \
  --changed-pixel-loss-weight 0.1 \
  --rollout-horizon 5 \
  --rollout-loss-weight 1.0 \
  --dynamics-epochs 50 \
  --dynamics-batch-size 256
```

Diagnostics:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_rollout \
  --data data/visual_episodes.npz \
  --baseline-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01.pt \
  --aligned-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01_h5.pt \
  --output-dir artifacts/diagnostics/visual-rollout-h5-comparison \
  --horizons 1 5 10 \
  --windows-per-episode 8 \
  --counterfactual-seeds 0 1 2 3 4 5 6 7 8 9
```

## Results

Training selected epoch `22` with total validation loss `0.2285969728`:

```text
validation one-step loss  0.0714106310
validation H5 loss        0.1571863417
```

The controlled diagnostic retained 18 H10-capable test episodes and 136
episode-equal windows. Lower is better in every table cell.

Free-rollout results:

| Horizon | Metric | Source H1 | H5 candidate | Relative change |
|---:|---|---:|---:|---:|
| 1 | normalized latent MSE | 0.0362509983 | 0.0328628360 | -9.35% |
| 1 | cumulative changed-pixel MAE | 0.3034423844 | 0.3068128490 | +1.11% |
| 5 | normalized latent MSE | 0.3129600827 | 0.2189544909 | -30.04% |
| 5 | cumulative changed-pixel MAE | 0.2759844311 | 0.2767157306 | +0.265% |
| 10 | normalized latent MSE | 0.8571245613 | 0.5937570375 | -30.73% |
| 10 | cumulative changed-pixel MAE | 0.2991328294 | 0.2932146689 | -1.98% |

Teacher-forced results:

| Horizon | Metric | Source H1 | H5 candidate | Relative change |
|---:|---|---:|---:|---:|
| 1 | normalized latent MSE | 0.0362509983 | 0.0328628360 | -9.35% |
| 1 | cumulative changed-pixel MAE | 0.3034423844 | 0.3068128490 | +1.11% |
| 5 | normalized latent MSE | 0.0351926187 | 0.0332450700 | -5.53% |
| 5 | cumulative changed-pixel MAE | 0.2176932547 | 0.2196959777 | +0.920% |
| 10 | normalized latent MSE | 0.0510044302 | 0.0519381637 | +1.83% |
| 10 | cumulative changed-pixel MAE | 0.2012868287 | 0.2027314127 | +0.718% |

Pre-registered gates:

| Gate | Limit | Candidate | Result |
|---|---:|---:|---|
| H5 free latent MSE | `< 0.3129600827` | 0.2189544909 | PASS |
| H5 free cumulative changed-pixel MAE | `< 0.2759844311` | 0.2767157306 | **FAIL** |
| H1 free latent MSE | `<= 0.0398760981` | 0.0328628360 | PASS |
| H1 free cumulative changed-pixel MAE | `<= 0.3186145036` | 0.3068128490 | PASS |

The H5 candidate's counterfactual normalized-latent divergence is `0.974x`,
`1.110x`, and `1.071x` the source at horizons 1, 5, and 10. Its decoded
pixel-MAE divergence is `0.959x`, `1.073x`, and `1.059x`. Thus recursive
training modestly increases long-horizon action sensitivity, but this
sensitivity still does not establish counterfactual accuracy.

Exact invariance checks passed for the autoencoder state, latent/action
normalizers, episode splits, dataset SHA-256, and model configuration.
Independent diagnostic publication produced byte-identical manifest, metrics,
and PNG files.

Generated artifact digests:

```text
4764d89d7399694574ad91a6703c90872d9c0f7c1b6e7743572dc58d3ff0e424  artifacts/visual_latent_spatial8_objective_w01_h5.pt
9173bff0a879a7fd4e392cf476b43d6144a072aba8cc4869dc91f831208cd029  artifacts/diagnostics/visual-rollout-h5-comparison/metrics.json
db0b88c43b2a60d4f63b9494da1ce6ae17a49bc1df97dd13be0416eef368a267  artifacts/diagnostics/visual-rollout-h5-comparison/visual_rollout_comparison.png
```

## Decision

Do not promote the H5 candidate. It substantially reduces recursive latent
drift and improves the secondary H10 image metric, but it misses the primary H5
cumulative changed-pixel gate by `0.0007313` absolute (`0.265%`). The current
objective-aligned H1 checkpoint remains the default.

This is evidence that the recursive objective helps state-space stability but
does not yet reliably improve the moving-object pixel target at the trained
horizon. Do not connect either checkpoint to MPC. Before any horizon or weight
sweep, the next experiment should add simulator-matched counterfactual targets
or a more direct object-motion objective so that action-conditioned trajectory
quality can be measured rather than inferred from sensitivity alone.
