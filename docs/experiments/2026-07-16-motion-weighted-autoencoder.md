# Motion-weighted autoencoder experiment

## Question

Can an image-derived motion mask make the global-latent autoencoder preserve
the small moving car without materially degrading the static scene?

## Controlled change

The architecture, dataset, episode split, training seed, training duration,
batch sizes, and dynamics model were held fixed. The only changed training
parameter was `motion_loss_weight`.

For current frame `x_t` and preceding frame `x_{t-1}`, the spatial mask is:

```text
motion_mask = any_rgb_channel(x_t != x_{t-1})
```

The reconstruction objective is:

```text
weight = 1 + motion_loss_weight * motion_mask
loss = sum(weight * squared_pixel_error) / sum_expanded_rgb_weights
```

The first frame of every episode compares with itself, so its mask is zero.
The mask never crosses an episode boundary and does not use `states`, rewards,
dones, or hand-authored car labels.

## Protocol

- Dataset: `data/visual_episodes.npz`
- Dataset SHA-256:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- Train/validation/test episode split seed: `42`
- Training seed: `0`
- Autoencoder: `latent_dim=32`, `base_channels=16`, `20` epochs
- Latent dynamics: hidden size `256`, `50` epochs
- Train/validation/test windows: `6330 / 803 / 920`
- Device: CPU

Training motion masks marked `61,426` of `29,065,216` spatial pixels, or
`0.2113%`. Weight `100` made motion pixels contribute approximately `17.6%`
of the effective reconstruction objective. Weight `500` raised that share to
approximately `51.5%`.

Before the weighted runs, weight `0` was trained for three epochs. Its
train/validation histories exactly matched the original implementation:

```text
train:      0.06812361699, 0.00895903957, 0.00374280740
validation: 0.01920169001, 0.00461212351, 0.00326709130
```

This verifies that the new adapter preserves the baseline training path when
motion weighting is disabled.

## Predetermined gates

A candidate passes only if all quantitative gates hold:

```text
Oracle changed-pixel MAE < 0.271
Oracle full-frame MSE <= 0.00279
World-model changed-pixel MAE < 0.338870
```

The Oracle preview must also keep the car visually identifiable.

## Results

| Run | Oracle full-frame MSE | Oracle changed MAE | World full-frame MSE | World changed MAE | Normalized latent MSE |
|---|---:|---:|---:|---:|---:|
| Weight 0 baseline | 0.00253248 | 0.338871 | 0.00253248 | 0.338870 | 0.048106 |
| Weight 100 | 0.00694741 | 0.336620 | 0.00694710 | 0.336622 | 0.102152 |
| Weight 500 | 0.03335157 | 0.332476 | 0.03334972 | 0.332478 | 0.138061 |

Relative to baseline:

- Weight `100` improved Oracle changed-pixel MAE by only `0.66%`, while
  worsening Oracle full-frame MSE by `2.74x`.
- Weight `500` improved Oracle changed-pixel MAE by only `1.89%`, while
  worsening Oracle full-frame MSE by `13.17x`.
- Neither candidate reached the changed-pixel gate or the full-frame gate.
- Both weighted runs also increased normalized latent-dynamics MSE.

The previews show that stronger weighting introduces broad scene noise while
only partially recovering the car. Oracle and world-model decoded metrics
remain almost identical within each run, so the decoded result is still
limited by the visual representation rather than one-step dynamics.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_motion100.pt
artifacts/visual_latent_motion100_predictions.png
artifacts/visual_latent_motion500.pt
artifacts/visual_latent_motion500_predictions.png
```

SHA-256:

```text
083dcd0f2ed576934cbe685afa7b8da7cb968f6cc7a43f7fe9876383d6556f65  visual_latent_motion100.pt
6856c581f3069193eda897f38c95010f517d498368c5ef50b42a5b991b2c826d  visual_latent_motion100_predictions.png
21a3576187373df913c76efda3569b0deefcdebd401eb3b78e31919f8de24b2b  visual_latent_motion500.pt
80858428faadf8b2bd339a9df65a37675ea2d05c3f5e61c3f1f9eb67fd97981d  visual_latent_motion500_predictions.png
```

## Decision

Reject both weighted-loss candidates and stop tuning scalar motion weights.
The negative result indicates that a single 32-dimensional global latent
cannot cheaply trade static scene fidelity for precise small-object structure
under this objective.

The next model experiment should preserve spatial structure in the latent,
for example `[C, 8, 8]`, while returning to ordinary reconstruction MSE first.
That experiment should retain the same Oracle/world/copy-last diagnostics and
must pass the same full-frame and changed-pixel gates before multi-step rollout
or MPC work begins.
