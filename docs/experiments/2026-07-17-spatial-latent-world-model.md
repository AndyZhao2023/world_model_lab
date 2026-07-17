# Spatial latent world-model experiment

## Question

Does preserving an `8 x 8` latent grid improve reconstruction of the small
moving car and one-step visual prediction relative to the 32-value global
latent baseline?

## Controlled change

The dataset, complete-episode split, training seed, training duration, pixel
MSE objective, and held-out diagnostics were unchanged. The model pair changed
from:

```text
ConvAutoencoder:       image -> [B, 32]
LatentDynamicsMLP:     4 global latents + 4 actions -> next global latent
```

to:

```text
SpatialConvAutoencoder:    image -> [B, 8, 8, 8]
SpatialLatentDynamicsCNN:  4 latent grids + 4 broadcast actions
                           -> residual next latent grid
```

The spatial encoder and decoder contain no fully connected projection. The
grid is reversibly flattened to 512 values only at the compact array,
normalization, and checkpoint boundaries; the dynamics model and decoder
restore the exact channel-row-column layout before convolution.

This is not a pure topology ablation: the global baseline has 32 latent
scalars and this spatial candidate has 512. A successful result supports the
new candidate, but does not by itself separate spatial structure from latent
capacity.

## Protocol

- Dataset: `data/visual_episodes.npz`
- Dataset SHA-256:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- Train/validation/test split seed: `42`
- Training seed: `0`
- Autoencoder: `base_channels=16`, `latent_channels=8`, `20` epochs
- Spatial dynamics: hidden channels `64`, `50` epochs
- Reconstruction loss: ordinary pixel MSE (`motion_loss_weight=0`)
- Train/validation/test windows: `6330 / 803 / 920`
- Device: CPU
- Approximate wall time: `157` seconds

## Predetermined gates

```text
Oracle changed-pixel MAE < 0.271
Oracle full-frame MSE <= 0.00279
World-model changed-pixel MAE < 0.338870
car remains visible in the Oracle preview
```

## Results

| Metric | Global 32 baseline | Spatial 8 x 8 x 8 | Relative change |
|---|---:|---:|---:|
| Oracle full-frame MSE | 0.00253248 | 0.00151472 | -40.19% |
| Oracle changed-pixel MAE | 0.338871 | 0.266102 | -21.47% |
| World full-frame MSE | 0.00253248 | 0.00158799 | -37.30% |
| World changed-pixel MAE | 0.338870 | 0.314072 | -7.32% |
| Copy-last full-frame MSE | 0.00086896 | 0.00086896 | unchanged |
| Copy-last changed-pixel MAE | 0.601716 | 0.601716 | unchanged |

All three quantitative gates passed. The Oracle preview now contains a
localized foreground object at the car position instead of omitting it. The
car body and heading marker remain blurred, so the representation has improved
substantially but is not yet a sharp reconstruction.

The autoencoder's best validation epoch was `20`, with test-frame MSE
`0.00151079`. The spatial dynamics best validation epoch was `13`, with
normalized latent test MSE `0.030859`.

## Interpretation

The spatial candidate is the first visual model in this sequence to improve
both static-scene fidelity and the moving-object metric under ordinary MSE.
Unlike the global baseline, the world prediction is now measurably worse than
the Oracle reconstruction:

```text
world - oracle full-frame MSE       = 0.00007327
world - oracle changed-pixel MAE    = 0.047970
```

The bottleneck has therefore shifted. Visual compression is no longer the
only visible limit; latent dynamics now accounts for part of the changed-region
error. Multi-step rollout is still premature because one-step car geometry is
blurred and the world/Oracle changed-pixel gap remains meaningful.

## Decision

Retain the spatial latent path as the leading candidate and stop tuning the
global-32 reconstruction loss. Before attributing the gain specifically to
spatial topology, run a global-512 capacity control under the same protocol.
If spatial still wins that control, the next modeling work should target the
spatial dynamics gap rather than adding more autoencoder weighting.

The capacity control has now been completed. Global-512 repeated the global-32
failure and produced Oracle changed-pixel MAE `0.339062`; spatial-512 remained
substantially better at `0.266102`. See
`docs/experiments/2026-07-17-global512-capacity-control.md` for the full
comparison and parameter counts.

Frozen-checkpoint dynamics diagnostics also show that the spatial CNN reduces
changed-pixel MAE from `0.359824` for decoded-last-latent to `0.314072`, closing
approximately `48.8%` of the available gap to Oracle. Action shuffling affects
normalized latent MSE more than decoded pixels; see
`docs/experiments/2026-07-17-spatial-dynamics-diagnostics.md`.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_spatial8.pt
artifacts/visual_latent_spatial8_predictions.png
```

SHA-256:

```text
1b6d1b2ace6d9da013fbc18dff82de0fa7afb6480f6e7a538aacbacbe17c8792  visual_latent_spatial8.pt
72e90004265758d0b5ae5763617e9cadf970f81664ace4a0ebdeb7e44e85082e  visual_latent_spatial8_predictions.png
```
