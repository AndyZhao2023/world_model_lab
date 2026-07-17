# Global-512 latent capacity control

## Question

Did the spatial `8 x 8 x 8` model improve mainly because it contains 512
latent scalars instead of the global baseline's 32, or does preserving a
spatial grid provide a useful inductive bias?

## Controlled change

This run keeps the original global architecture:

```text
four stride-2 convolutions -> flatten -> Linear -> global latent vector
four global latent vectors + actions -> residual MLP -> next global latent
```

Only `latent_dim` changes from `32` to `512`. The dataset, split, seed,
ordinary pixel MSE, base channels, autoencoder epochs, dynamics epochs, batch
sizes, and learning rates remain fixed. The global dynamics MLP keeps its
baseline hidden size of `256`.

The comparison to spatial-512 is most direct for Oracle reconstruction because
Oracle bypasses both dynamics models. World-model metrics additionally compare
the global MLP with the spatial CNN.

## Protocol

- Dataset SHA-256:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- Split seed: `42`
- Training seed: `0`
- Autoencoder: global latent `512`, base channels `16`, `20` epochs
- Dynamics: MLP hidden size `256`, `50` epochs
- Reconstruction loss: ordinary pixel MSE
- Train/validation/test windows: `6330 / 803 / 920`
- Device: CPU
- Approximate wall time: `150` seconds

## Results

| Metric | Global-32 | Global-512 | Spatial-512 |
|---|---:|---:|---:|
| Oracle full-frame MSE | 0.00253248 | 0.00255494 | **0.00151472** |
| Oracle changed-pixel MAE | 0.338871 | 0.339062 | **0.266102** |
| World full-frame MSE | 0.00253248 | 0.00255494 | **0.00158799** |
| World changed-pixel MAE | 0.338870 | 0.339060 | **0.314072** |
| Copy-last full-frame MSE | 0.00086896 | 0.00086896 | 0.00086896 |
| Copy-last changed-pixel MAE | 0.601716 | 0.601716 | 0.601716 |

Increasing the global latent from 32 to 512 produced no meaningful
improvement:

- Oracle full-frame MSE worsened by approximately `0.89%`.
- Oracle changed-pixel MAE worsened by approximately `0.06%`.
- The Oracle and world-model outputs remained nearly identical, so visual
  compression was still the dominant bottleneck.

At equal latent scalar count, spatial-512 reduced Oracle changed-pixel MAE by
approximately `21.5%` and Oracle full-frame MSE by approximately `40.7%`
relative to global-512.

## Parameter-count check

| Model | Autoencoder parameters | Dynamics parameters |
|---|---:|---:|
| Global-32 | 281,411 | 109,088 |
| Global-512 | 1,264,931 | 723,968 |
| Spatial-512 | 26,219 | 64,648 |

Global-512 had far more trainable parameters than spatial-512 but still
omitted the car. The result is therefore not explained by a lack of raw global
capacity.

## Visual check

The global-512 Oracle and predicted columns reconstruct the static scene but
omit the car. The missing car remains clearly visible only in the error
columns. This matches the changed-pixel metric and repeats the global-32
failure mode.

## Interpretation

The evidence supports the spatial model's architectural inductive bias: local
latent cells and convolutional decoding make it easier to preserve a small
localized object than a global fully connected bottleneck. Simply allocating
more numbers to the global vector does not force the model to use them for the
car under whole-image MSE.

This experiment still changes more than tensor notation. Spatial-512 retains
an `8 x 8` encoder feature map and uses a shallower convolutional decoder,
whereas the global architecture reaches `4 x 4` and crosses learned Linear
projections. The result establishes that the tested spatial architecture is
better, not that grid shape is the only causal difference.

## Decision

Reject global-512 and retain spatial-512 as the leading visual model. Do not
spend another run increasing the global bottleneck. The next useful work is to
reduce the spatial world/Oracle changed-pixel gap (`0.047970`) by improving or
diagnosing spatial dynamics, while leaving the successful spatial
autoencoder fixed.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_global512.pt
artifacts/visual_latent_global512_predictions.png
```

SHA-256:

```text
f16134ae3a661c2ab46cb0050811ebc81692145bbe2ebc52b907a4a7ac119de4  visual_latent_global512.pt
7a04ef7b759c324609405876b57667f2f683532166ef305cfca852da7e2bf3a5  visual_latent_global512_predictions.png
```
