# Spatial CNN versus ConvGRU dynamics

## Question

Does explicit recurrent temporal state improve one-step spatial latent
dynamics over concatenating the same four latent grids into a CNN?

## Controlled change

The successful spatial autoencoder was frozen and reused from
`artifacts/visual_latent_spatial8.pt`. The comparison kept the following
identical:

- dataset and dataset SHA-256;
- train/validation/test episode IDs;
- encoded latent frames;
- latent and action normalizers;
- four-frame context and aligned actions;
- one-step normalized latent MSE objective;
- optimizer, seed, batch size, learning rate, and 50-epoch budget;
- decoder and held-out evaluation protocol.

Only latent dynamics changed:

```text
CNN:
    concatenate 4 x (latent grid + action map) along channels
    -> three spatial convolutions
    -> residual next latent

ConvGRU:
    for each of 4 ordered (latent grid + action map) inputs:
        update one recurrent hidden grid
    hidden grid -> output convolution -> residual next latent
```

Both return a flattened `[B, C*8*8]` prediction and add their learned residual
to the final context latent. ConvGRU used 40 hidden channels so its parameter
count remained close to, and slightly below, the CNN.

## Protocol

- Dataset: `data/visual_episodes.npz`
- Dataset SHA-256:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- Frozen autoencoder checkpoint:
  `artifacts/visual_latent_spatial8.pt`
- Source checkpoint SHA-256:
  `1b6d1b2ace6d9da013fbc18dff82de0fa7afb6480f6e7a538aacbacbe17c8792`
- Spatial latent: `[B, 8, 8, 8]`
- Train/validation/test windows: `6330 / 803 / 920`
- Training seed: `0`
- Dynamics epochs: `50`
- Batch size: `256`
- Learning rate: `1e-3`
- Device: CPU
- ConvGRU wall time including encoding, evaluation, and saving: `188.7 s`

| Dynamics | Hidden channels | Parameters | Best validation epoch | Best validation MSE |
|---|---:|---:|---:|---:|
| Stacked-frame CNN | 64 | 64,648 | 13 | 0.03937 |
| ConvGRU | 40 | 57,008 | 35 | 0.04207 |

## Primary results

All values were recomputed after loading both checkpoints and using the same
920 held-out windows.

| Method | Normalized latent MSE | Full-frame MSE | Changed-pixel MAE |
|---|---:|---:|---:|
| Stacked-frame CNN | **0.0308590** | 0.00158799 | 0.314072 |
| ConvGRU | 0.0327334 | **0.00158724** | **0.313924** |
| Decode last latent, no dynamics | n/a | 0.00163793 | 0.359824 |
| Oracle target reconstruction | n/a | 0.00151472 | 0.266102 |

ConvGRU relative to CNN:

- normalized latent MSE worsened by `6.07%`;
- full-frame MSE improved by `0.047%`;
- changed-pixel MAE improved by `0.047%`.

The two decoded improvements are too small to treat as a material architecture
gain from this single-seed experiment. ConvGRU trades worse latent prediction
for effectively unchanged decoded pixels.

## Action and history diagnostics

The table reports relative degradation from each model's own recorded-input
result. Negative values mean the ablation happened to improve the metric.
Action shuffling uses the same fixed seed for both models.

| Model and ablation | Latent MSE | Full-frame MSE | Changed-pixel MAE |
|---|---:|---:|---:|
| CNN: mean action | +0.40% | +0.013% | -0.58% |
| ConvGRU: mean action | +0.23% | +0.004% | +0.10% |
| CNN: shuffled action | +3.07% | +0.126% | +0.27% |
| ConvGRU: shuffled action | +2.40% | +0.092% | +0.25% |
| CNN: repeat last context | +62.62% | +3.62% | +13.82% |
| ConvGRU: repeat last context | +54.59% | +3.60% | +13.18% |
| CNN: reverse first three latents | +74.76% | +3.99% | +13.38% |
| ConvGRU: reverse first three latents | +64.45% | +3.76% | +9.85% |

Both models use ordered visual history: repeating or reversing history causes
large latent and changed-region degradation. Both also use action/vision
alignment: shuffling actions increases latent MSE by more than 2%. The small
pixel deltas show that this one-step decoded metric remains weakly sensitive
to actions; they do not show that actions are unused.

## Interpretation

Explicit recurrent state did not close the remaining World-to-Oracle gap. A
four-frame stacked CNN already has access to the complete short context and
can learn temporal filters directly across concatenated frame channels. For
this fixed four-frame, one-step task, recurrence adds an inductive bias but no
material decoded advantage.

The result does not establish that ConvGRU is generally inferior. Recurrence
may become useful with variable or longer context, multi-step latent rollout,
or online hidden-state reuse. None of those capabilities are exercised here.

The mismatch between latent MSE and changed-pixel MAE remains important:
selecting solely on normalized latent MSE favors CNN, while decoded car quality
is effectively tied. Future dynamics work should use an image-space or
object-aware auxiliary objective rather than relying only on architecture
replacement.

## Decision

Keep the simpler stacked-frame CNN as the current default. Do not spend another
single-seed tuning round on ConvGRU for the same four-frame one-step objective.
The next controlled experiment should target objective alignment: train
dynamics with normalized latent MSE plus a frozen-decoder changed-region loss,
then check whether the World-to-Oracle changed-pixel gap closes without
damaging latent stability.

## Artifacts

- ConvGRU checkpoint:
  `artifacts/visual_latent_spatial8_convgru.pt`
- Checkpoint SHA-256:
  `39c4b588428e9a1b420be116643c98bce7b285a6c23fbed6de774da243875c57`
- Prediction preview:
  `artifacts/visual_latent_spatial8_convgru_predictions.png`
- Preview SHA-256:
  `a6a81c6db156284927902ed14dc70c6eeac4e4ddfe10744aa3b3fcd9b8e4883b`

The saved checkpoint records `autoencoder_frozen=true`, the source checkpoint,
the concrete `convgru` architecture, training history, and complete held-out
metrics. Reloaded evaluation reproduced every stored metric exactly.
