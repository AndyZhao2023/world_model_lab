# Spatial dynamics objective alignment

## Question

Can a frozen-decoder changed-pixel auxiliary loss reduce the spatial CNN's
remaining World-to-Oracle car-pixel gap without materially damaging normalized
latent prediction?

## Controlled change

The successful spatial checkpoint is reused as the source:

```text
artifacts/visual_latent_spatial8.pt
```

The following remain fixed:

- visual dataset and dataset SHA-256;
- train, validation, and test episode IDs;
- spatial autoencoder weights and `[B, 8, 8, 8]` latent layout;
- train-only latent and action normalizers;
- four ordered context frames and four aligned actions;
- stacked-frame spatial CNN architecture with 64 hidden channels;
- optimizer, seed, learning rate, batch size, and 50-epoch budget;
- held-out evaluation metrics and action/history ablations.

Only the dynamics training objective changes from:

```text
normalized latent MSE
```

to:

```text
normalized latent MSE + 0.1 * decoded changed-pixel MAE
```

For the auxiliary term, the predicted normalized latent is denormalized and
decoded through the frozen source decoder. The binary supervision mask marks
pixels whose RGB value differs between the last context frame and true target
frame. The mask is used only to calculate training and validation loss; it is
not an inference input and does not use physical state or object labels.

The decoder participates in autograd so image-space gradients reach the
predicted latent, but all decoder parameters have `requires_grad=False` and
are absent from the optimizer.

## Predetermined protocol and gates

- Dataset: `data/visual_episodes.npz`
- Source checkpoint: `artifacts/visual_latent_spatial8.pt`
- Dynamics: stacked-frame spatial CNN, 64 hidden channels
- Training seed: `0`
- Dynamics epochs: `50`
- Batch size: `256`
- Learning rate: `1e-3`
- Auxiliary weight: `0.1`

Baseline:

```text
normalized latent MSE = 0.0308590
changed-pixel MAE     = 0.314072
Oracle changed MAE    = 0.266102
```

The candidate passes only if:

```text
primary:
    changed-pixel MAE < 0.314072

latent stability:
    normalized latent MSE <= 0.0339449
    # no more than 10% worse than the latent-only CNN

representation invariance:
    source and candidate autoencoder weights are exactly equal
    Oracle metrics are unchanged
```

These gates were recorded before the candidate run. If changed-pixel MAE does
not improve, weight `0.1` will be rejected before any additional weight is
tried. If changed-pixel MAE improves but latent stability fails, the candidate
will not replace the current baseline.

## Reproduction command

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_dynamics_objective \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8.pt \
  --output artifacts/visual_latent_spatial8_objective_w01.pt \
  --preview artifacts/visual_latent_spatial8_objective_w01_predictions.png \
  --changed-pixel-loss-weight 0.1 \
  --dynamics-epochs 50 \
  --dynamics-batch-size 256
```

## Results

The candidate trained successfully on CPU in `101.15 s`, including source
loading, frame encoding, 50 dynamics epochs, held-out evaluation, checkpoint
publication, and preview generation. Its best combined validation objective
occurred at epoch `20`.

### Primary metrics

| Method | Normalized latent MSE | Full-frame MSE | Changed-pixel MAE |
|---|---:|---:|---:|
| Latent-only spatial CNN | **0.03085897** | **0.00158799** | 0.31407161 |
| Aligned objective, weight 0.1 | 0.03136505 | 0.00158927 | **0.29990801** |
| Decode last latent, no dynamics | n/a | 0.00163793 | 0.35982404 |
| Oracle target reconstruction | n/a | 0.00151472 | **0.26610151** |
| Raw RGB copy-last | n/a | **0.00086896** | 0.60171581 |

Relative to the latent-only CNN, the aligned objective:

- improved changed-pixel MAE by `4.51%`;
- worsened normalized latent MSE by `1.64%`;
- worsened full-frame MSE by `0.081%`.

Both predetermined gates passed. The latent degradation remained well below
the allowed `10%`, while the primary changed-region metric improved. The
candidate reduced the remaining baseline World-to-Oracle changed-pixel gap by
`29.53%`. It closes `63.93%` of the complete decoded-last-to-Oracle gap,
compared with `48.8%` for the latent-only CNN.

### Representation invariance

The candidate was reloaded independently before verification:

- every autoencoder state-dict tensor was exactly equal to the source;
- train, validation, and test episode IDs were exactly equal;
- latent and action normalizer means and standard deviations were exactly
  equal;
- Oracle metrics were unchanged;
- every stored candidate held-out metric was reproduced exactly.

The improvement therefore crosses only the dynamics training-objective seam.
It is not explained by retraining the representation or changing the split.

### Action and history diagnostics

| Input variant | Normalized latent MSE | Changed-pixel MAE | Relative changed-MAE delta |
|---|---:|---:|---:|
| Recorded ordered inputs | **0.03136505** | 0.29990801 | baseline |
| Mean actions | 0.03214180 | **0.29807562** | `-0.61%` |
| Shuffled actions, 10-seed mean | 0.03283302 | 0.30133261 | `+0.48%` |
| Repeat last visual context | 0.05013434 | 0.35624279 | `+18.78%` |
| Reverse first three latents | 0.06209820 | 0.36070723 | `+20.27%` |

The ten action shuffles produced changed-pixel MAE:

```text
0.30133261 ± 0.00022282  # sample standard deviation
```

Relative to recorded actions, the ten-shuffle mean worsened normalized latent
MSE by `4.68%`, full-frame MSE by `0.21%`, and changed-pixel MAE by `0.48%`.
This is a stronger action-alignment signal than the latent-only CNN, but it is
still small after decoding. Mean action remains `0.61%` better on changed
pixels despite being `2.48%` worse in latent space. The aligned objective
therefore improves motion localization but does not establish strong
sample-specific action use.

Ordered visual history remains essential. Repeating the last latent worsens
changed-pixel MAE by `18.78%`; reversing the older history worsens it by
`20.27%`. The candidate has not replaced temporal motion inference with an
image-space shortcut.

### Visual check

The candidate preview keeps the predicted car at approximately the target
location in all six fixed rows. The car body and heading marker remain visibly
blurred, and the improvement over the latent-only preview is not obvious by
eye at this sample count. Error remains concentrated around the moving car,
which is consistent with the quantitative changed-region improvement rather
than a broad background change.

## Decision

Retain weight `0.1` as the leading one-step spatial dynamics objective. Do not
run an immediate weight sweep: the preregistered candidate passed both the
primary and stability gates.

This result addresses objective alignment but not action causality. Before
MPC, the next controlled experiment should compare latent-only and aligned
models under multi-step free rollout with recorded and counterfactual action
sequences. The purpose is to determine whether the stronger latent
action-alignment signal grows into a meaningful trajectory difference over
horizons `1`, `5`, and `10`, or remains negligible after decoding.

## Artifacts

Ignored local outputs:

```text
artifacts/visual_latent_spatial8_objective_w01.pt
artifacts/visual_latent_spatial8_objective_w01_predictions.png
```

SHA-256:

```text
5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369  visual_latent_spatial8_objective_w01.pt
f01d48042662d48dd71480fa3f0b30daaa0293afd72576d3865d9199b027ea01  visual_latent_spatial8_objective_w01_predictions.png
```

Source:

```text
1b6d1b2ace6d9da013fbc18dff82de0fa7afb6480f6e7a538aacbacbe17c8792  visual_latent_spatial8.pt
2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba  visual_episodes.npz
```
