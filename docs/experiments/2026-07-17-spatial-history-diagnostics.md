# Spatial visual-history diagnostics

## Question

Does the frozen spatial dynamics CNN use ordered visual motion history, or can
it achieve the same one-step prediction from only the current latent frame?

## Context variants

All variants keep the trained model, target latent, recorded actions, test
windows, normalizers, and final context latent fixed.

```text
Recorded:
[z[t-3], z[t-2], z[t-1], z[t]]

Repeat last:
[z[t],   z[t],   z[t],   z[t]]

Reverse history:
[z[t-1], z[t-2], z[t-3], z[t]]
```

`repeat last` removes observed visual motion while preserving the current
latent anchor. `reverse history` retains the three older latent frames but
destroys their temporal order; the current `z[t]` remains in the final slot
because the model predicts a residual from that slot.

Recorded actions are intentionally unchanged, so only visual context arrays
are modified. This also means reverse-history introduces inconsistency between
the reordered frames and recorded action sequence; degradation cannot be
attributed exclusively to frame order.

## Protocol

- Dataset: `data/visual_episodes.npz`
- Dataset SHA-256:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- Frozen checkpoint: `artifacts/visual_latent_spatial8.pt`
- Checkpoint SHA-256:
  `1b6d1b2ace6d9da013fbc18dff82de0fa7afb6480f6e7a538aacbacbe17c8792`
- Test windows: `920`
- Batch size: `256`
- No retraining or checkpoint mutation

## Results

| Context | Normalized latent MSE | Full-frame MSE | Changed-pixel MAE |
|---|---:|---:|---:|
| Recorded ordered history | **0.0308590** | **0.00158799** | **0.314072** |
| Repeat last latent | 0.0501814 | 0.00164555 | 0.357479 |
| Reverse first three latents | 0.0539280 | 0.00165142 | 0.356096 |
| Decode last latent, no dynamics | n/a | 0.00163793 | 0.359824 |
| Oracle target reconstruction | n/a | 0.00151472 | 0.266102 |

Relative to recorded ordered history:

- repeat-last worsens latent MSE by `62.6%`, full-frame MSE by `3.62%`, and
  changed-pixel MAE by `13.82%`;
- reverse-history worsens latent MSE by `74.8%`, full-frame MSE by `3.99%`,
  and changed-pixel MAE by `13.38%`.

Using decoded-last and Oracle as the no-dynamics and representation-limited
endpoints:

| Context | Fraction of decoded-last-to-Oracle changed-pixel gap closed |
|---|---:|
| Recorded ordered history | **48.8%** |
| Repeat last latent | 2.5% |
| Reverse first three latents | 4.0% |

## Interpretation

The useful one-step dynamics gain depends on ordered visual history. Removing
motion history or supplying it in the wrong order eliminates almost all of the
improvement over decoded-last-latent. The model is therefore not predicting
mainly from the current image, mean motion, or a fixed residual bias.

Reverse-history and repeat-last have nearly the same changed-pixel error.
Simply retaining older frames is insufficient; their temporal relationship is
important. Because recorded actions remain aligned to the original order, the
reverse result includes a cross-modal mismatch. Previous action diagnostics
showed only a small one-step pixel effect, which makes visual order the leading
explanation, but not an isolated causal proof.

The much larger latent-MSE degradation than full-frame-MSE degradation again
shows that whole-image pixels hide motion errors behind static background.
Changed-pixel MAE remains the more decision-relevant visual metric.

## Decision

Keep four-frame ordered visual context. Do not simplify the model to a
single-frame latent predictor. The next controlled architecture experiment
should freeze the successful spatial autoencoder and compare the current
channel-concatenation CNN against a small ConvGRU dynamics model under the same
one-step protocol. This tests whether explicit recurrent temporal state can
close more of the remaining world-to-Oracle gap.

That comparison is now complete. ConvGRU changed-pixel MAE was `0.313924`
versus CNN `0.314072`, an immaterial `0.047%` improvement, while normalized
latent MSE worsened by `6.07%`. Both architectures remained strongly dependent
on ordered visual history. The result supports retaining the simpler CNN for
this fixed four-frame one-step task; see
`docs/experiments/2026-07-17-spatial-convgru-dynamics.md`.
