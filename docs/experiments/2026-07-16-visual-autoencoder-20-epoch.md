# Visual autoencoder 20-epoch experiment

## Question

Does the default-capacity autoencoder only need more training, or does the
visual bottleneck still dominate one-step world-model quality?

## Protocol

- Dataset: `data/visual_episodes.npz`
- Dataset SHA-256:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- Episode split seed: `42`
- Training seed: `0`
- Autoencoder: `latent_dim=32`, `base_channels=16`, `20` epochs
- Latent dynamics: hidden size `256`, `50` epochs
- Train/validation/test windows: `6330 / 803 / 920`
- Device: CPU

The comparison uses the same 920 held-out windows:

- `oracle reconstruction`: decode the encoder latent of the true target frame.
- `world model`: decode the latent predicted from four frames and four actions.
- `copy-last`: use the last context frame as the next-frame prediction.

## Result

| Metric | Oracle reconstruction | World model | Copy-last |
|---|---:|---:|---:|
| Full-frame pixel MSE | 0.00253248 | 0.00253248 | 0.00086896 |
| Changed-pixel MAE | 0.338871 | 0.338870 | 0.601716 |

The autoencoder test-frame MSE was `0.00253331`, down from the earlier
two-epoch smoke result of approximately `0.01075`. Its best validation loss
occurred at epoch 20, so the run had not visibly converged early.

## Interpretation

Longer training improved full-frame reconstruction by roughly `4.2x`, but the
Oracle MSE remained approximately `2.9x` worse than copy-last. Oracle and
world-model decoded metrics were nearly identical
(`world - oracle MSE = 1.17e-9`), which shows that decoded one-step quality is
currently capped by the visual representation rather than latent-dynamics
error.

The preview also shows that checkerboard texture is no longer the dominant
failure. Static scene geometry is reconstructed well, while the small car is
blurred or omitted. Replacing `ConvTranspose2d` alone is therefore not the
best-supported next experiment.

## Decision

Do not start multi-step rollout or MPC work yet. The next controlled
experiment should make reconstruction more sensitive to small foreground and
motion-bearing structures while keeping the dynamics architecture fixed. Test
one change at a time, beginning with a reconstruction objective that adds an
edge- or foreground-sensitive term; compare it against this run using the same
Oracle, world-model, and copy-last metrics.
