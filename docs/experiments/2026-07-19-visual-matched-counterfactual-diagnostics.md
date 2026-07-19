# Visual matched counterfactual diagnostics

## Status

Completed. The candidate passed four of five pre-registered gates, so the
strict decision remains negative and the H1 source remains the default.

## Question

When a complete future action row is replaced after one shared held-out visual
history, does H5 recursive training improve the accuracy of the resulting
trajectory against the true `CarEnv` branch, rather than merely increasing
the model's sensitivity to action changes?

## Fixed inputs

```text
2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba  data/visual_episodes.npz
5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369  artifacts/visual_latent_spatial8_objective_w01.pt
4764d89d7399694574ad91a6703c90872d9c0f7c1b6e7743572dc58d3ff0e424  artifacts/visual_latent_spatial8_objective_w01_h5.pt
```

The source and candidate must have exactly equal:

- spatial autoencoder tensors;
- latent and action normalizers;
- test episode IDs;
- dataset SHA-256;
- spatial latent layout and CNN architecture.

Neither checkpoint is changed or trained in this experiment.

## Matched branch protocol

The window population remains the existing H10-capable held-out population:

- horizons `1`, `5`, and `10`;
- at most eight evenly spaced windows per test episode;
- 18 eligible episodes and 136 windows under the existing dataset;
- counterfactual seeds `0..9`.

For each seed:

1. Build the existing deterministic Sattolo permutation of all selected
   windows. It is a complete single-cycle permutation with no fixed point.
2. Give each recipient window the donor window's complete current-plus-future
   action row.
3. Recreate `CarEnv` from the recipient's exact aligned state at its final
   context frame.
4. Execute the donated actions in the simulator and render every returned
   state with the schema-v1 default scene.
5. Feed the simulator's actual clipped actions to both world models.

The transition that causes goal, collision, out-of-bounds, or time-limit is a
valid target. Every later step is invalid and masked. Terminal branches are
reported rather than silently removed. Recreating an anchor starts a fresh
environment step counter; the selected windows are H10-capable, and that
counter reset is recorded as a protocol limitation.

## Metrics

For every valid branch step:

```text
matched normalized latent MSE
matched pixel MSE
matched transition changed-pixel MAE
matched cumulative changed-pixel MAE
normalized latent action-effect MSE
pixel action-effect MSE
```

The action-effect target is:

```text
true counterfactual future - true factual future
```

and the model action effect is:

```text
predicted counterfactual future - predicted factual future
```

True counterfactual frames are encoded with the shared frozen autoencoder.
The same encoded latents are also decoded to report the oracle reconstruction
pixel metrics, which expose the representation/decoder floor separately from
dynamics error.

For each seed and step, valid windows are aggregated inside their recipient
episode, then episodes are weighted equally. Seed-level episode-macro values
are summarized by mean and sample standard deviation. Source and candidate
always use the same branches and masks.

## Pre-registered decision gates

The H5 candidate passes only if all three H5 primary gates pass:

```text
H5 candidate matched normalized latent MSE
  < H5 source matched normalized latent MSE

H5 candidate matched cumulative changed-pixel MAE
  < H5 source matched cumulative changed-pixel MAE

H5 candidate normalized latent action-effect MSE
  < H5 source normalized latent action-effect MSE
```

and both H1 stability gates pass:

```text
H1 candidate matched normalized latent MSE
  <= 1.10 * H1 source matched normalized latent MSE

H1 candidate matched cumulative changed-pixel MAE
  <= 1.05 * H1 source matched cumulative changed-pixel MAE
```

H10 is a secondary generalization result. Strict equality does not pass an
improvement gate. If any primary or stability gate fails, the current H1
source remains preferred. Passing all gates permits only promotion to the
next offline planning experiment; it does not authorize MPC or environment
control.

No seed, horizon, action donor, metric, or gate sweep will be started inside
this run.

## Reproduction command

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_counterfactual \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --candidate-checkpoint \
    artifacts/visual_latent_spatial8_objective_w01_h5.pt \
  --output-dir \
    artifacts/diagnostics/visual-matched-counterfactual-h5-comparison \
  --horizons 1 5 10 \
  --windows-per-episode 8 \
  --counterfactual-seeds 0 1 2 3 4 5 6 7 8 9 \
  --decision-horizon 5
```

## Results

The diagnostic retained the expected 18 H10-capable test episodes and 136
recipient windows. Ten Sattolo seeds produced 1,360 matched simulator
branches. No donated action required clipping.

Matched counterfactual model results:

| Horizon | Metric | H1 source | H5 candidate | Relative change |
|---:|---|---:|---:|---:|
| 1 | normalized latent MSE | 0.0369668000 | 0.0337177107 | -8.79% |
| 1 | cumulative changed-pixel MAE | 0.3065254702 | 0.3085565448 | +0.663% |
| 1 | normalized latent action-effect MSE | 0.0013556197 | 0.0013369048 | -1.38% |
| 1 | pixel MSE | 0.0015334148 | 0.0015233096 | -0.659% |
| 1 | pixel action-effect MSE | 0.0001846632 | 0.0001842885 | -0.203% |
| 5 | normalized latent MSE | 0.3065971604 | 0.2225983635 | -27.40% |
| 5 | cumulative changed-pixel MAE | 0.2764975452 | 0.2767559394 | +0.093% |
| 5 | normalized latent action-effect MSE | 0.1209001768 | 0.1020056328 | -15.63% |
| 5 | pixel MSE | 0.0022084709 | 0.0019769668 | -10.48% |
| 5 | pixel action-effect MSE | 0.0016152114 | 0.0015526775 | -3.87% |
| 10 | normalized latent MSE | 0.7903406148 | 0.5830562159 | -26.23% |
| 10 | cumulative changed-pixel MAE | 0.3044660457 | 0.3009542084 | -1.15% |
| 10 | normalized latent action-effect MSE | 0.7099581705 | 0.5779368527 | -18.60% |
| 10 | pixel MSE | 0.0029753581 | 0.0025636647 | -13.84% |
| 10 | pixel action-effect MSE | 0.0032122288 | 0.0029681283 | -7.60% |

The H5 transition changed-pixel MAE also improved `3.66%`, from
`0.3467591122` to `0.3340672414`. The H10 transition metric worsened `0.183%`,
so the long-horizon cumulative improvement should not be read as uniform
per-transition improvement.

Pre-registered gates:

| Gate | Source/limit | Candidate | Result |
|---|---:|---:|---|
| H5 matched latent MSE strictly improves | 0.3065971604 | 0.2225983635 | PASS |
| H5 cumulative changed-pixel MAE strictly improves | 0.2764975452 | 0.2767559394 | **FAIL** |
| H5 latent action-effect MSE strictly improves | 0.1209001768 | 0.1020056328 | PASS |
| H1 matched latent MSE <= 110% source | 0.0406634801 | 0.0337177107 | PASS |
| H1 cumulative changed-pixel MAE <= 105% source | 0.3218517437 | 0.3085565448 | PASS |

The shared oracle reconstruction cumulative changed-pixel MAE is
`0.2598294834`, `0.2111001221`, and `0.1926178559` at H1, H5, and H10. The
large candidate-to-oracle gap shows that neither lower latent drift nor the
frozen representation alone solves precise moving-object pixels.

Coverage:

| Horizon | Valid branches | Possible branches | Coverage |
|---:|---:|---:|---:|
| 1 | 1,360 | 1,360 | 100.00% |
| 5 | 1,360 | 1,360 | 100.00% |
| 10 | 1,216 | 1,360 | 89.41% |

Across the complete H10 simulations, 49 branches ended in collision, five at
the goal, and 143 out of bounds. The transition that caused termination was
retained; only later steps were masked. Source and candidate used the same
masks.

Independent execution produced byte-identical manifest, metrics, and plot
files. Published artifact digests:

```text
47e907e99ff09649ba685f1d48b1e871e6224925ea3ef1dc40245b02bd771a95  artifacts/diagnostics/visual-matched-counterfactual-h5-comparison/manifest.json
ee956a3f9e29f7f2cc33c89152312b2f7c2dd8faa37f19ab2dd6c0781dd9a7ef  artifacts/diagnostics/visual-matched-counterfactual-h5-comparison/metrics.json
4c879e98f870249117a8007dda01833270c5748c20b87c3a4d37ea40cdd31f00  artifacts/diagnostics/visual-matched-counterfactual-h5-comparison/matched_counterfactual_comparison.png
```

## Decision

Do not promote the H5 candidate under the pre-registered rule. It now has
matched simulator evidence—not sensitivity alone—that recursive training
improves H5 latent trajectory accuracy, latent action-effect accuracy, full
pixel MSE, and transition changed-pixel MAE. However, it misses the required
H5 cumulative changed-pixel improvement by `0.0002583942` absolute
(`0.093%`).

This is a much narrower failure than the previous unmatched diagnostic, but
changing the gate after seeing the result would invalidate the controlled
decision. The H1 source therefore remains the default, and neither model is
connected to MPC.

The next experiment should target object motion directly—such as a
renderer-derived car-centre/heading auxiliary objective or an object mask
trajectory loss—while preserving the matched simulator diagnostic as the
held-out decision protocol. A rollout-weight or seed sweep is not justified
before that mismatch is addressed.
