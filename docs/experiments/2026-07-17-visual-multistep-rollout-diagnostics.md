# Visual multi-step rollout diagnostics

## Status

Completed after protocol pre-registration. Neither checkpoint was retrained or
mutated.

## Questions fixed before the run

1. Does objective alignment reduce free-rollout cumulative changed-pixel MAE
   at horizons 1, 5, and 10?
2. How quickly does free rollout separate from teacher forcing?
3. Does replacing only the current and future action sequence produce
   divergence that grows with horizon?
4. Is the objective-aligned model more action-sensitive than the latent-only
   baseline?

## Protocol

- Dataset: `data/visual_episodes.npz`
- Baseline: `artifacts/visual_latent_spatial8.pt`
- Objective-aligned:
  `artifacts/visual_latent_spatial8_objective_w01.pt`
- Snapshot horizons: `1`, `5`, and `10`
- Window limit: at most `8` evenly spaced starts per eligible test episode
- Initial context: four true latent frames and three recorded history actions
- Free rollout: predicted latents are recursively appended to the context
- Teacher forcing: every step receives the corresponding true four-latent
  rolling context
- Counterfactual seeds: `0` through `9`
- Counterfactual intervention: preserve the visual context and history actions,
  then replace the complete current-plus-future `[10, 2]` action row with one
  from another selected window using a no-fixed-point Sattolo permutation
- Aggregation: average windows inside each episode first, then average eligible
  episodes equally

Both checkpoints must contain exactly the same spatial autoencoder weights,
latent/action normalizers, test split, dataset digest, latent dimensions, and
CNN dynamics architecture. The evaluator rejects the comparison otherwise.

## Metrics

Recorded-action teacher-forced and free rollouts report:

- normalized latent MSE;
- full-frame pixel MSE;
- adjacent-transition changed-pixel MAE;
- cumulative-from-initial-frame changed-pixel MAE.

Changed-pixel masks come only from true RGB frame differences. Physical states,
rewards, dones, and object labels are not used.

Counterfactual rollouts report divergence from the recorded-action free
rollout:

- normalized latent RMS;
- decoded full-frame pixel MSE;
- decoded full-frame pixel MAE.

Counterfactual divergence measures action sensitivity, not counterfactual
accuracy, because no ground-truth frame exists for an action sequence that was
not executed from that exact visual context.

## Results

The run used `18` eligible test episodes and `136` rollout windows. Seven test
episodes were too short to provide three history actions plus ten future
actions and were recorded as skipped.

Source digests:

- dataset:
  `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`
- baseline:
  `1b6d1b2ace6d9da013fbc18dff82de0fa7afb6480f6e7a538aacbacbe17c8792`
- objective-aligned:
  `5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369`

### Recorded-action free rollout

| Horizon | Baseline cumulative changed MAE | Aligned cumulative changed MAE | Relative change |
|---:|---:|---:|---:|
| 1 | 0.318182 | **0.303442** | **-4.63%** |
| 5 | 0.320132 | **0.275984** | **-13.79%** |
| 10 | 0.331149 | **0.299133** | **-9.67%** |

The objective-aligned model therefore improves the pre-registered cumulative
motion-region metric at all three horizons. The strongest gain occurs at
horizon 5.

These horizon-1 values intentionally differ slightly from the earlier one-step
report: this protocol keeps only episodes long enough for horizon 10, samples
at most eight starts per episode, and weights episodes equally.

The horizon-10 result is mixed outside that primary metric. Aligned versus
baseline is nearly tied on normalized latent MSE (`+0.09%`) and adjacent
transition changed-pixel MAE (`+0.59%`), but is worse on full-frame pixel MSE
(`+11.70%`). The improvement is specifically about preserving the cumulative
moving-object region, not a uniform improvement in every visual metric.

### Teacher forcing versus free rollout

| Horizon | Baseline cumulative-MAE gap | Aligned cumulative-MAE gap |
|---:|---:|---:|
| 1 | 0.000000 | 0.000000 |
| 5 | 0.096660 | **0.058291** |
| 10 | 0.127325 | **0.097846** |

Objective alignment reduces the decoded motion-region compounding gap, but
does not remove recursive drift. At horizon 10, teacher-forced normalized
latent MSE remains about `0.051` for both models while free-rollout latent MSE
reaches about `0.856`. The dominant long-horizon failure is therefore feeding
predicted latents back into later predictions, not local one-step capacity.

### Counterfactual action sensitivity

| Horizon | Baseline latent RMS | Aligned latent RMS | Aligned / baseline | Pixel-MSE ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.013209 | 0.020406 | 1.545× | 2.802× |
| 5 | 0.091071 | 0.126127 | 1.385× | 2.126× |
| 10 | 0.199343 | 0.317163 | 1.591× | 2.796× |

Replacing only the current and future actions produces increasing divergence:
horizon-10 latent RMS is `15.09×` horizon 1 for baseline and `15.54×` for
aligned. The aligned checkpoint is also consistently more sensitive to action
replacement. This is direct evidence that action input affects the imagined
trajectory; it contradicts the hypothesis that actions are being ignored.

It still does not prove that the alternative trajectory is correct. Each
counterfactual action sequence was borrowed from another window, so there is
no matched true future frame for that intervention.

### Reproducibility

The complete evaluator was run a second time into a separate directory. The
two `manifest.json`, `metrics.json`, and PNG files were byte-identical.

Generated artifact digests:

- `manifest.json`:
  `410e7112a94c5c8656c715d704b32caf35aad8eb50373a9d2d2a21061bc1495b`
- `metrics.json`:
  `fc8746edb9cb63e3435d913fbe5550f3c4c718191325d1c517f9878e30675bd0`
- `visual_rollout_comparison.png`:
  `48b2ddf76e41a5282dbad374a372dd2e761efac1c522829e0063fe155416437b`

## Decision

Do not connect this visual model to MPC yet. The objective-aligned checkpoint
is the better motion-aware one-step model and actions demonstrably influence
its rollout, but free latent error still grows from about `0.036` at step 1 to
`0.857` at step 10. Action sensitivity without matched counterfactual truth is
insufficient for planning.

The next training experiment should optimize the dynamics through a short
recursive rollout, keeping the frozen decoder and decoded changed-pixel term.
After that, generate matched counterfactual futures from the simulator so
action-conditioned accuracy can be measured before any MPC integration.
