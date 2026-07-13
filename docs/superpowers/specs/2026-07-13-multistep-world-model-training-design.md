# Multi-Step World Model Training Design

## Goal

Extend the existing deterministic state-delta world model with an optional
multi-step training objective that reduces recursive rollout error while
preserving the current one-step training path as the controlled baseline.

The experiment changes the training objective, not the model architecture,
dataset split, action source, or diagnostic protocol. Its purpose is to test
whether training through predicted states reduces compounding error.

## Scope

This stage adds:

- contiguous sequence-window construction within individual episodes;
- a differentiable PyTorch free rollout over recorded action sequences;
- a hybrid one-step and rollout-state training objective;
- training and validation histories for each loss component;
- backward-compatible checkpoint loading;
- command-line controls for rollout horizon and loss weight;
- documentation and tests for the new training mode;
- a controlled horizon-1 versus horizon-10 experiment using the existing
  diagnostic benchmark.

This stage does not change `WorldModelMLP`, predict rewards or termination,
learn a policy, implement MPC/PPO, introduce latent observations, or add an
external experiment-tracking service.

## Compatibility Contract

The existing `world-model-train` command remains the only training entry
point. It gains two options:

```text
--rollout-horizon 1
--rollout-loss-weight 1.0
```

The CLI default for `rollout_horizon` is `1`, so an unchanged command uses the
existing transition-wise one-step training behavior. A value greater than one
enables hybrid multi-step training. The first comparison experiment explicitly
uses `rollout_horizon=10` and `rollout_loss_weight=1.0`.

`rollout_horizon` must be a positive integer. `rollout_loss_weight` must be a
finite non-negative number. For horizon 1, the effective rollout weight is
zero and no sequence arrays are required, which preserves compatibility with
older transition datasets that do not contain `step_ids`.

## Sequence Data

Multi-step training requires `states`, `actions`, `next_states`,
`episode_ids`, and `step_ids` from the NPZ dataset. Episode splitting happens
before sequence construction, using the existing deterministic episode split.
No sequence may include transitions from two splits or two episodes.

A focused dataset helper builds a `SequenceWindows` value with:

```text
states          [W, H, 4]  true state at each transition
actions         [W, H, 2]  recorded action at each transition
next_states     [W, H, 4]  true next state at each transition
episode_ids     [W]        episode owning each window
start_step_ids  [W]        first step of each window
```

For every selected episode, transitions are ordered by `step_ids`. The helper
requires step IDs to be contiguous from zero and requires
`states[1:] == next_states[:-1]` within numerical tolerance. Every possible
length-`H` window is included deterministically. Episodes shorter than `H`
produce no rollout windows.

Short episodes are not discarded from the overall training population. The
one-step term continues to sample every transition in the training split;
only the rollout term samples sequence windows. This avoids changing the
baseline one-step supervision merely because an episode is shorter than the
rollout horizon.

The validation split must contain at least one eligible sequence window when
`H > 1`. Otherwise training fails with a clear message that includes the
requested horizon.

## Model Inputs During Rollout

The recursive training loop operates on physical states so the update remains
interpretable:

\[
\hat{s}_{t+k+1}
=
\hat{s}_{t+k}
+
\widehat{\Delta s}_{t+k}.
\]

At every step, the current predicted state and recorded action are encoded as

\[
[x,y,\sin\theta,\cos\theta,v,\text{steering},\text{acceleration}],
\]

then normalized with statistics fitted only on training transitions. The MLP
returns a normalized delta, which is converted back to physical units with the
training target normalizer before it is added to the current state.

Heading is normalized after each update with the differentiable identity

\[
\operatorname{wrap}(\theta)
=
\operatorname{atan2}(\sin\theta,\cos\theta).
\]

The training rollout must not call the NumPy/no-gradient evaluation helper.
All encoding, normalization, denormalization, state updates, and heading
wrapping remain in PyTorch so gradients flow through the full horizon.

## Hybrid Training Objective

### One-Step Term

The one-step term preserves the current normalized delta regression objective.
For a batch sampled from every training transition:

\[
L_{\text{one}}
=
\operatorname{MSE}
\left(
f_\phi(\operatorname{norm}_x(x_t)),
\operatorname{norm}_\Delta(\Delta s_t)
\right).
\]

This term uses true recorded states and protects local dynamics accuracy.

### Rollout Term

For a sequence batch, free rollout receives the true state only at offset zero.
Every later input state is the model's previous prediction. At each horizon
offset `k`, define the physical state error as

\[
e_k=
[
\hat{x}_k-x_k,
\hat{y}_k-y_k,
\operatorname{wrap}(\hat{\theta}_k-\theta_k),
\hat{v}_k-v_k
].
\]

The error is divided component-wise by the target-delta standard deviation
fitted on training transitions, preventing metres, radians, and metres per
second from being combined without a common scale. The rollout loss averages
over batch, horizon, and state dimension:

\[
L_{\text{rollout}}
=
\frac{1}{H}
\sum_{k=1}^{H}
\operatorname{mean}
\left(
\left(
\frac{e_k}{\sigma_{\Delta s}}
\right)^2
\right).
\]

Averaging over `H` keeps the nominal scale from growing merely because the
configured horizon is longer. Later-step errors can still be larger because
they contain genuine accumulated error.

### Total Loss

For `H > 1`:

\[
L_{\text{total}}
=
L_{\text{one}}
+
\lambda L_{\text{rollout}}.
\]

The first experiment uses `lambda=1.0`. For `H=1`, training uses only the
existing one-step loss; it does not add a duplicated one-step rollout term.

Each multi-step optimizer update consumes one ordinary transition batch and
one sequence-window batch. The epoch length is defined by the number of
ordinary transition batches, preserving the baseline number of optimizer
updates. Sequence batches use a deterministic seeded permutation and cycle if
there are fewer sequence batches than transition batches.

## Validation and Checkpoint Selection

Input and target normalizers are always fitted from all training transitions,
never from validation or test data.

For horizon 1, validation remains the existing full-split normalized one-step
MSE. For multi-step training, validation reports:

- one-step loss over every validation transition;
- rollout loss over every eligible validation sequence window;
- total validation loss using the configured lambda.

The best epoch is selected by total validation loss. Test transitions are not
used for optimization, normalization, early selection, or loss weighting.
The saved `test_metrics` remain the existing physical-unit one-step metrics;
long-horizon test results continue to come from `world-model-diagnose`.

## Training Results and Checkpoints

`TrainingResult` and `LoadedWorldModel` retain `train_losses` and
`validation_losses` as the total-loss histories. They add explicit histories
for:

```text
train_one_step_losses
train_rollout_losses
validation_one_step_losses
validation_rollout_losses
```

Horizon-1 runs store zero rollout histories, so all histories have consistent
epoch lengths. Existing consumers that read `train_losses`,
`validation_losses`, `best_epoch`, model weights, or normalizers continue to
work.

New checkpoints use format version 3 and record the component histories plus
`rollout_horizon` and `rollout_loss_weight` in `training_config`. The loader
continues to accept versions 1 and 2. When loading an older one-step
checkpoint, its total loss is also exposed as its one-step loss and its rollout
history is synthesized as zeros of matching length.

The existing training plot continues to show total train and validation loss.
For version-3 checkpoints it additionally shows one-step and rollout component
curves without changing how older checkpoints render.

## Training Summary

The JSON summary returned by `run_training` retains its current keys and adds:

```text
rollout_horizon
rollout_loss_weight
train_sequence_windows
validation_sequence_windows
initial_train_one_step_loss
final_train_one_step_loss
initial_train_rollout_loss
final_train_rollout_loss
best_validation_one_step_loss
best_validation_rollout_loss
```

Sequence-window counts are zero for horizon-1 runs. Component losses remain
finite JSON numbers; rollout losses are `0.0` when rollout training is disabled.
The two `best_validation_*` component values come from the epoch selected by
minimum total validation loss; they are not independently minimized epochs.

## Controlled Experiment

After implementation, train two new checkpoints from the same dataset with the
same architecture, split seed, optimizer settings, batch size, and epoch count:

```text
artifacts/world_model_h1.pt   rollout_horizon=1
artifacts/world_model_h10.pt  rollout_horizon=10, rollout_loss_weight=1.0
```

Run `world-model-diagnose` for both checkpoints with identical horizons
`1 5 10 20 50`, window count, bins, and test episodes. Generated checkpoints,
datasets, diagnostic JSON, and plots remain ignored artifacts and are not
committed.

The experiment tests the following hypothesis rather than hard-coding it as a
unit-test requirement:

- horizon-1 physical errors do not materially degrade;
- horizon-10 free-rollout errors decrease;
- horizon-20 and horizon-50 improvements indicate stability beyond the
  training horizon;
- the absolute teacher-forcing/free-rollout gap narrows without both curves
  becoming worse.

A negative result is still a valid experiment result and must be reported, not
hidden by changing seeds or evaluation windows.

## Error Handling

Multi-step training rejects:

- a missing `step_ids` array;
- non-finite states, actions, or next states;
- non-contiguous or duplicate step IDs;
- discontinuous adjacent transitions;
- a non-positive rollout horizon;
- a negative or non-finite rollout weight;
- no eligible training or validation windows for the requested horizon.

Messages identify the failing episode or requested horizon where applicable.

## Testing

Tests use small deterministic state sequences and cover:

- exact sequence-window shapes and deterministic ordering;
- prevention of episode-boundary crossing;
- rejection of non-contiguous steps and discontinuous transitions;
- preservation of short-episode transitions in the one-step population;
- a differentiable rollout whose gradients reach model parameters through
  later prediction steps;
- wrapped heading errors in the rollout loss;
- horizon-1 behavior matching the existing one-step path;
- separate finite one-step, rollout, and total histories;
- total-loss best-epoch selection in multi-step mode;
- version-3 checkpoint round trips and version-1/version-2 loading;
- training-plot output for old and new checkpoint formats;
- CLI validation and sequence-window counts in the training summary.

All existing tests must remain green. A real-data smoke experiment must train
both checkpoints, generate both diagnostic bundles, and produce finite metrics
without adding generated artifacts to Git.

## Success Criteria

Implementation is complete when horizon-1 training remains backward
compatible, horizon-10 training backpropagates through predicted states,
checkpoint loading remains compatible with versions 1 and 2, and the existing
diagnostic command can compare the two newly trained models under an identical
held-out protocol.

The research hypothesis is supported only if the resulting metrics improve
long-horizon free rollout without a material one-step regression. The code is
still considered correct if the controlled experiment rejects that hypothesis
and reports the result faithfully.
