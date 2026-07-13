# State-Delta World Model Design

## Goal

Train the first learned world model for `CarEnv`: given the current state and
applied action, predict the one-step state change.

## Model contract

- Raw state: `[x, y, heading, velocity]`
- Raw action: `[steering, acceleration]`
- Model input: `[x, y, sin(heading), cos(heading), velocity, steering, acceleration]`
- Model target: `[delta_x, delta_y, wrapped_delta_heading, delta_velocity]`
- Reconstructed prediction: current state plus the predicted delta, with the
  resulting heading wrapped to `[-pi, pi)`.

The heading is encoded with sine and cosine so the input is continuous across
the `-pi/pi` boundary. The target heading difference is wrapped for the same
reason.

## Data flow

1. Load `data/transitions.npz`.
2. Split unique episode IDs deterministically into 80% train, 10% validation,
   and 10% test sets. Transitions from one episode must stay in one split.
3. Compute input and target mean/std from the training split only.
4. Train a two-hidden-layer PyTorch MLP with normalized inputs and targets.
5. Report MAE for `x`, `y`, heading in degrees, and velocity on validation and
   test splits in physical units.
6. Save a checkpoint containing model weights, architecture, normalization
   statistics, split episode IDs, and training configuration.

## Components

- `dataset.py`: angle wrapping, episode split, feature/target construction,
  normalization statistics, and array validation.
- `model.py`: the small MLP and checkpoint-safe model configuration.
- `train_world_model.py`: deterministic training loop, evaluation, CLI, and
  checkpoint writing.

## Error handling

Reject missing arrays, incompatible shapes, non-finite numeric values, fewer
than three episodes, invalid split ratios, or zero-variance normalization
dimensions. The CLI must fail with a clear message rather than silently train
on malformed data.

## Verification

Unit tests cover episode isolation, deterministic splitting, wrapped heading
deltas, feature shapes, training loss reduction, physical-unit metrics, and
checkpoint creation. A smoke run trains on the real NPZ dataset and prints
validation/test metrics.
