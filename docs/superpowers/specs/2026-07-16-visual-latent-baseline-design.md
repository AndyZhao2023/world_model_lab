# Visual Latent World-Model Baseline Design

## Goal

Train the first visual world-model baseline on the existing schema-v1 visual
episode artifact:

1. a convolutional autoencoder compresses one `64 x 64` RGB frame into a
   fixed-size latent vector and reconstructs the frame;
2. a residual MLP uses four latent frames plus the four aligned actions to
   predict the next latent;
3. the frozen decoder converts the predicted latent back into a next-frame
   prediction for quantitative and qualitative evaluation.

This increment establishes a reproducible one-step latent-dynamics baseline.
It does not add multi-step visual rollout training, reward prediction, policy
learning, planning, or MPC.

## Why This Is the Next Increment

The visual artifact and lazy window layer already guarantee the temporal
contract:

```text
frames[t-3:t+1], actions[t-3:t], action[t] -> frames[t+1]
```

The remaining unknowns are now model-level:

- can a small CNN preserve the visual state in a compact representation?
- given that representation and the applied actions, can a small dynamics
  model predict the next representation?

Training the two phases separately keeps those failure modes distinguishable.
If frame reconstruction is poor, latent dynamics metrics are not yet
meaningful. If reconstruction is good but next-frame prediction is poor, the
problem is in temporal modeling rather than compression.

## Scope

This stage adds:

- a deterministic `64 x 64` convolutional autoencoder;
- an action-conditioned residual latent dynamics MLP;
- PyTorch adapters for episode-selected frames and latent windows;
- one shared whole-episode `80/10/10` split for both phases;
- safe latent and action normalizers fitted from the training split only;
- a two-phase training command;
- a versioned, atomic, no-clobber checkpoint containing both models;
- held-out reconstruction and next-frame metrics;
- a copy-last-frame baseline;
- a deterministic prediction preview;
- tests for shapes, alignment, split reuse, checkpoint round trips, CLI
  behavior, and a tiny end-to-end training run.

This stage does not add:

- stochastic VAE sampling or a KL loss;
- recurrent, Transformer, JEPA, diffusion, or video-token models;
- multi-step latent rollout loss;
- reward, termination, value, policy, or action prediction;
- online environment interaction;
- MPS/CUDA-specific behavior;
- a persisted latent dataset;
- modification of visual schema version 1.

## Risk Classification

This is a Tier-1 change under `maintaining-system-understanding`.

The changed mechanism is an offline training and checkpoint contract. It does
not alter robot control, concurrency, authorization, destructive state, or
distributed coordination.

## Architecture

```text
visual_episodes.npz
        |
        +--> split complete episode IDs once
        |       train / validation / test
        |
        +--> selected frame indices
        |       |
        |       +--> ConvAutoencoder
        |               frame -> latent -> reconstructed frame
        |
        +--> encode every canonical frame once with best autoencoder
                |
                +--> fit train-only latent/action normalizers
                |
                +--> build compact latent windows from existing offsets
                        |
                        +--> LatentDynamicsMLP
                                4 latents + 4 actions
                                -> next normalized latent
                                -> frozen decoder
                                -> predicted next frame
```

The full visual frames remain stored once in the visual artifact. The
autoencoder reads selected frames lazily. After autoencoder training, every
frame is encoded once into a small in-memory latent array. Dynamics training
may materialize compact latent windows because their memory cost is small and
they no longer duplicate images.

## Model Definitions

### Convolutional autoencoder

The encoder uses four stride-2 convolutions:

```text
[B, 3, 64, 64]
 -> [B, C, 32, 32]
 -> [B, 2C, 16, 16]
 -> [B, 4C, 8, 8]
 -> [B, 4C, 4, 4]
 -> Linear
 -> [B, latent_dim]
```

The decoder mirrors this path with a linear projection and four transposed
convolutions. Its final sigmoid constrains reconstructed pixels to `[0, 1]`.

Default configuration:

```text
latent_dim = 32
base_channels = 16
```

The first implementation uses pixel MSE. It does not add perceptual loss,
foreground weighting, or a variational objective. The metrics and copy-last
baseline must make any static-background shortcut visible.

### Latent dynamics MLP

For each window:

```text
context_latents      [B, 4, latent_dim]
history_actions      [B, 3, 2]
current_action       [B, 2]
```

The four action vectors are the three actions connecting the context frames
plus the current action connecting the last context frame to the target.

The model concatenates:

```text
flatten(context_latents)
+ flatten(history_actions)
+ current_action
```

and predicts a residual relative to the final context latent:

```text
predicted_next = context_latents[:, -1] + predicted_delta
```

Default hidden size:

```text
dynamics_hidden_size = 256
```

The MLP operates on normalized latent vectors and normalized actions.

## Data and Normalization

### Frame phase

`VisualFrameDataset` receives selected episode IDs and resolves all frame
indices through `frame_offsets`. It returns:

```text
float32 CHW frame in [0, 1]
```

Frames from one episode remain entirely within one split. The autoencoder does
not train from overlapping visual windows, so interior frames are not
implicitly overweighted.

### Latent phase

After the best autoencoder is restored, the encoder processes every canonical
frame exactly once:

```text
latent_frames [F, latent_dim]
```

For each `VisualWindowIndex` row, vectorized offsets create:

```text
context_latents          [N, 4, latent_dim]
history_actions          [N, 3, 2]
current_actions          [N, 2]
target_latents           [N, latent_dim]
last_frame_indices       [N]
target_frame_indices     [N]
episode_ids              [N]
step_ids                 [N]
```

No physical `states`, rewards, done flags, or terminal reasons become model
inputs.

### Train-only normalizers

The latent normalizer is fitted from all encoded frames belonging to training
episodes. The action normalizer is fitted from all transitions belonging to
training episodes.

For each feature:

```text
normalized = (value - mean) / std
```

If a training feature has standard deviation below `1e-6`, its stored
standard deviation becomes `1.0`. This keeps constant action or latent
dimensions finite without inventing variation.

Validation and test values never influence these statistics.

## Training Protocol

### Phase 1: autoencoder

- optimizer: Adam;
- loss: pixel MSE;
- train loader: shuffled with an explicitly seeded generator;
- validation loader: deterministic order;
- best checkpoint: lowest validation MSE;
- test metrics: evaluated once after restoring the best weights.

Defaults:

```text
epochs = 20
batch_size = 128
learning_rate = 1e-3
```

### Phase 2: latent dynamics

- freeze the trained autoencoder parameters;
- encode all frames once;
- fit train-only latent and action normalizers;
- train the MLP with normalized next-latent MSE;
- select the lowest validation latent MSE;
- evaluate held-out latent and decoded-frame metrics once.

Defaults:

```text
epochs = 50
batch_size = 256
learning_rate = 1e-3
```

Both phases run on CPU in this first baseline. Device selection and mixed
precision are deferred until the baseline contract is stable.

## Metrics

### Autoencoder reconstruction

Report on held-out test frames:

```text
pixel_mse
pixel_mae
psnr_db
frames
```

PSNR uses `max(mse, 1e-12)` so JSON and checkpoints remain finite.

### Latent next-frame prediction

Report on held-out test windows:

```text
normalized_latent_mse
pixel_mse
pixel_mae
psnr_db
changed_pixel_mae
changed_pixel_count
copy_last_pixel_mse
copy_last_pixel_mae
copy_last_changed_pixel_mae
windows
```

`changed_pixel_mae` only includes pixels whose target RGB value differs from
the final context frame. This prevents the static scene background from
hiding motion errors.

The copy-last metrics are mandatory. Since adjacent frames are similar, a
learned model is not useful merely because its global MSE is small.

## Checkpoint Contract

The command writes one atomic, no-clobber PyTorch checkpoint:

```text
format_version = 1
kind = "visual_latent_world_model"
dataset
model_config
autoencoder_state_dict
dynamics_state_dict
latent_mean
latent_std
action_mean
action_std
split_episode_ids
training_config
autoencoder_history
dynamics_history
autoencoder_test_metrics
dynamics_test_metrics
```

`dataset` records:

```text
resolved path
SHA-256
schema_version
renderer_version
```

The loader uses `torch.load(..., weights_only=True)`, rejects unsupported
format/kind values, reconstructs both models from `model_config`, validates
normalizer shapes and finite positive standard deviations, and restores the
episode split and metric histories.

The checkpoint refuses to overwrite an existing path. Training preflights the
checkpoint and preview paths before performing expensive work.

## Prediction Preview

The command writes an optional PNG for the first deterministic test windows.
Each row shows:

```text
last context | target next frame | predicted next frame | absolute error
```

The preview is diagnostic only and does not affect training or checkpoint
selection. It also refuses to overwrite an existing file.

## Public API and Files

### `src/world_model_lab/visual_latent_model.py`

```python
class ConvAutoencoder(nn.Module): ...
class LatentDynamicsMLP(nn.Module): ...
```

### `src/world_model_lab/visual_latent_data.py`

```python
class VisualFrameDataset(Dataset): ...

@dataclass(frozen=True)
class LatentWindowArrays: ...

def frames_to_tensor(frames: np.ndarray) -> torch.Tensor: ...
def frame_indices_for_episode_ids(...) -> np.ndarray: ...
def transition_indices_for_episode_ids(...) -> np.ndarray: ...
def fit_safe_normalizer(...) -> Normalizer: ...
def encode_all_frames(...) -> np.ndarray: ...
def build_latent_window_arrays(...) -> LatentWindowArrays: ...
```

### `src/world_model_lab/train_visual_latent_model.py`

```python
@dataclass
class PhaseTrainingResult: ...

@dataclass
class LoadedVisualLatentModel: ...

def train_autoencoder(...) -> PhaseTrainingResult: ...
def train_latent_dynamics(...) -> PhaseTrainingResult: ...
def evaluate_autoencoder(...) -> dict[str, float | int]: ...
def evaluate_latent_dynamics(...) -> dict[str, float | int]: ...
def save_visual_latent_checkpoint(...) -> Path: ...
def load_visual_latent_checkpoint(...) -> LoadedVisualLatentModel: ...
def plot_visual_latent_predictions(...) -> Path: ...
def run_visual_latent_training(...) -> dict[str, object]: ...
```

### CLI

Register:

```text
world-model-train-visual-latent
```

Default artifacts:

```text
artifacts/visual_latent_world_model.pt
artifacts/visual_latent_predictions.png
```

## Validation and Failure Behavior

- the visual artifact is validated before indexing or training;
- at least three episodes are required by the shared split function;
- train, validation, and test must each contain at least one eligible visual
  window;
- dimensions, epoch counts, batch sizes, and learning rates must be positive;
- seeds must be non-negative integers;
- non-finite model inputs, latent arrays, losses, metrics, or normalizers are
  rejected;
- the autoencoder only accepts `[B, 3, 64, 64]`;
- latent dynamics enforces exactly four context latents, three history actions,
  and one current action;
- selected episode IDs must remain disjoint and exhaustive;
- checkpoint dataset SHA-256 allows later evaluation to reject a different
  visual artifact;
- output collisions fail before training begins;
- temporary checkpoint or PNG files are cleaned up on encoder failure.

## Test Strategy

Tests cover:

1. autoencoder encode/decode shapes and output range;
2. dynamics input contract and residual output shape;
3. frame tensor conversion and frame-index episode isolation;
4. latent-window alignment with the existing visual offsets;
5. safe normalizer behavior for constant dimensions;
6. training loss histories and best-weight restoration on tiny synthetic data;
7. decoded next-frame metrics and copy-last metrics;
8. checkpoint round trip and strict format validation;
9. CLI registration and help;
10. output no-clobber preflight;
11. tiny end-to-end training from a valid visual artifact;
12. README documentation;
13. the complete existing repository test suite.

## Canonical Smoke Experiment

After tests pass, run a bounded CPU smoke experiment on the real 250-episode
artifact with reduced capacity and epochs:

```bash
world-model-train-visual-latent \
  --data data/visual_episodes.npz \
  --output artifacts/visual_latent_world_model.pt \
  --preview artifacts/visual_latent_predictions.png \
  --latent-dim 16 \
  --base-channels 8 \
  --dynamics-hidden-size 64 \
  --autoencoder-epochs 2 \
  --dynamics-epochs 5 \
  --autoencoder-batch-size 128 \
  --dynamics-batch-size 256 \
  --seed 0 \
  --split-seed 42
```

This is a pipeline smoke baseline, not a convergence claim. The summary must
include phase losses, held-out metrics, split sizes, window counts, checkpoint
path, and preview path.

## Completion Criteria

This increment is complete when:

- both model classes and data adapters implement the contracts above;
- the two-phase command trains on a tiny test artifact;
- checkpoint loading reproduces predictions;
- full tests pass;
- the real-data smoke experiment writes a checkpoint and preview;
- the split still contains `200/25/25` episodes and `8053` total windows;
- output metrics include copy-last comparison;
- no physical state, reward, termination, policy, or MPC behavior enters the
  model.
