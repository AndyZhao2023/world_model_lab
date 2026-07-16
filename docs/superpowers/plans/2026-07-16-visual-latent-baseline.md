# Visual Latent World-Model Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and persist a two-phase visual baseline consisting of a convolutional frame autoencoder and an action-conditioned one-step latent dynamics model.

**Architecture:** Split complete episode IDs once, train the autoencoder from unique frames, encode all canonical frames once, and train a residual MLP from compact normalized latent windows. Persist both models, normalizers, split IDs, histories, dataset provenance, held-out metrics, and a qualitative prediction preview in one reproducible artifact contract.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Matplotlib, existing schema-v1 visual artifacts, existing lazy visual-window indexes, standard-library `unittest`.

## Global Constraints

- Preserve visual schema version 1 and its exact renderer provenance contract.
- Use exactly four context frames, three history actions, and one current action.
- Split complete episodes once before frame or window construction.
- Autoencoder input and output are `float32 NCHW` tensors in `[0, 1]`.
- Default `latent_dim=32`, `base_channels=16`, and `dynamics_hidden_size=256`.
- Latent dynamics predicts a residual relative to the final normalized context latent.
- Fit latent and action normalizers from training episodes only.
- Replace feature standard deviations below `1e-6` with `1.0`.
- Run the first implementation on CPU without mixed precision.
- Do not expose physical states, rewards, done flags, or terminal reasons as model inputs.
- Do not add VAE sampling, multi-step visual rollout, policy learning, or MPC.
- Do not persist a second latent dataset.
- Checkpoints and previews are atomic and refuse to overwrite existing files.

---

## File Structure

- Create `src/world_model_lab/visual_latent_model.py`: CNN autoencoder and residual latent dynamics MLP.
- Create `src/world_model_lab/visual_latent_data.py`: frame tensor adapter, episode frame/transition indexes, safe normalizers, full-frame encoder, and compact latent-window arrays.
- Create `src/world_model_lab/train_visual_latent_model.py`: two training phases, evaluation, checkpoint loading/saving, preview rendering, orchestration, and CLI.
- Create `tests/test_visual_latent_model.py`: model shape and contract tests.
- Create `tests/test_visual_latent_data.py`: tensor, index, normalization, and temporal-alignment tests.
- Create `tests/test_train_visual_latent_model.py`: training, metrics, checkpoint, no-clobber, CLI, and tiny end-to-end tests.
- Modify `pyproject.toml`: register `world-model-train-visual-latent`.
- Modify `README.md`: explain the two phases, command, metrics, artifact boundaries, and next step.

### Task 1: Visual Model Primitives

**Files:**
- Create: `src/world_model_lab/visual_latent_model.py`
- Create: `tests/test_visual_latent_model.py`

**Interfaces:**
- Produces: `ConvAutoencoder(latent_dim=32, base_channels=16)`.
- Produces: `LatentDynamicsMLP(latent_dim=32, hidden_size=256, context_frames=4, action_size=2)`.
- Later tasks call `ConvAutoencoder.encode`, `ConvAutoencoder.decode`, and both modules' `forward` methods.

- [ ] **Step 1: Write failing autoencoder shape tests**

Create `tests/test_visual_latent_model.py`:

```python
import unittest

import torch

from world_model_lab.visual_latent_model import (
    ConvAutoencoder,
    LatentDynamicsMLP,
)


class ConvAutoencoderTest(unittest.TestCase):
    def test_encode_decode_and_forward_shapes_are_exact(self):
        model = ConvAutoencoder(latent_dim=12, base_channels=4)
        images = torch.zeros((5, 3, 64, 64), dtype=torch.float32)

        latents = model.encode(images)
        reconstructions = model.decode(latents)
        forwarded = model(images)

        self.assertEqual(tuple(latents.shape), (5, 12))
        self.assertEqual(tuple(reconstructions.shape), (5, 3, 64, 64))
        self.assertEqual(tuple(forwarded.shape), (5, 3, 64, 64))
        self.assertTrue(torch.all(reconstructions >= 0.0))
        self.assertTrue(torch.all(reconstructions <= 1.0))

    def test_invalid_shapes_and_configuration_are_rejected(self):
        for kwargs in (
            {"latent_dim": 0},
            {"base_channels": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ConvAutoencoder(**kwargs)
        model = ConvAutoencoder(latent_dim=8, base_channels=4)
        with self.assertRaisesRegex(ValueError, r"\[B, 3, 64, 64\]"):
            model.encode(torch.zeros((2, 3, 32, 32)))
        with self.assertRaisesRegex(ValueError, r"\[B, 8\]"):
            model.decode(torch.zeros((2, 7)))
```

- [ ] **Step 2: Write failing latent dynamics tests**

Append:

```python
class LatentDynamicsMLPTest(unittest.TestCase):
    def test_predicts_one_next_latent_from_exact_context(self):
        model = LatentDynamicsMLP(
            latent_dim=8,
            hidden_size=16,
            context_frames=4,
        )
        output = model(
            torch.zeros((6, 4, 8)),
            torch.zeros((6, 3, 2)),
            torch.zeros((6, 2)),
        )

        self.assertEqual(tuple(output.shape), (6, 8))

    def test_zero_network_residual_returns_last_context_latent(self):
        model = LatentDynamicsMLP(
            latent_dim=4,
            hidden_size=8,
            context_frames=4,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        context = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)

        output = model(
            context,
            torch.zeros((2, 3, 2)),
            torch.zeros((2, 2)),
        )

        torch.testing.assert_close(output, context[:, -1])

    def test_dynamics_rejects_misaligned_inputs(self):
        model = LatentDynamicsMLP(latent_dim=4, hidden_size=8)
        cases = (
            (
                torch.zeros((2, 3, 4)),
                torch.zeros((2, 3, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 4)),
                torch.zeros((2, 2, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 4)),
                torch.zeros((2, 3, 2)),
                torch.zeros((3, 2)),
            ),
        )
        for context, history, current in cases:
            with self.subTest(shapes=(context.shape, history.shape, current.shape)):
                with self.assertRaises(ValueError):
                    model(context, history, current)
```

- [ ] **Step 3: Run model tests and verify import failure**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_model -v
```

Expected: import fails because `world_model_lab.visual_latent_model` does not exist.

- [ ] **Step 4: Implement the two models**

Create `src/world_model_lab/visual_latent_model.py` with:

```python
"""Neural networks for the first visual latent world-model baseline."""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    image_size = 64
    image_channels = 3

    def __init__(self, *, latent_dim: int = 32, base_channels: int = 16) -> None:
        super().__init__()
        if latent_dim <= 0 or base_channels <= 0:
            raise ValueError("latent_dim and base_channels must be positive")
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        encoded_channels = 4 * self.base_channels
        self.encoder_convolutions = nn.Sequential(
            nn.Conv2d(3, self.base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.base_channels, 2 * self.base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(2 * self.base_channels, encoded_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(encoded_channels, encoded_channels, 4, 2, 1),
            nn.ReLU(),
        )
        self.encoder_projection = nn.Linear(
            encoded_channels * 4 * 4,
            self.latent_dim,
        )
        self.decoder_projection = nn.Linear(
            self.latent_dim,
            encoded_channels * 4 * 4,
        )
        self.decoder_convolutions = nn.Sequential(
            nn.ConvTranspose2d(
                encoded_channels,
                encoded_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                encoded_channels,
                2 * self.base_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * self.base_channels,
                self.base_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(self.base_channels, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("images must have shape [B, 3, 64, 64]")
        features = self.encoder_convolutions(images)
        return self.encoder_projection(features.flatten(start_dim=1))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"latents must have shape [B, {self.latent_dim}]"
            )
        encoded_channels = 4 * self.base_channels
        features = self.decoder_projection(latents).reshape(
            latents.shape[0],
            encoded_channels,
            4,
            4,
        )
        return self.decoder_convolutions(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


class LatentDynamicsMLP(nn.Module):
    action_size = 2

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        hidden_size: int = 256,
        context_frames: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_size <= 0:
            raise ValueError("latent_dim and hidden_size must be positive")
        if context_frames < 2:
            raise ValueError("context_frames must be at least two")
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.context_frames = int(context_frames)
        input_size = (
            self.context_frames * self.latent_dim
            + self.context_frames * self.action_size
        )
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.latent_dim),
        )

    def forward(
        self,
        context_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_latents.shape[0]
        if context_latents.shape != (
            batch_size,
            self.context_frames,
            self.latent_dim,
        ):
            raise ValueError("context_latents have an invalid shape")
        if history_actions.shape != (
            batch_size,
            self.context_frames - 1,
            self.action_size,
        ):
            raise ValueError("history_actions have an invalid shape")
        if current_action.shape != (batch_size, self.action_size):
            raise ValueError("current_action has an invalid shape")
        model_input = torch.cat(
            (
                context_latents.flatten(start_dim=1),
                history_actions.flatten(start_dim=1),
                current_action,
            ),
            dim=1,
        )
        return context_latents[:, -1] + self.network(model_input)
```

- [ ] **Step 5: Run focused model tests**

Run the command from Step 3.

Expected: all model tests pass.

### Task 2: Visual-to-Torch and Latent Window Data Layer

**Files:**
- Create: `src/world_model_lab/visual_latent_data.py`
- Create: `tests/test_visual_latent_data.py`

**Interfaces:**
- Consumes: schema-v1 visual mappings, `VisualWindowIndex`, `Normalizer`, and `ConvAutoencoder`.
- Produces: `VisualFrameDataset`, `LatentWindowArrays`, safe normalizers, full-frame latents, and exact frame/transition index helpers.

- [ ] **Step 1: Add failing frame conversion and episode-index tests**

Create tests that assert:

```python
frames_to_tensor(np.zeros((2, 64, 64, 3), dtype=np.uint8))
```

returns `[2, 3, 64, 64]`, `float32`, and values in `[0, 1]`; reject wrong
dtype/shape; and assert `frame_indices_for_episode_ids` plus
`transition_indices_for_episode_ids` follow selected episode order without
crossing offsets.

- [ ] **Step 2: Add failing safe-normalizer tests**

Use:

```python
values = np.asarray([[1.0, 2.0], [1.0, 4.0]])
normalizer = fit_safe_normalizer(values)
```

Assert the first standard deviation is `1.0`, the second is positive, all
values are finite, and normalization round-trips through `denormalize`.

- [ ] **Step 3: Add failing latent-window alignment tests**

Build a synthetic visual artifact using `make_visual_dataset((5, 4, 2))`,
create deterministic latent rows:

```python
latent_frames = np.arange(frame_count * 3, dtype=np.float32).reshape(
    frame_count,
    3,
)
```

Build the index for episode IDs `[10, 11]` and assert that the first row maps
to latent frames `0:4`, target frame `4`, actions `0:3`, current action `3`,
and metadata `(episode_id=10, step_id=3)`. Assert the second episode starts
from its own offsets.

- [ ] **Step 4: Implement the data adapter**

Create the exact public objects:

```python
class VisualFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        selected_episode_ids: np.ndarray,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, item: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class LatentWindowArrays:
    context_latents: np.ndarray
    history_actions: np.ndarray
    current_actions: np.ndarray
    target_latents: np.ndarray
    last_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray

    @property
    def count(self) -> int: ...


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor: ...
def frame_indices_for_episode_ids(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> np.ndarray: ...
def transition_indices_for_episode_ids(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> np.ndarray: ...
def fit_safe_normalizer(
    values: np.ndarray,
    *,
    minimum_std: float = 1e-6,
) -> Normalizer: ...
def encode_all_frames(
    model: ConvAutoencoder,
    frames: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray: ...
def build_latent_window_arrays(
    dataset: Mapping[str, np.ndarray],
    index: VisualWindowIndex,
    latent_frames: np.ndarray,
) -> LatentWindowArrays: ...
```

Use vectorized frame/action offsets:

```python
history = CONTEXT_FRAMES - 1
frame_starts = frame_offsets[index.episode_indices] + index.step_ids - history
action_starts = (
    transition_offsets[index.episode_indices] + index.step_ids - history
)
context_indices = frame_starts[:, None] + np.arange(CONTEXT_FRAMES)
history_action_indices = action_starts[:, None] + np.arange(history)
current_action_indices = action_starts + history
target_frame_indices = frame_starts + CONTEXT_FRAMES
```

Return owned arrays and reject non-finite/misaligned latents.

- [ ] **Step 5: Run focused data tests**

Run:

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_latent_data tests.test_visual_windows -v
```

Expected: all tests pass.

### Task 3: Autoencoder Training Phase

**Files:**
- Create: `src/world_model_lab/train_visual_latent_model.py`
- Create: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Consumes: `VisualFrameDataset`, `ConvAutoencoder`, selected split IDs.
- Produces: `PhaseTrainingResult`, `train_autoencoder`, and `evaluate_autoencoder`.

- [ ] **Step 1: Add failing phase-result and autoencoder-training tests**

Use a tiny valid visual artifact with at least ten episodes and four
transitions per episode. Train:

```python
result = train_autoencoder(
    dataset,
    split_episode_ids=splits,
    latent_dim=4,
    base_channels=2,
    epochs=2,
    batch_size=8,
    learning_rate=1e-3,
    seed=3,
)
```

Assert two finite train/validation losses, a best epoch in `[1, 2]`, model
output shape, and finite test `pixel_mse`, `pixel_mae`, and `psnr_db`.

- [ ] **Step 2: Implement shared phase training result**

Add:

```python
@dataclass
class PhaseTrainingResult:
    model: nn.Module
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
```

Add validation helpers for positive integer/float hyperparameters and
non-negative seeds.

- [ ] **Step 3: Implement autoencoder training**

`train_autoencoder` must:

1. construct train and validation `VisualFrameDataset` objects;
2. create seeded CPU data loaders with `num_workers=0`;
3. optimize pixel MSE with Adam;
4. aggregate each epoch by sample count;
5. deep-copy the lowest validation-loss state dict;
6. restore the best weights and return an eval-mode model.

Use a separate seed offset for the frame loader:

```python
generator = torch.Generator().manual_seed(seed)
```

- [ ] **Step 4: Implement autoencoder metrics**

`evaluate_autoencoder` iterates held-out frames and returns:

```python
{
    "frames": count,
    "pixel_mse": squared_error_sum / value_count,
    "pixel_mae": absolute_error_sum / value_count,
    "psnr_db": 10.0 * math.log10(
        1.0 / max(pixel_mse, 1e-12)
    ),
}
```

Reject empty test data and non-finite outputs.

- [ ] **Step 5: Run focused autoencoder tests**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_train_visual_latent_model.VisualAutoencoderTrainingTest -v
```

Expected: all tests pass.

### Task 4: Latent Dynamics Training and Metrics

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_train_visual_latent_model.py`

**Interfaces:**
- Consumes: frozen autoencoder, `LatentWindowArrays`, latent/action normalizers.
- Produces: `train_latent_dynamics` and `evaluate_latent_dynamics`.

- [ ] **Step 1: Add failing latent dynamics training tests**

Create compact synthetic arrays where the target normalized latent is a
deterministic residual of the final context latent and current action. Assert
that training histories are finite, best epoch is valid, and the final
training loss is below the initial loss for a sufficiently small deterministic
problem.

- [ ] **Step 2: Implement tensor preparation**

Normalize:

```python
context = latent_normalizer.normalize(arrays.context_latents)
target = latent_normalizer.normalize(arrays.target_latents)
history = action_normalizer.normalize(arrays.history_actions)
current = action_normalizer.normalize(arrays.current_actions)
```

Convert all four arrays to `float32` tensors and use `TensorDataset`.

- [ ] **Step 3: Implement latent dynamics training**

Use Adam and MSE on normalized next latent. Use seeded shuffled train batches,
deterministic validation batches, sample-weighted epoch means, best validation
state restoration, and eval mode on return.

- [ ] **Step 4: Add failing decoded metric and copy-last tests**

Use a stub autoencoder decoder and controlled target/last frames. Assert:

- `normalized_latent_mse` uses predicted versus target normalized latents;
- full-frame MSE/MAE use decoded prediction versus target;
- changed-pixel MAE only counts RGB values in pixels whose target differs from
  the final context frame;
- copy-last metrics use the final context frame;
- every reported numeric value is finite.

- [ ] **Step 5: Implement held-out dynamics evaluation**

Batch predicted normalized latents, denormalize, decode, clamp to `[0, 1]`,
and aggregate:

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

The changed-pixel mask is:

```python
changed = torch.any(target != last_context, dim=1, keepdim=True)
```

Expand it across the three RGB channels when summing MAE.

- [ ] **Step 6: Run focused dynamics tests**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_train_visual_latent_model.VisualLatentDynamicsTrainingTest -v
```

Expected: all tests pass.

### Task 5: Checkpoint, Preview, Orchestration, and CLI

**Files:**
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_train_visual_latent_model.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `LoadedVisualLatentModel`, checkpoint save/load, preview rendering,
  `run_visual_latent_training`, and `main`.

- [ ] **Step 1: Add failing checkpoint round-trip tests**

Build tiny model instances and normalizers. Save a checkpoint, load it, and
assert:

- format version and kind are exact;
- model configurations and predictions round-trip;
- split IDs, histories, metrics, training config, and dataset provenance are
  preserved;
- normalizer arrays return as `float64`;
- object-array/pickle loading is never enabled.

Also mutate `format_version`, `kind`, normalizer shape, and standard deviation
in separate payloads and assert named `ValueError` messages.

- [ ] **Step 2: Implement atomic no-clobber checkpoint**

Use:

```python
write_new_file_atomically(
    output,
    writer=lambda handle: torch.save(payload, handle),
    exists_message=f"checkpoint already exists: {output}",
)
```

The payload follows the design specification exactly and stores tensors for
weights, normalizers, and split IDs.

- [ ] **Step 3: Implement strict loader**

Load with:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

Validate `format_version == 1`, exact `kind`, positive model dimensions,
normalizer shapes, finite positive standard deviations, exact split names,
finite histories, and valid best epochs before returning:

```python
@dataclass
class LoadedVisualLatentModel:
    autoencoder: ConvAutoencoder
    dynamics: LatentDynamicsMLP
    latent_normalizer: Normalizer
    action_normalizer: Normalizer
    split_episode_ids: dict[str, np.ndarray]
    training_config: dict[str, Any]
    dataset_metadata: dict[str, Any]
    autoencoder_history: dict[str, Any]
    dynamics_history: dict[str, Any]
    autoencoder_test_metrics: dict[str, float | int]
    dynamics_test_metrics: dict[str, float | int]
```

- [ ] **Step 4: Add failing preview tests**

Render two deterministic rows and assert the output is a PNG with four titled
columns. Assert an existing preview is not overwritten.

- [ ] **Step 5: Implement atomic prediction preview**

Use Matplotlib with rows:

```text
Last context | Target | Predicted | Absolute error
```

Encode the selected context latents, run the dynamics model, denormalize,
decode, and write PNG bytes through `write_new_file_atomically`.

- [ ] **Step 6: Add failing end-to-end and preflight tests**

For a tiny ten-episode visual artifact:

```python
summary = run_visual_latent_training(
    data_path=data_path,
    output_path=checkpoint_path,
    preview_path=preview_path,
    latent_dim=4,
    base_channels=2,
    dynamics_hidden_size=8,
    autoencoder_epochs=1,
    dynamics_epochs=1,
    autoencoder_batch_size=8,
    dynamics_batch_size=8,
    seed=3,
    split_seed=19,
)
```

Assert both files exist, the loader succeeds, split episodes are `8/1/1`,
metrics contain copy-last values, and all windows are accounted for. Create an
existing output and preview in separate tests and assert failure occurs before
either training function is called.

- [ ] **Step 7: Implement orchestration**

`run_visual_latent_training` must execute:

```text
preflight paths
load_visual_dataset
split_episode_ids
build VisualFrameDataset per split
train autoencoder
evaluate autoencoder test frames
encode all frames
fit train-only latent normalizer
fit train-only action normalizer
build VisualWindowIndex and LatentWindowArrays per split
train latent dynamics
evaluate held-out dynamics
save checkpoint
write preview
return JSON-safe summary
```

Use `sha256_file` for dataset provenance. Reject any split with zero eligible
windows.

- [ ] **Step 8: Register and test the CLI**

Add to `pyproject.toml`:

```toml
world-model-train-visual-latent = "world_model_lab.train_visual_latent_model:main"
```

The CLI accepts:

```text
--data
--output
--preview
--latent-dim
--base-channels
--dynamics-hidden-size
--autoencoder-epochs
--dynamics-epochs
--autoencoder-batch-size
--dynamics-batch-size
--autoencoder-learning-rate
--dynamics-learning-rate
--seed
--split-seed
```

Convert `FileNotFoundError`, `FileExistsError`, and `ValueError` into
`parser.error(...)`. Print sorted indented JSON with `allow_nan=False`.

- [ ] **Step 9: Run focused end-to-end tests**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_visual_latent_model \
  tests.test_visual_latent_data \
  tests.test_train_visual_latent_model -v
```

Expected: all focused tests pass.

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_train_visual_latent_model.py`
- Generate locally, do not commit:
  - `data/visual_episodes.npz`
  - `artifacts/visual_latent_world_model.pt`
  - `artifacts/visual_latent_predictions.png`

**Interfaces:**
- Documents the user-facing training workflow and the next research boundary.

- [ ] **Step 1: Add a failing README contract test**

Assert the README includes:

```text
world-model-train-visual-latent
ConvAutoencoder
LatentDynamicsMLP
copy-last
不读取 states
暂不接入 MPC
```

- [ ] **Step 2: Document the two-phase baseline**

Add a section after visual-window construction explaining:

- the exact frame and latent flows;
- why autoencoder and dynamics are trained separately;
- the command with default artifacts;
- checkpoint contents;
- full-frame, changed-pixel, and copy-last metrics;
- CPU baseline limitations;
- that this stage is one-step and does not read physical states or use MPC.

- [ ] **Step 3: Run focused and complete tests**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_visual_latent_model \
  tests.test_visual_latent_data \
  tests.test_train_visual_latent_model \
  tests.test_visual_windows -v

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v

PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 4: Run the canonical real-data smoke experiment**

First ensure `data/visual_episodes.npz` exists. Then run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_latent_model \
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

Expected:

- checkpoint and PNG are written;
- episode split is `200/25/25`;
- total latent windows are `8053`;
- all metric values are finite;
- output includes learned and copy-last errors;
- the command makes no convergence or model-quality claim.

- [ ] **Step 5: Reload and verify the real checkpoint**

Run a short Python command that:

- loads the checkpoint with `weights_only=True`;
- verifies its dataset SHA-256 matches the visual artifact;
- rebuilds the exact test split;
- predicts at least one next latent and decoded frame;
- asserts output shape `[1, 3, 64, 64]` and finite values.

- [ ] **Step 6: Final review and commit**

Run:

```bash
git status --short
git diff main...HEAD --check
git diff --stat
```

Stage only source, tests, plan/spec, `pyproject.toml`, and README. Do not stage
ignored data or artifacts. Commit:

```bash
git commit -m "feat: train visual latent world model"
```

## Self-Review

### Spec coverage

- model architecture: Tasks 1 and 4;
- unique-frame autoencoder training: Tasks 2 and 3;
- shared episode split: Tasks 2, 3, and 5;
- train-only normalizers: Tasks 2, 4, and 5;
- copy-last and changed-pixel metrics: Task 4;
- atomic checkpoint and preview: Task 5;
- strict loading and provenance: Task 5;
- CLI, docs, complete tests, and real smoke: Tasks 5 and 6;
- excluded policy/MPC/state inputs: Global Constraints and Task 6.

### Placeholder scan

The plan contains no `TBD`, `TODO`, “implement later,” unspecified error
handling, or unnamed tests. Ellipses only appear in public interface summaries
where the exact implementation is defined by the corresponding task steps.

### Type consistency

- both model and checkpoint APIs use `latent_dim`;
- dynamics uses `history_actions` plural and `current_action` singular;
- data arrays use `current_actions` because they are batched;
- checkpoint normalizers use the existing `Normalizer`;
- every phase uses `PhaseTrainingResult`;
- the loader returns `LoadedVisualLatentModel`.
