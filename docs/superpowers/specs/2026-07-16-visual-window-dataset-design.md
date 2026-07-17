# Visual Window Dataset Design

## Goal

Add a model-independent, memory-efficient data layer that converts the
episode-oriented visual artifact into causally aligned one-step training
samples. The layer must split complete episodes before constructing windows,
preserve the artifact's original array formats, and remain reusable by a
future pixel predictor, autoencoder, VAE, or latent world model.

This increment prepares visual training data but does not train a neural
network.

## Why This Is the Next Increment

The visual observation bridge already provides one canonical sequence per
episode:

\[
o_0, a_0, o_1, a_1, \ldots, a_{T-1}, o_T.
\]

The next model needs four recent frames and the actions that connect them.
Constructing those samples correctly is a separate invariant from choosing a
network architecture. Isolating the invariant now makes later model
comparisons use the same episode split and temporal alignment.

## Scope

This stage adds:

- a lightweight index for every valid four-frame, one-step visual sample;
- lazy sample extraction without pre-materializing repeated image windows;
- whole-episode train, validation, and test splitting through the existing
  `split_episode_ids` function;
- explicit reporting of eligible and too-short episodes;
- deterministic sample ordering and metadata for diagnostics;
- tests for alignment, isolation, split integrity, validation, and dtypes;
- generation and verification of the canonical local visual artifact.

This stage does not add:

- image normalization, channel reordering, tensor conversion, or batching;
- an autoencoder, VAE, latent dynamics model, or pixel predictor;
- reward, termination, value, policy, MPC, or PPO targets;
- a second NPZ containing duplicated windows;
- changes to the visual artifact schema or existing state-model pipeline.

## Alternatives Considered

### Eagerly materialize every window

This is simple to inspect, but 8,053 samples would duplicate four context
frames and one target frame. At 64 by 64 RGB this is roughly 500 MB of raw
image arrays before training batches or tensor conversion. It also creates an
unnecessary second representation of the same data.

### Persist a window-oriented NPZ

This reduces slicing work at training time but has the same duplication cost,
creates another artifact version to maintain, and risks divergence from the
canonical episode artifact.

### Index-backed lazy windows

This is the selected design. Frames and actions remain stored once. Each
sample index records only an episode location and a time step; arrays are
sliced and copied when that sample is requested. It is model-independent and
compatible with PyTorch's map-style dataset protocol without importing
PyTorch into the core data layer.

## Module and Public API

Add `src/world_model_lab/visual_windows.py` with these public concepts:

```python
@dataclass(frozen=True)
class VisualWindowIndex:
    episode_indices: np.ndarray
    step_ids: np.ndarray
    selected_episode_ids: np.ndarray
    eligible_episode_ids: np.ndarray
    skipped_episode_ids: np.ndarray

    @property
    def count(self) -> int: ...


class VisualWindowDataset:
    index: VisualWindowIndex

    def __len__(self) -> int: ...
    def __getitem__(self, item: int) -> dict[str, np.ndarray | int]: ...


def build_visual_window_index(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowIndex: ...


def build_visual_window_dataset(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> VisualWindowDataset: ...


def build_visual_window_splits(
    dataset: Mapping[str, np.ndarray],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, VisualWindowDataset]: ...
```

Factory functions are the supported construction path. They validate the
visual artifact and index contract once during construction. `__getitem__`
does not repeat full-artifact validation.

`build_visual_window_splits` calls the existing `split_episode_ids` function;
it does not reimplement split counts or randomization. It validates the source
once, creates the three indexes from the returned episode-ID arrays, and
returns exactly `train`, `validation`, and `test` datasets.

## Index Semantics

For an episode with `T` transitions, valid current-action steps are:

\[
t \in \{3, \ldots, T-1\}.
\]

The episode contributes:

\[
\max(0, T-3)
\]

samples. Each index row stores the episode's position in the canonical
`episode_ids` array and the local `step_id=t`. Global frame and action offsets
are resolved lazily through `frame_offsets` and `transition_offsets`.

Indexes follow the order of `selected_episode_ids`. Within one episode,
`step_ids` are strictly increasing. The same visual artifact, selected IDs,
and seed therefore produce the same sample ordering.

An episode with fewer than four transitions contributes no samples. This is a
valid condition, not malformed data. Its ID is recorded in
`skipped_episode_ids`; contributing IDs are recorded in
`eligible_episode_ids`. A non-empty selected split may therefore produce an
empty dataset.

## Sample Contract

For `(episode, t)`, `dataset[i]` returns:

```text
context_frames   frames[t-3:t+1]   uint8    [4, 64, 64, 3]
history_actions  actions[t-3:t]    float64  [3, 2]
current_action   actions[t]        float64  [2]
target_frame     frames[t+1]       uint8    [64, 64, 3]
episode_id                           int
step_id                              int
```

The history actions are exactly the actions connecting the four context
frames. The current action connects the last context frame to the target:

\[
(o_{t-3:t}, a_{t-3:t-1}), a_t \rightarrow o_{t+1}.
\]

Returned arrays are owned copies. Mutating a sample cannot mutate the source
artifact or another sample. Copying is bounded to the requested sample, about
60 KB of image data, rather than all possible windows.

The data layer deliberately preserves `uint8` height-width-channel images and
`float64` actions. A later model adapter owns conversion to `float32`, division
by 255, and channel-first layout. Physical states remain in the underlying
artifact for diagnostics but are not exposed as model inputs, preventing the
visual model from accidentally observing hidden velocity.

`episode_id` and `step_id` are returned as Python integers. They support
debugging and per-episode evaluation but are not model inputs.

## Split Integrity

Episode IDs are split before any visual window is constructed:

```text
load_visual_dataset
        -> split_episode_ids
        -> build one independent index per split
        -> construct train, validation, and test datasets
```

No frame or action from an episode can appear in more than one split. This is
stronger than randomly splitting windows, which would place highly overlapping
four-frame histories from the same trajectory into training and evaluation.

The returned datasets retain their `selected_episode_ids`, allowing training
checkpoints and diagnostics to record or compare the exact split later.

## Validation and Errors

Construction first delegates the artifact contract to
`validate_visual_dataset`. Additional window-layer validation requires
`selected_episode_ids` to be a non-empty, one-dimensional integer array with
no duplicates. Boolean IDs are not accepted. Every selected ID must exist in
the artifact.

The index arrays must be one-dimensional, have equal sample counts, use
`int64`, and refer only to valid episodes and valid steps. Factories create
these arrays; callers do not supply raw global offsets.

`VisualWindowDataset.__getitem__` accepts integer-like scalar indices,
including NumPy integers. Negative indices follow normal Python sequence
semantics. Out-of-range access raises `IndexError`. Slices, arrays, floats, and
booleans raise `TypeError` in this first version.

Neither construction nor item access mutates caller-owned arrays.

## Test Strategy

Add `tests/test_visual_windows.py` using small synthetic visual artifacts.
Tests cover:

1. exact frame and action alignment for the first and middle windows;
2. the last legal window targeting the final episode frame;
3. absence of windows that cross episode offsets;
4. deterministic ordering across selected episodes;
5. correct eligible and skipped episode reporting;
6. a legitimate empty dataset when all selected episodes are too short;
7. disjoint and exhaustive whole-episode train, validation, and test splits;
8. deterministic split results for a fixed seed;
9. rejection of missing, duplicate, empty, non-integer, Boolean, and
   incorrectly shaped selected IDs;
10. scalar index validation, negative indexing, and bounds behavior;
11. exact sample keys, shapes, dtypes, and Python metadata types;
12. copy isolation between returned samples and the source artifact.

The complete existing test suite must also pass. No current state-model or
visual-artifact test should require modification except shared test fixtures if
needed to express valid episode lengths.

## Canonical Local Artifact

After implementation and tests pass, generate the formal local dataset with:

```bash
world-model-build-visual-data \
  --data data/transitions.npz \
  --output data/visual_episodes.npz \
  --preview artifacts/visual_episode_preview.gif
```

The command must report the established source counts:

```text
episodes                       250
transitions                  8,768
frames                       9,018
four_frame_eligible_episodes   225
one_step_visual_samples      8,053
```

The new window-split helper is then exercised against that artifact to verify
that split sample counts sum to 8,053 and episode sets remain disjoint. The
generated NPZ and preview are local experimental artifacts and are not added
to Git.

## Completion Criteria

This stage is complete when:

- the lazy window API implements the contracts above;
- focused and full tests pass;
- the canonical local visual artifact is generated and validates;
- its split window counts and episode isolation are verified;
- documentation explains how the next model can consume the returned sample;
- no neural-network training behavior has been introduced.
