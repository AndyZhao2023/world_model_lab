# Visual Observation Bridge Design

## Goal

Add a deterministic visual-observation layer and an episode-oriented RGB data
artifact to the existing 2D car world-model lab. This is the first bridge from
the current state-delta model to an action-conditioned latent video world
model.

The deliverable does not train a new neural network. It turns the existing
ground-truth state trajectories into compact, causally aligned video episodes
that a later encoder and latent-dynamics model can consume.

## Why This Is the Next Increment

The current lab already provides:

- a deterministic `CarEnv` transition function;
- reproducible episode collection;
- episode-level train, validation, and test splits;
- one-step and differentiable multi-step state prediction;
- free-rollout diagnostics and controlled ensemble experiments.

The missing conceptual bridge to systems such as WorldGym is visual
observation modeling. Adding uncertainty statistics or another state-space
model would not close that gap. This stage therefore changes the observation
modality while keeping the existing dynamics, actions, episode boundaries, and
ground-truth states unchanged.

## Scope

This stage adds:

- a deterministic 64-by-64 RGB renderer for an arbitrary finite car state;
- an aspect-preserving world-to-pixel coordinate transform;
- an episode-oriented visual dataset converted from the existing transition
  NPZ;
- causal alignment between consecutive frames and applied actions;
- fixed metadata declaring a four-frame temporal context for later models;
- a deterministic episode GIF preview;
- a CLI for generating and validating the visual artifact;
- tests for rendering, episode reconstruction, action alignment, schema
  validation, and CLI output;
- README documentation for the artifact and its future model contract.

This stage does not add:

- an autoencoder, VAE, latent-dynamics model, diffusion model, or transformer;
- visual one-step or free-rollout training;
- reward, termination, value, policy, MPC, PPO, or Dreamer training;
- uncertainty or confidence-interval experiments;
- a velocity gauge, speed bar, motion trail, or other synthetic proprioceptive
  overlay;
- CUDA, MPS, WorldGym, or World-Gymnast dependencies;
- custom scene geometry beyond the current default `CarEnv` scene.

## Compatibility Contract

The existing `data/transitions.npz` remains the source of truth and is never
modified in place. Existing state-model commands, checkpoints, and tests remain
unchanged.

The new converter reads the same required arrays used by the current training
and diagnostic pipeline:

```text
states
actions
next_states
rewards
dones
episode_ids
step_ids
terminal_reasons
```

The converter validates the source before rendering. It requires finite state,
action, next-state, and reward arrays; matching transition counts; unique and
contiguous step IDs beginning at zero within every episode; and continuity
checked with `np.allclose(..., rtol=0.0, atol=1e-10)` between each ordered
`states[1:]` and `next_states[:-1]`. It also requires exactly one terminal
transition at the end of each episode.

Output episodes are canonical: unique episode IDs are sorted in ascending
numeric order, and transitions inside each episode are sorted by `step_ids`
before validation and rendering. Source row order therefore cannot change the
visual offsets, preview choice, or resulting arrays.

The visual artifact is a separate versioned NPZ. A future schema change must
increment its `schema_version`; consumers must reject unsupported versions.

## Observation Contract

### Input and Output

The renderer receives:

```text
state       [4] float64: x, y, heading, velocity
scene       current default CarEnv geometry
image_size  fixed at 64 for schema version 1
```

It returns:

```text
frame       [64, 64, 3] uint8 RGB
```

The returned array uses height-width-channel order because it is the natural
storage and image-export representation. A later PyTorch dataset may convert it
to channel-height-width float tensors at the model boundary.

The renderer is observational only. It must not call `CarEnv.step`, reset the
environment, append to its trajectory, or mutate any environment property.
The same state and scene always produce an identical frame.

### Visible State

A single frame visually represents:

- `x` and `y` through the car centre;
- `heading` through an orientation marker;
- the fixed world boundary, obstacle, and goal.

Velocity is deliberately not drawn. Two states that differ only in velocity
produce identical individual frames. This is intentional: the later model
must infer motion from a fixed four-frame history, matching the partial
observability of ordinary video instead of receiving an artificial dashboard.

Four frames alone do not preserve every hidden velocity update. In
`CarEnv.step`, acceleration changes `velocity[t+1]`, while the position and
heading in frame `t+1` still use `velocity[t]`. Different accelerations can
therefore produce the same immediate next frame and diverge only later. The
future dynamics design must retain recent applied actions alongside frame
history, or carry an equivalent recurrent latent state. A memoryless
`(current_frame, current_action) -> next_frame` model is explicitly outside the
promised follow-on contract.

### Coordinate Transform

The world is wider than it is tall, while the output is square. The renderer
must preserve physical aspect ratio rather than stretch the world independently
along the two axes.

For world bounds

```text
(min_x, max_x, min_y, max_y)
```

and a 64-by-64 image, define one common pixel scale:

\[
s = \min\left(
\frac{63}{\mathrm{max\_x}-\mathrm{min\_x}},
\frac{63}{\mathrm{max\_y}-\mathrm{min\_y}}
\right).
\]

The unused dimension is centred with a letterbox offset. Scale, offsets, point
coordinates, and radii use `np.rint` when converted to integer pixels, giving
NumPy's deterministic round-half-to-even rule. A world position maps to the
nearest pixel as:

\[
u = \operatorname{round}(o_x + (x-\mathrm{min\_x})s),
\]

\[
v = \operatorname{round}(o_y + (\mathrm{max\_y}-y)s).
\]

Image rows grow downward, so the world `y` axis is inverted. World-space radii
use the same scale `s`, keeping the obstacle and goal circular.

Drawing is clipped by the image canvas; terminal states near or slightly past a
boundary are not numerically clamped to a different physical position.

### Raster Layers

The renderer draws layers in this order:

1. letterbox/background and the world rectangle;
2. obstacle;
3. goal region;
4. car body;
5. heading marker.

The car retains circular collision geometry, but its visible body and heading
marker have a small minimum pixel size so orientation remains legible at
64-by-64 resolution. This visibility adjustment affects only pixels, never
physics or evaluation state.

The implementation uses NumPy plus Pillow rasterization, not Matplotlib. Pillow
becomes an explicit project dependency rather than an accidental transitive
dependency.

Schema version 1 fixes the semantic layers, coordinate transform, aspect ratio,
and output shape, but it does not claim that an independent renderer must
produce byte-identical pixels. The artifact records a `renderer_version` and
`pillow_version`. Repeated runs of the same renderer version and dependency
version must be byte-identical; a deliberate palette or raster-primitive change
increments `renderer_version`.

## Episode-Oriented Visual Dataset

### Why Frames Are Stored Once

A transition-oriented artifact containing both `observations` and
`next_observations` would duplicate nearly every frame. The visual artifact
instead stores each episode as one logical frame sequence:

\[
o_0, a_0, o_1, a_1, \ldots, a_{T-1}, o_T.
\]

An episode with `T` transitions therefore owns `T + 1` frames and `T` actions.
All episodes are flattened into contiguous arrays with offsets, avoiding Python
object arrays and pickle.

### Schema Version 1

```text
schema_version       []      int64, value 1
image_size           []      int64, value 64
context_frames       []      int64, value 4
renderer_version     []      fixed-width Unicode string, value pillow-raster-v1
pillow_version       []      fixed-width Unicode string
frames               [F,64,64,3] uint8
states               [F,4]   float64
actions              [N,2]   float64
rewards              [N]     float64
dones                [N]     bool
terminal_reasons     [N]     fixed-width Unicode string
episode_ids          [E]     int64
frame_offsets        [E+1]   int64
transition_offsets   [E+1]   int64
scene_world_bounds   [4]     float64
scene_obstacle       [2]     float64
scene_obstacle_radius []     float64
scene_goal           [2]     float64
scene_goal_radius    []      float64
scene_car_radius     []      float64
scene_dt             []      float64
```

Here:

- `E` is the number of episodes;
- `N` is the number of transitions in the source dataset;
- `F = N + E` is the number of frames;
- episode `i` owns frames
  `frames[frame_offsets[i]:frame_offsets[i+1]]`;
- episode `i` owns actions and transition metadata
  `actions[transition_offsets[i]:transition_offsets[i+1]]`;
- the frame slice length must equal the transition slice length plus one.

Within an episode, the first frame is rendered from the first `state`. Every
remaining frame is rendered from the ordered `next_states`. The aligned physical
state array is constructed in exactly the same way and is retained only for
diagnostics, not as a future visual-model input.

The converter records the default scene geometry because the current transition
dataset does not carry scene metadata. Schema version 1 therefore explicitly
supports only datasets produced by the current default-scene
`collect_transitions` workflow. It cannot retrospectively prove the scene used
by an externally modified dataset; custom-scene support requires a later source
schema that persists its own scene metadata.

### Causal Action Alignment

For every episode-local transition index `k`, the artifact guarantees:

```text
frames[k] --actions[k]--> frames[k + 1]
```

The action is the clipped, actually applied steering and acceleration already
stored by `collect_transitions`, not an unclipped requested command.

A future one-step visual training sample at episode time `t >= 3` is:

\[
(o_{t-3:t}, a_{t-3:t-1}), a_t \rightarrow o_{t+1}.
\]

A future horizon-`H` rollout sample uses the same initial four-frame and
three-action history, followed by the recorded future action sequence:

\[
(o_{t-3:t}, a_{t-3:t-1}), (a_t,\ldots,a_{t+H-1})
\rightarrow
(o_{t+1},\ldots,o_{t+H}).
\]

The later latent model may satisfy this contract by explicitly encoding the
recent action history or by updating a recurrent latent state with every
action. That choice belongs to the latent-dynamics design, not this data stage.

No context or rollout window may cross an episode boundary. The first three
frames of an episode do not form a training target; schema version 1 does not
pad them with repeated frames.

The existing episode-ID split function remains the split authority. A later
visual-model loader selects whole `episode_ids` before constructing temporal
windows, so no frame from one episode can leak across train, validation, and
test.

## Command-Line Workflow

Add one command:

```text
world-model-build-visual-data \
  --data data/transitions.npz \
  --output data/visual_episodes.npz \
  --preview artifacts/visual_episode_preview.gif
```

The command:

1. resolves the input, output, and preview paths and requires them to be
   pairwise distinct;
2. refuses to overwrite an existing output NPZ or preview GIF;
3. loads and validates the transition arrays;
4. reconstructs episodes in canonical episode-ID and step-ID order;
5. renders every physical frame once;
6. writes the compressed NPZ and deterministic preview GIF;
7. prints a sorted JSON summary.

The summary includes source and output paths, schema and renderer versions,
image size, context length, episode count, transition count, frame count,
four-frame-eligible episode count, available one-step visual sample count,
output bytes, and preview episode ID. An episode with `T` transitions contributes
`max(0, T - 3)` one-step samples and is eligible when `T >= 4`.

The preview contains every frame from the selected episode at a frame rate
derived from `scene_dt`. It does not add a speed overlay or mutate the stored
frames. The preview is a human inspection artifact and is not consumed during
training.

Pre-existing source files are never modified or removed. This learning-stage
CLI does not promise transactional publication across the NPZ and GIF; each
artifact is independently valid, and consumers validate the NPZ before use.

## Module Boundaries

The first implementation stage introduces focused modules rather than adding
visual responsibilities to the existing state-model files:

```text
visual_observation.py  world-to-pixel mapping and RGB rendering
visual_dataset.py      source validation, episode reconstruction, schema I/O
build_visual_data.py   CLI orchestration, JSON summary, and GIF preview
```

`CarEnv` remains the owner of physics. `visual_observation.py` may read public
scene properties but cannot compute transitions. `visual_dataset.py` owns data
alignment but cannot train a model. `build_visual_data.py` contains no rendering
math.

Future autoencoder and latent-dynamics modules consume the versioned visual
artifact through public loader functions; they must not depend on converter CLI
internals.

## Validation and Error Handling

Public functions reject:

- missing source arrays;
- non-finite numeric inputs;
- invalid state, action, or metadata shapes;
- duplicate, missing, negative, or non-contiguous episode step IDs;
- transition discontinuities within an episode;
- non-terminal final rows or terminal rows before an episode end;
- unsupported schema versions;
- invalid offsets or inconsistent frame/action counts;
- non-`uint8` frames or frames with a shape other than `[F,64,64,3]`;
- a missing or empty renderer-version field;
- an unavailable requested preview episode;
- input, output, or preview paths that are directories, non-distinct, or cannot
  be read or written as required.

Error messages name the failing array, episode, step, or invariant. The loader
uses `allow_pickle=False` for both transition and visual NPZ files.

## Testing Strategy

### Renderer Tests

- output has exact shape `[64,64,3]`, dtype `uint8`, and finite byte values;
- repeated rendering is byte-identical;
- the default obstacle, goal, boundary, car, and heading marker occupy expected
  regions;
- `x` increases rightward and `y` increases upward in world coordinates;
- heading rotation changes the orientation marker without moving the car;
- changing only velocity leaves a single frame unchanged;
- circles retain equal pixel radius along both axes within one raster pixel;
- terminal near-boundary states are clipped visually rather than physically
  moved;
- rendering leaves `state`, `trajectory`, `steps`, and `done` unchanged.

### Dataset Tests

- a synthetic multi-episode source becomes `N + E` frames and `N` actions;
- shuffled source rows still produce ascending episode IDs and step-ordered
  frames;
- offsets reconstruct every episode exactly and never overlap;
- each rendered frame corresponds to its aligned ground-truth state;
- `actions[k]` remains between the correct before and after frames;
- terminal metadata remains aligned to transitions;
- renderer version, Pillow version, and scene metadata round-trip;
- save/load preserves all arrays without pickle;
- malformed continuity, step IDs, terminal placement, and offsets fail with
  focused messages;
- short episodes remain in the artifact while eligible-episode and one-step
  sample counts remain exact;
- the same source produces identical arrays and metadata.

### CLI Tests

- `--help` documents the source, output, preview, and preview-episode options;
- a tiny end-to-end conversion writes a valid NPZ and non-empty GIF;
- missing input and invalid preview episode produce argument errors;
- the JSON summary contains the documented finite, JSON-safe fields.

The full existing test suite must continue to pass.

## Acceptance Criteria

This stage is complete when:

1. an unchanged state transition dataset can be converted deterministically to
   a validated schema-version-1 visual episode artifact;
2. every episode satisfies the `T + 1` frames and `T` causally aligned actions
   invariant;
3. a preview GIF visibly shows continuous car motion, orientation, obstacle,
   goal, and boundaries without a velocity overlay;
4. individual frames do not expose velocity, while every reported eligible
   episode contains at least one four-frame-history-plus-target sample and short
   episodes remain present without being misreported as eligible;
5. no existing state-model behavior changes;
6. focused tests and the complete repository test suite pass.

## Follow-On Stages

After this design is implemented and accepted, subsequent designs proceed in
this order:

1. train a small frame autoencoder or VAE and verify reconstruction quality;
2. encode four consecutive frames plus recent aligned actions and train
   action-conditioned dynamics with recurrent latent memory or an equivalent
   explicit action-history state;
3. perform visual free rollout by recursively updating that latent state with
   predicted latents and recorded actions;
4. add counterfactual left-versus-right action controllability tests;
5. map the local encoder, latent dynamics, action conditioning, decoder, and
   rollout interfaces to remote-CUDA WorldGym inference;
6. only then introduce a policy and reinforcement learning in imagined
   rollouts.

These are separate design and implementation stages. They are intentionally not
part of the visual-observation bridge implementation.
