# Visual Object-Slot Global Affine Locator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed the train-only full-latent ridge centre probe as a frozen raw-latent affine locator while retaining the 11x11 local writer, then decide one candidate with six preregistered gates.

**Architecture:** Add a `global_affine` object-slot locator alongside the existing `spatial_attention` mode. The global mode uses a frozen `Linear(512, 2)` centre head initialized from converted ridge coefficients and a trainable `512 -> 64 -> 2` heading MLP; both feed the unchanged four-value slot and local patch compositor.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, unittest, existing world-model-lab checkpoint and matched autoencoder diagnostics.

## Global Constraints

- Use `data/visual_episodes.npz` with SHA-256 `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`.
- Use source `artifacts/visual_latent_spatial8_objective_w01.pt` with SHA-256 `5022fe5f11c099550f6971f6f92362505126beb70883b415f1a2ad338e803369`.
- Freeze source encoder, base decoder, dynamics, normalizers, split IDs, and converted centre locator.
- Keep patch size `11`, hidden size `64`, initial alpha `0.01`, 20 epochs, batch size `128`, learning rate `0.001`, seed `0`, and ridge `0.001`.
- Train exactly one candidate; do not run H5 unless all six representation gates pass.
- Do not stage `.DS_Store`, `data/`, or generated `artifacts/`.

---

### Task 1: Add locator mode and checkpoint compatibility

**Files:**

- Modify: `src/world_model_lab/visual_latent_model.py`
- Modify: `src/world_model_lab/train_visual_latent_model.py`
- Modify: `tests/test_visual_object_slot.py`
- Modify: `tests/test_train_visual_latent_model.py`

**Interfaces:**

- Consumes: existing `SpatialConvAutoencoder(..., object_slot_decoder=True)`.
- Produces: constructor option `object_slot_locator: str = "spatial_attention"` and property with values `spatial_attention` or `global_affine`.
- Produces in global mode: `object_center: nn.Linear`, frozen parameters, and `object_heading: nn.Sequential`.

- [ ] **Step 1: Write failing model tests**

Add tests creating:

```python
model = SpatialConvAutoencoder(
    latent_channels=2,
    base_channels=2,
    object_slot_decoder=True,
    object_slot_patch_size=11,
    object_slot_hidden_size=8,
    object_slot_locator="global_affine",
)
```

Assert the centre head consumes `model.latent_dim`, its parameters are frozen,
heading parameters are trainable, and `decode_object_slot_components` returns
finite `[B, 4]` slots plus exactly local alpha.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_visual_object_slot.ObjectSlotGeometryTest \
  tests.test_train_visual_latent_model.VisualLatentCheckpointTest -v
```

Expected: FAIL because `object_slot_locator` is not accepted or persisted.

- [ ] **Step 3: Implement locator construction and prediction**

For `spatial_attention`, preserve the current 1x1 attention and 8-D heading
path exactly. For `global_affine`, construct:

```python
self.object_center = nn.Linear(self.latent_dim, 2)
self.object_heading = nn.Sequential(
    nn.Linear(self.latent_dim, self.object_slot_hidden_size),
    nn.ReLU(),
    nn.Linear(self.object_slot_hidden_size, 2),
)
for parameter in self.object_center.parameters():
    parameter.requires_grad_(False)
```

In `_predict_object_slot`, flatten `[B, C, 8, 8]`, apply the centre and heading
heads, L2-normalize heading, and concatenate four values.

- [ ] **Step 4: Persist the mode**

Add `object_slot_locator` to checkpoint `model_config`. Missing keys on legacy
slot checkpoints must load as `spatial_attention`; non-slot checkpoints must
store an empty locator string and reject a non-empty one.

- [ ] **Step 5: Run focused tests**

Run the command from Step 2. Expected: PASS, including existing residual,
legacy spatial, and first slot checkpoint round trips.

### Task 2: Convert and freeze the train-only affine centre probe

**Files:**

- Modify: `src/world_model_lab/visual_object_slot.py`
- Modify: `src/world_model_lab/train_visual_object_slot.py`
- Modify: `tests/test_visual_object_slot.py`

**Interfaces:**

- Produces:

```python
def normalized_affine_to_raw(
    normalized_weight: np.ndarray,
    normalized_bias: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]
```

- Extends:

```python
initialize_object_slot_autoencoder(
    source,
    *,
    locator: str,
    centre_weight: np.ndarray | None,
    centre_bias: np.ndarray | None,
    patch_size: int,
    hidden_size: int,
    initial_alpha: float,
    seed: int,
)
```

- [ ] **Step 1: Write failing conversion tests**

Use finite toy arrays and assert:

```text
Wn * ((z - mean) / std) + bn
==
Wr * z + br
```

to `rtol=0`, `atol=1e-6`. Reject wrong shapes, non-finite values, and
non-positive standard deviations.

- [ ] **Step 2: Write failing freeze and gradient tests**

Initialize a global candidate with known centre coefficients. Assert exact
weight/bias copies, no centre gradients, gradients in heading/patch
parameters, and bit-identical source encoder/base decoder tensors.

- [ ] **Step 3: Verify failures**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_visual_object_slot -v
```

Expected: FAIL because conversion and global initialization are absent.

- [ ] **Step 4: Implement minimal conversion and initialization**

Use:

```python
raw_weight = normalized_weight / std[None, :]
raw_bias = normalized_bias - normalized_weight @ (mean / std)
```

Require centre arrays only for `global_affine`; reject them for
`spatial_attention`.

- [ ] **Step 5: Run focused tests**

Run the command from Step 3. Expected: PASS.

### Task 3: Generalize the deterministic trainer and state gate

**Files:**

- Modify: `src/world_model_lab/visual_object_slot.py`
- Modify: `src/world_model_lab/train_visual_object_slot.py`
- Modify: `tests/test_train_visual_object_slot.py`

**Interfaces:**

- Extends `train_object_slot_decoder` with:

```python
locator: str = "spatial_attention"
centre_weight: np.ndarray | None = None
centre_bias: np.ndarray | None = None
```

- Extends `run_visual_object_slot_training` and CLI with:

```text
--locator {spatial_attention,global_affine}
```

- [ ] **Step 1: Write failing runner tests**

Run a one-epoch toy global-affine candidate. Assert:

- checkpoint mode is `global_affine`;
- loaded centre tensors equal fitted converted tensors;
- centre parameters are frozen;
- source encoder/base/dynamics remain exact;
- summary centre gate uses `<= 1.05 * source_probe`;
- heading gate remains strict `< source_probe`;
- CLI help lists `--locator`.

- [ ] **Step 2: Verify the runner tests fail**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_train_visual_object_slot -v
```

Expected: FAIL because the runner supports only spatial attention.

- [ ] **Step 3: Return probe coefficients internally**

Refactor the existing deterministic ridge fit so it retains its float64
solution for initialization while keeping the JSON summary numeric and
serializable. Convert only the two centre rows into raw-latent coefficients.

- [ ] **Step 4: Train only permitted parameters**

Pass locator and centre arrays through the trainer. Build the Adam parameter
list from `requires_grad=True`; assert centre tensors remain bit-identical
after training and checkpoint publication.

- [ ] **Step 5: Implement mode-specific centre gate**

For `global_affine`, emit:

```text
name      held_out_centre_error_stability
operator  <=
limit     1.05 * source test mean centre error
```

Retain strict centre improvement for the legacy spatial-attention mode.

- [ ] **Step 6: Run focused and compatibility tests**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest \
  tests.test_visual_object_slot \
  tests.test_train_visual_object_slot \
  tests.test_visual_object_residual \
  tests.test_train_visual_object_residual \
  tests.test_train_visual_latent_model.VisualLatentCheckpointTest -v
```

Expected: PASS.

### Task 4: Train and diagnose the single registered candidate

**Files:**

- Modify: `docs/experiments/2026-07-20-visual-object-slot-affine-locator.md`
- Generated and ignored: `artifacts/visual_latent_spatial8_object_slot_affine.pt`
- Generated and ignored: `artifacts/visual_latent_spatial8_object_slot_affine_predictions.png`
- Generated and ignored: `artifacts/diagnostics/visual-object-slot-affine/`

**Interfaces:**

- Consumes the CLI and checkpoint mode from Tasks 1–3.
- Produces exact training, state, image, gate, and artifact-hash evidence.

- [ ] **Step 1: Verify registered input hashes and empty outputs**

Run `shasum -a 256` on data/source and abort if either differs from Global
Constraints. Require all registered output paths to be absent.

- [ ] **Step 2: Train exactly once**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_object_slot \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --output artifacts/visual_latent_spatial8_object_slot_affine.pt \
  --preview artifacts/visual_latent_spatial8_object_slot_affine_predictions.png \
  --locator global_affine \
  --patch-size 11 --hidden-size 64 --initial-alpha 0.01 \
  --epochs 20 --batch-size 128 --learning-rate 0.001 \
  --foreground-loss-weight 1.0 --mask-loss-weight 0.01 \
  --centre-loss-weight 1.0 --heading-loss-weight 0.1 \
  --source-probe-ridge 0.001
```

- [ ] **Step 3: Run the matched diagnostic**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.diagnose_visual_autoencoder \
  --data data/visual_episodes.npz \
  --source-checkpoint artifacts/visual_latent_spatial8_objective_w01.pt \
  --candidate-checkpoint artifacts/visual_latent_spatial8_object_slot_affine.pt \
  --output-dir artifacts/diagnostics/visual-object-slot-affine
```

- [ ] **Step 4: Apply all six gates**

Combine four diagnostic gates with centre stability and heading improvement.
If any fail, do not create an H5 checkpoint. Update the experiment document
with exact results, hashes, and strict decision.

### Task 5: Verify and commit

**Files:**

- All modified source, tests, plans, and experiment docs.

**Interfaces:**

- Produces one reviewed local commit; no push.

- [ ] **Step 1: Run full verification**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v

PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m compileall -q src tests

PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.train_visual_object_slot --help

git diff --check
```

Expected: all tests pass, compile/CLI exit zero, and no whitespace errors.

- [ ] **Step 2: Inspect and commit scope**

Stage only source, tests, `pyproject.toml` if changed, and both new docs. Keep
`.DS_Store`, data, and generated artifacts untracked or ignored.

Commit:

```bash
git commit -m "feat: add global affine object slot experiment"
```

## Self-Review

- Spec coverage: locator conversion, frozen centre, global heading MLP,
  checkpoint compatibility, one-candidate training, six gates, H5 stop, and
  final verification each have an explicit task.
- Placeholder scan: no deferred implementation or unspecified test step
  remains.
- Type consistency: `object_slot_locator`, `locator`, `centre_weight`, and
  `centre_bias` names are consistent across model, trainer, runner, tests, and
  CLI.
- Scope: the local writer and all source-world-model invariants remain
  unchanged; Transformer work is explicitly excluded.
