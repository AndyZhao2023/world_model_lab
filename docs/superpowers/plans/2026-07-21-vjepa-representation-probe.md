# Frozen V-JEPA Representation Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the frozen `facebook/vjepa2-vitl-fpc64-256` encoder preserves enough state and temporal information from four rendered car frames to justify building an action-conditioned latent predictor.

**Architecture:** Reuse the existing episode split and four-frame visual-window contract. A lazy Hugging Face adapter freezes V-JEPA 2, extracts `[B, 512, 1024]` encoder tokens, and pools the first and last tubelets into a 2048-value feature `[last, last-first]`. A deterministic ridge probe predicts `[x, y, sin(heading), cos(heading), velocity]`; held-out recorded, reversed, and repeat-last clips measure state readability and temporal sensitivity.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, optional `transformers>=5.14,<6` and `torchvision>=0.21,<0.29`, unittest, existing schema-v1 visual artifacts.

## Global Constraints

- Keep `facebook/vjepa2-vitl-fpc64-256` frozen; no encoder gradients or EMA updates.
- Keep the existing episode-level split with `split_seed=42` and never split frames independently.
- Use exactly four context frames and the state aligned with the final recorded context frame.
- Do not train an action predictor, decoder, or MPC policy in this phase.
- Load `transformers` lazily so the base package and unit tests still work without the optional dependency.
- Treat generated features, model weights, and result JSON as ignored local artifacts.
- Pre-register pilot gates on original 64x64 coordinates: centre mean error `<=3 px`, heading mean error `<45 deg`, and recorded velocity MAE at least `5%` better than both temporal ablations.

---

### Task 1: Add clip and target contracts

**Files:**
- Create: `src/world_model_lab/vjepa_probe.py`
- Create: `tests/test_vjepa_probe.py`

**Interfaces:**
- Consumes: schema-v1 visual dataset and `VisualWindowIndex`.
- Produces: `ProbeClipBatch`, `build_probe_clip_batch`, `select_evenly_spaced_positions`, and `state_probe_metrics`.

- [ ] **Step 1: Write failing alignment and validation tests**

```python
batch = build_probe_clip_batch(visual, index, np.asarray([0, 2]), order="recorded")
self.assertEqual(batch.frames.shape, (2, 4, 64, 64, 3))
np.testing.assert_array_equal(batch.states[:, :2], expected_last_context_xy)

reversed_batch = build_probe_clip_batch(visual, index, np.asarray([0]), order="reversed")
np.testing.assert_array_equal(reversed_batch.frames[0], batch.frames[0, ::-1])
np.testing.assert_array_equal(reversed_batch.states, batch.states[:1])
```

- [ ] **Step 2: Run the focused test and confirm it fails because the module is absent**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_vjepa_probe -v
```

- [ ] **Step 3: Implement the immutable clip batch and exact frame/state offsets**

```python
@dataclass(frozen=True)
class ProbeClipBatch:
    frames: np.ndarray
    states: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray

def build_probe_clip_batch(
    dataset: Mapping[str, np.ndarray],
    index: VisualWindowIndex,
    positions: np.ndarray,
    *,
    order: str,
) -> ProbeClipBatch:
    selected = validated_probe_positions(positions, count=index.count)
    frame_offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
    frames = []
    states = []
    episode_ids = []
    for position in selected.tolist():
        episode_index = int(index.episode_indices[position])
        step_id = int(index.step_ids[position])
        frame_start = int(frame_offsets[episode_index]) + step_id - 3
        clip = np.asarray(dataset["frames"])[frame_start : frame_start + 4]
        frames.append(apply_clip_order(clip, order=order))
        states.append(np.asarray(dataset["states"])[frame_start + 3])
        episode_ids.append(np.asarray(dataset["episode_ids"])[episode_index])
    return ProbeClipBatch(
        frames=np.stack(frames),
        states=np.stack(states),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        step_ids=index.step_ids[selected],
    )
```

- [ ] **Step 4: Add deterministic even-spacing, state target conversion, circular-heading and pixel-centre metrics**

```python
def state_to_probe_targets(states: np.ndarray) -> np.ndarray:
    return np.column_stack((states[:, :2], np.sin(states[:, 2]), np.cos(states[:, 2]), states[:, 3]))
```

- [ ] **Step 5: Re-run the focused tests and commit**

```bash
git add src/world_model_lab/vjepa_probe.py tests/test_vjepa_probe.py
git commit -m "feat: add V-JEPA probe data contracts"
```

### Task 2: Add the frozen Hugging Face adapter and pooling contract

**Files:**
- Modify: `src/world_model_lab/vjepa_probe.py`
- Modify: `tests/test_vjepa_probe.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: uint8 clips `[B,4,64,64,3]`.
- Produces: finite float32 features `[B,2048]` and encoder metadata.

- [ ] **Step 1: Add failing tests with fake processor/model objects**

```python
encoder = FrozenVJEPAEncoder(processor=fake_processor, model=fake_model, device="cpu")
features = encoder.encode(clips)
self.assertEqual(features.shape, (2, 2048))
self.assertTrue(all(not p.requires_grad for p in fake_model.parameters()))
self.assertFalse(fake_model.training)
self.assertTrue(fake_model.skip_predictor_seen)
```

- [ ] **Step 2: Add an exact token-pooling test**

```python
tokens = torch.arange(2 * 512 * 4, dtype=torch.float32).reshape(2, 512, 4)
pooled = pool_vjepa_tokens(tokens, tubelet_count=2, spatial_tokens=256)
self.assertEqual(tuple(pooled.shape), (2, 8))
```

- [ ] **Step 3: Implement lazy loading and predictor-free feature extraction**

```python
@classmethod
def from_pretrained(cls, model_id: str, *, revision: str, device: str):
    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as error:
        raise RuntimeError("install world-model-lab[vjepa]") from error
    processor = AutoVideoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision)
    return cls(processor=processor, model=model, device=device)
```

```python
with torch.inference_mode():
    tokens = self.model.get_vision_features(pixel_values)
return pool_vjepa_tokens(tokens, tubelet_count=2, spatial_tokens=256)
```

- [ ] **Step 4: Add the optional dependency without changing the base install**

```toml
[project.optional-dependencies]
vjepa = ["transformers>=5.14,<6", "torchvision>=0.21,<0.29"]
```

- [ ] **Step 5: Run focused and base import tests, then commit**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_vjepa_probe tests.test_visual_windows -v
git add pyproject.toml src/world_model_lab/vjepa_probe.py tests/test_vjepa_probe.py
git commit -m "feat: add frozen V-JEPA encoder adapter"
```

### Task 3: Add deterministic ridge probe fitting and decisions

**Files:**
- Modify: `src/world_model_lab/vjepa_probe.py`
- Modify: `tests/test_vjepa_probe.py`

**Interfaces:**
- Consumes: train features/targets and held-out feature variants.
- Produces: `LinearStateProbe`, per-variant metrics, baselines, and gate decisions.

- [ ] **Step 1: Add failing exact-fit, dual/primal, and invalid-input tests**

```python
probe = fit_linear_state_probe(features, targets, ridge=1e-3)
np.testing.assert_allclose(probe.predict(features), targets, atol=1e-4)
self.assertEqual(probe.feature_dim, features.shape[1])
```

- [ ] **Step 2: Implement centered ridge with primal/dual selection**

```python
if sample_count <= feature_dim:
    weights = x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(sample_count), y)
else:
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(feature_dim), x.T @ y)
bias = target_mean - feature_mean @ weights
```

- [ ] **Step 3: Add mean-target baseline and strict gates**

```python
gates = {
    "centre_mean_le_3px": recorded["mean_centre_error_pixels"] <= 3.0,
    "heading_mean_lt_45deg": recorded["mean_heading_error_degrees"] < 45.0,
    "velocity_beats_reversed_5pct": recorded_velocity <= 0.95 * reversed_velocity,
    "velocity_beats_repeat_last_5pct": recorded_velocity <= 0.95 * repeat_velocity,
}
```

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_vjepa_probe -v
git add src/world_model_lab/vjepa_probe.py tests/test_vjepa_probe.py
git commit -m "feat: add frozen representation probe metrics"
```

### Task 4: Add the no-clobber CLI and feature cache

**Files:**
- Create: `src/world_model_lab/run_vjepa_probe.py`
- Create: `tests/test_run_vjepa_probe.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: visual NPZ, model ID/revision, sample caps, batch size, ridge.
- Produces: atomic feature NPZ and JSON result with dataset/model digests, runtime, metrics, and decisions.

- [ ] **Step 1: Add failing integration tests using an injected fake encoder factory**

```python
summary = run_vjepa_probe(
    data_path=data_path,
    feature_path=feature_path,
    result_path=result_path,
    encoder_factory=fake_encoder_factory,
    max_train=8,
    max_validation=4,
    max_test=4,
)
self.assertTrue(feature_path.is_file())
self.assertTrue(result_path.is_file())
self.assertIn("recorded", summary["test"])
```

- [ ] **Step 2: Add failure tests for existing outputs, split leakage, malformed cache, and missing optional dependency**

```python
feature_path.touch()
with self.assertRaises(FileExistsError):
    run_vjepa_probe(
        data_path=data_path,
        feature_path=feature_path,
        result_path=result_path,
        model_id=MODEL_ID,
        model_revision="main",
        batch_size=1,
        max_train=8,
        max_validation=4,
        max_test=4,
        ridge=1e-3,
        split_seed=42,
        encoder_factory=fake_factory,
    )

with mock.patch.dict("sys.modules", {"transformers": None}):
    with self.assertRaisesRegex(RuntimeError, r"world-model-lab\[vjepa\]"):
        FrozenVJEPAEncoder.from_pretrained(MODEL_ID, revision="main", device="cpu")

with np.load(feature_path, allow_pickle=False) as cache:
    self.assertTrue(set(train_episode_ids).isdisjoint(test_episode_ids))
    self.assertEqual(int(cache["schema_version"]), 1)
```

- [ ] **Step 3: Implement extraction in deterministic split/order sequence**

```text
train/recorded
validation/recorded
test/recorded
test/reversed
test/repeat_last
```

```python
for split_name, order in extraction_sequence:
    positions = select_evenly_spaced_positions(
        indexes[split_name].count,
        limit=sample_limits[split_name],
    )
    clips = build_probe_clip_batch(
        dataset,
        indexes[split_name],
        positions,
        order=order,
    )
    arrays[f"{split_name}_{order}_features"] = encode_in_batches(
        encoder,
        clips.frames,
        batch_size=batch_size,
    )
```

- [ ] **Step 4: Publish NPZ and JSON through same-directory temporary files without overwrite**

```python
write_new_file_atomically(
    feature_path,
    writer=lambda handle: np.savez_compressed(handle, **arrays),
    exists_message=f"feature cache already exists: {feature_path}",
)
payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode()
write_new_file_atomically(
    result_path,
    writer=lambda handle: handle.write(payload),
    exists_message=f"result already exists: {result_path}",
)
```

- [ ] **Step 5: Register and document the command**

```toml
world-model-vjepa-probe = "world_model_lab.run_vjepa_probe:main"
```

```bash
world-model-vjepa-probe \
  --data data/visual_episodes.npz \
  --features artifacts/vjepa2_probe_features.npz \
  --result artifacts/vjepa2_probe_result.json \
  --model facebook/vjepa2-vitl-fpc64-256 \
  --revision main --batch-size 1 \
  --max-train 128 --max-validation 32 --max-test 32
```

- [ ] **Step 6: Run integration tests and commit**

```bash
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest tests.test_run_vjepa_probe -v
git add README.md pyproject.toml src/world_model_lab/run_vjepa_probe.py tests/test_run_vjepa_probe.py
git commit -m "feat: add V-JEPA feasibility runner"
```

### Task 5: Run environment and checkpoint feasibility gates

**Files:**
- Create: `docs/experiments/2026-07-21-vjepa-representation-probe.md`
- Generated and ignored: `data/visual_episodes.npz`
- Generated and ignored: `artifacts/vjepa2_probe_features.npz`
- Generated and ignored: `artifacts/vjepa2_probe_result.json`

**Interfaces:**
- Consumes: the merged visual dataset protocol and public checkpoint.
- Produces: measured CPU memory/runtime and a pass/reject decision for P2.

- [ ] **Step 1: Rebuild and verify the deterministic visual artifact**

```bash
cp /Users/andyzhao/Workspace/world_model_lab/data/transitions.npz data/transitions.npz
PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.build_visual_data \
  --data data/transitions.npz --output data/visual_episodes.npz \
  --preview artifacts/visual_dataset_preview.gif
shasum -a 256 data/visual_episodes.npz
```

Expected registered SHA-256: `2f260fb55c1e90f5d4e836050c9ddfd1e91fa6dab78bb189e2d6bd8736207aba`.

- [ ] **Step 2: Install the optional dependency into the existing project virtual environment**

```bash
/Users/andyzhao/Workspace/world_model_lab/.venv/bin/python -m pip install \
  "transformers>=5.14,<6"
```

- [ ] **Step 3: Download only the Transformers checkpoint files and verify the cache**

```bash
/Users/andyzhao/Workspace/world_model_lab/.venv/bin/hf download \
  facebook/vjepa2-vitl-fpc64-256 \
  --include config.json video_preprocessor_config.json model.safetensors
```

- [ ] **Step 4: Run one-clip and eight-clip CPU timing/memory smoke tests before the registered pilot**

```bash
/usr/bin/time -l env PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.run_vjepa_probe \
  --data data/visual_episodes.npz \
  --features artifacts/vjepa2_probe_smoke_1.npz \
  --result artifacts/vjepa2_probe_smoke_1.json \
  --batch-size 1 --max-train 1 --max-validation 1 --max-test 1

/usr/bin/time -l env PYTHONPATH=src \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.run_vjepa_probe \
  --data data/visual_episodes.npz \
  --features artifacts/vjepa2_probe_smoke_8.npz \
  --result artifacts/vjepa2_probe_smoke_8.json \
  --batch-size 1 --max-train 8 --max-validation 2 --max-test 2
```

- [ ] **Step 5: Abort the pilot if one clip exceeds 60 seconds or peak resident memory exceeds 12 GiB; otherwise run the capped 128/32/32 pilot once**

```bash
env PYTHONPATH=src /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m world_model_lab.run_vjepa_probe \
  --data data/visual_episodes.npz \
  --features artifacts/vjepa2_probe_features.npz \
  --result artifacts/vjepa2_probe_result.json \
  --batch-size 1 --max-train 128 --max-validation 32 --max-test 32
```

- [ ] **Step 6: Record exact environment, checkpoint revision, data hash, sample IDs, metrics, gates, and decision in the experiment document**

```markdown
## Registered protocol
- Dataset SHA-256:
- Model revision:
- Split seed and selected episode/window IDs:
- CPU, accelerator availability, peak RSS, seconds per clip:

## Results
- Mean-baseline metrics:
- Recorded metrics:
- Reversed metrics:
- Repeat-last metrics:
- Gate table and pass/reject decision:
```

- [ ] **Step 7: Run the complete test suite and commit the experiment record**

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  /Users/andyzhao/Workspace/world_model_lab/.venv/bin/python \
  -m unittest discover -s tests -v
git add docs/experiments/2026-07-21-vjepa-representation-probe.md
git commit -m "docs: record frozen V-JEPA feasibility probe"
```
