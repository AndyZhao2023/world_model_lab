# State-Delta World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and smoke-test a PyTorch MLP that predicts one-step `CarEnv` state deltas from the collected NPZ transitions.

**Architecture:** Keep deterministic NumPy preprocessing separate from the PyTorch model and training CLI. Split by episode before computing train-only normalization statistics, then report errors after converting predictions back to physical units.

**Tech Stack:** Python 3.12, NumPy, PyTorch, `unittest`

## Global Constraints

- Model input is `[x, y, sin(heading), cos(heading), velocity, steering, acceleration]`.
- Model target is `[delta_x, delta_y, wrapped_delta_heading, delta_velocity]`.
- Split ratios are 80/10/10 by unique episode ID with seed-controlled shuffling.
- Normalization statistics come only from the training split.
- Tests use small synthetic data and must not depend on the generated real dataset.

---

### Task 1: Deterministic dataset preprocessing

**Files:**
- Create: `src/world_model_lab/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `wrap_angle(values)`, `split_episode_ids(episode_ids, seed, ratios)`, `build_model_arrays(states, actions, next_states)`, `fit_normalizer(values)`.

- [ ] Write tests proving angle deltas wrap across `-pi/pi`, all transitions from one episode stay together, splitting is deterministic, and model arrays have shapes `[N, 7]` and `[N, 4]`.
- [ ] Run `.venv/bin/python -m unittest tests.test_dataset -v`; expect import failure because `world_model_lab.dataset` does not exist.
- [ ] Implement input/target construction, split validation, and train-only mean/std helpers.
- [ ] Re-run `.venv/bin/python -m unittest tests.test_dataset -v`; expect all dataset tests to pass.

### Task 2: MLP and training behavior

**Files:**
- Create: `src/world_model_lab/model.py`
- Create: `src/world_model_lab/train_world_model.py`
- Test: `tests/test_train_world_model.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: dataset preprocessing functions from Task 1.
- Produces: `WorldModelMLP`, `train_model(...)`, `evaluate_model(...)`, and CLI `main()`.

- [ ] Add PyTorch as a project dependency and install the project extras in `.venv`.
- [ ] Write a synthetic deterministic dynamics test that requires training loss to decrease and a checkpoint round-trip test that validates required metadata.
- [ ] Run `.venv/bin/python -m unittest tests.test_train_world_model -v`; expect import failure because the training module does not exist.
- [ ] Implement a seeded CPU training loop using Adam and MSE, physical-unit MAE evaluation, and checkpoint persistence.
- [ ] Re-run `.venv/bin/python -m unittest tests.test_train_world_model -v`; expect all training tests to pass.

### Task 3: CLI and documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/test_train_world_model.py`

**Interfaces:**
- Consumes: `train_world_model.main()`.
- Produces: `world-model-train` command and documented invocation.

- [ ] Add a CLI smoke test using a temporary NPZ dataset and a temporary checkpoint path.
- [ ] Run the smoke test and verify it fails before the console entry point/documentation behavior exists.
- [ ] Add the console script and README commands describing training output and checkpoint contents.
- [ ] Run the full unit suite.
- [ ] Run `MPLBACKEND=Agg .venv/bin/python -m world_model_lab.train_world_model --data data/transitions.npz --epochs 40 --output artifacts/world_model.pt` and verify finite validation/test metrics and a saved checkpoint.
