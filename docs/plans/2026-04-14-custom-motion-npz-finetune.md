# Custom Motion NPZ Finetune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated NPZ-based G1 dataset path so Kimodo can phase-1 finetune on the 192 annotated `custom_motion` clips.

**Architecture:** Introduce a new MuJoCo NPZ loader plus a sibling dataset class instead of expanding the CSV loader. Reuse existing timeline sampling, feature encoding, and training loop plumbing while adding explicit resampling from 50 FPS to 30 FPS.

**Tech Stack:** Python, NumPy, PyTorch, SciPy rotations, pytest, OmegaConf/Hydra-style config dicts.

---

### Task 1: Add loader-focused failing tests

**Files:**
- Create: `tests/test_custom_motion_npz_loader.py`
- Modify: `kimodo/training/__init__.py`
- Test: `tests/test_custom_motion_npz_loader.py`

**Step 1: Write the failing test**

Add tests that assert:

- the loader reads one `custom_motion` NPZ and returns `root_positions` with shape `[T, 3]`
- the loader returns `local_joint_rots` with shape `[T, 34, 3, 3]`
- the loader ignores the object slice in `qpos[:, 36:43]`
- the loader metadata reports `input_fps == 50`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_custom_motion_npz_loader.py -q`

Expected: failure because `load_g1_npz_motion` does not exist yet.

**Step 3: Write minimal implementation**

Create the NPZ loader with just enough conversion logic to satisfy the tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_custom_motion_npz_loader.py -q`

Expected: pass.

### Task 2: Add annotated-subset dataset failing tests

**Files:**
- Create: `tests/test_custom_motion_dataset.py`
- Modify: `kimodo/training/dataset.py`
- Test: `tests/test_custom_motion_dataset.py`

**Step 1: Write the failing test**

Add tests that assert:

- the NPZ dataset keeps only stems present in `timeline_sub10.jsonl`
- the dataset length equals the matched annotation count
- the dataset returns text from the timeline and motion tensors padded to `max_frames`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_custom_motion_dataset.py -q`

Expected: failure because the NPZ dataset class does not exist yet.

**Step 3: Write minimal implementation**

Implement `G1NPZTextDataset` by reusing the existing text sampling and feature preparation flow.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_custom_motion_dataset.py -q`

Expected: pass.

### Task 3: Add resampling failing tests

**Files:**
- Modify: `tests/test_custom_motion_npz_loader.py`
- Modify: `kimodo/training/custom_motion_npz.py`
- Test: `tests/test_custom_motion_npz_loader.py`

**Step 1: Write the failing test**

Add a test asserting that 50 FPS input resamples to a 30 FPS-equivalent sequence length using time-based interpolation.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_custom_motion_npz_loader.py -q`

Expected: failure because only raw loading is implemented.

**Step 3: Write minimal implementation**

Add timestamp-based resampling helpers for positions and rotations.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_custom_motion_npz_loader.py -q`

Expected: pass.

### Task 4: Add training config wiring failing tests

**Files:**
- Modify: `tests/test_custom_motion_dataset.py`
- Modify: `kimodo/training/train.py`
- Create: `kimodo/training/train_config_phase1_custom_motion.yaml`
- Test: `tests/test_custom_motion_dataset.py`

**Step 1: Write the failing test**

Add a test asserting that `build_dataset(...)` can instantiate the new dataset from the custom config.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_custom_motion_dataset.py -q`

Expected: failure because training config and dataset selection are not wired yet.

**Step 3: Write minimal implementation**

Allow `build_dataset(...)` to instantiate either CSV or NPZ dataset based on config shape or explicit `_target_`, then add the custom phase-1 config.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_custom_motion_dataset.py -q`

Expected: pass.

### Task 5: End-to-end verification

**Files:**
- Verify only

**Step 1: Run targeted tests**

Run:

```bash
pytest tests/test_custom_motion_npz_loader.py tests/test_custom_motion_dataset.py -q
```

Expected: all pass.

**Step 2: Run syntax verification on touched training files**

Run:

```bash
python -m py_compile kimodo/training/custom_motion_npz.py kimodo/training/dataset.py kimodo/training/train.py
```

Expected: no output.

**Step 3: Smoke-check config resolution**

Run:

```bash
python - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load('kimodo/training/train_config_phase1_custom_motion.yaml')
print(cfg.data.dataset)
PY
```

Expected: config prints with the NPZ dataset fields.
