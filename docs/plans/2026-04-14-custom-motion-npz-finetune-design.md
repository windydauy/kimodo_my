# Custom Motion NPZ Finetune Design

**Date:** 2026-04-14

**Scope:** Adapt Kimodo phase-1 finetuning to the annotated subset of `custom_motion` stored as MuJoCo-style `.npz` trajectories.

## Goal

Enable phase-1 fine-tuning on the 192 annotated `custom_motion` clips without disturbing the existing CSV-based G1 finetuning path.

## Dataset Facts Confirmed

- Training source files live under `custom_motion/robot-object/*.npz`.
- The annotated subset is defined by `custom_motion/timeline_sub10.jsonl`.
- There are `1826` NPZ clips total and `192` timeline records. We will train on the intersection only.
- Each inspected clip stores:
  - `fps`: `50.0`
  - `qpos`: shape `[T, 43]`
  - `body_pos_w`, `body_quat_w`, object state, and contact metadata
- `qpos[:, :36]` matches the expected G1 MuJoCo state layout:
  - root translation `xyz` in columns `0:3`
  - root quaternion in columns `3:7`
  - 29 G1 joint DoFs in columns `7:36`
- `qpos[:, 36:43]` behaves like object `xyz + quat` and will not be consumed by the phase-1 motion encoder.

## Coordinate And Unit Assessment

- The stored translations are ordered as `xyz`.
- The values are consistent with MuJoCo world coordinates:
  - root `z` is the vertical axis and is typically around `0.4` to `0.8`
  - `body_pos_w[..., 2]` is near-ground for feet and larger for torso/root
- The translation scale appears to be meters, not centimeters:
  - root and object positions are in the `0.x` to low `1.x` range
  - this is consistent with MuJoCo simulation state rather than exported centimeter CSVs
- Root orientation is stored as quaternions, not Euler angles, so CSV-only settings such as `root_euler_order: xyz` do not apply to the raw NPZ input.

## Design

We will add a dedicated NPZ loading path instead of overloading the CSV loader. The current dataset stack is tightly coupled to `*.csv` discovery and `load_g1_csv_motion(...)`; forcing NPZ support into that path would blend two incompatible input formats and make future debugging harder.

The new path will:

1. Load MuJoCo-style NPZ files from `custom_motion/robot-object`.
2. Filter to the 192 clips present in `timeline_sub10.jsonl`.
3. Convert robot-only `qpos[:, :36]` into Kimodo training tensors:
   - `root_positions` in Kimodo coordinates
   - `local_joint_rots` for the G1 skeleton
4. Resample from 50 FPS to the model's 30 FPS using time-based interpolation instead of integer stride downsampling.
5. Reuse the existing text/timeline sampling, feature encoding, normalization, padding, and collate logic where possible.

## Architecture Choices

### Recommended

Add a new `load_g1_npz_motion(...)` loader and a sibling dataset class such as `G1NPZTextDataset`.

Why:

- Preserves the current CSV path unchanged.
- Makes NPZ-specific assumptions explicit.
- Keeps training config readable by selecting a dataset target rather than overloading CSV-only fields.

### Rejected

Extend `G1CSVTextDataset` with a `data_format` switch.

Why not:

- The class constructor, path scanning, and loader kwargs are CSV-shaped.
- NPZ uses quaternions and MuJoCo `qpos`; CSV uses Euler angles and column names.
- The resulting class would become harder to test and reason about.

## Training Configuration Changes

We will add a separate phase-1 config for custom motion. It should:

- point at the new NPZ dataset class
- use `npz_root: ./custom_motion/robot-object`
- use `timelines_path: ./custom_motion/timeline_sub10.jsonl`
- set `input_fps: 50`
- keep the existing G1 motion rep and pretrained denoiser initialization
- keep phase-2 disabled for this first pass

CSV-only fields like `root_euler_order`, `root_angle_unit`, and `joint_angle_unit` should be removed from the custom NPZ config.

## Testing Strategy

We will add tests before implementation to verify:

- NPZ loader interprets robot/object slices correctly.
- NPZ loader returns G1-shaped tensors with the expected frame count behavior.
- Annotated-subset filtering yields exactly the timeline-covered clips.
- 50 FPS input can be resampled to 30 FPS without relying on integer stride logic.

## Risks

- The reverse conversion from MuJoCo `qpos` to Kimodo local rotations must match the forward assumptions used by `MujocoQposConverter`.
- Resampling rotations requires care; naive per-matrix interpolation is invalid.
- Timeline event cropping is time-based, so the resampling stage must preserve consistent timestamps.

## First Implementation Boundary

This change will support only the 192 annotated clips and phase-1 finetuning. Object state and contact channels remain available in the NPZ files but will not be modeled yet.
