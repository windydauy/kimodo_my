# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kimodo is NVIDIA's kinematic motion diffusion model for generating 3D human and robot motions from text prompts with constraint support (full-body keyframes, end-effector positions/rotations, 2D paths). It supports multiple skeleton types (SOMA, SMPL-X, Unitree G1) and outputs to NPZ, BVH, MuJoCo CSV, and AMASS formats.

## Build & Run Commands

### Installation
```bash
pip install -e .                # Core install (requires cmake for MotionCorrection C++ extension)
pip install -e ".[demo]"        # Includes viser (custom fork) for interactive demo
pip install -e ".[soma]"        # Includes py-soma-x
pip install -e ".[all]"         # Everything
```

Set `SKIP_MOTION_CORRECTION_IN_SETUP=1` to skip the C++ extension build (used in Docker).

### CLI Entry Points
```bash
kimodo_gen "walking forward" --model kimodo-soma-rp --duration 5 --output out.npz
kimodo_demo                     # Launch interactive web demo (default port 7860)
kimodo_textencoder              # Run LLM2Vec text encoder service (default port 9550)
```

### Docker
```bash
docker build -t kimodo:1.0 .
docker-compose up -d            # Starts text-encoder (port 9550) + demo (port 7860)
```

### Documentation
```bash
cd docs && pip install -r requirements.txt
make apidoc && make html        # Generate API stubs then build HTML docs
```

### Linting & Formatting
Pre-commit hooks handle all formatting. Run manually with:
```bash
pre-commit run --all-files
```
- **ruff**: Import sorting (`--select I --fix`) and code formatting, line-length=120
- **docformatter**: Sphinx-style docstrings, wrap at 100 chars
- **prettier**: YAML files

### Commits
All commits require DCO sign-off: `git commit -s -m "message"`

## Architecture

### Core Pipeline
Text prompt → **LLM2Vec text encoder** (local or remote API on port 9550) → text embeddings → **Kimodo diffusion model** (DDIM sampling + classifier-free guidance) → motion representation → **post-processing** (foot skate reduction, constraint enforcement) → export

### Key Packages (`kimodo/`)

- **`model/`** — Diffusion model, denoiser backbone, CFG wrapper (3 modes: nocfg, regular, separated), DDIM sampling, model loading with HuggingFace cache. `kimodo_model.py` is the main inference class. `registry.py` defines 5 Kimodo models + 1 TMR model with their HF repo IDs.
- **`skeleton/`** — Skeleton hierarchy/kinematics. Concrete types: `SOMASkeleton77`, `SOMASkeleton30`, `SMPLXSkeleton22`, `G1Skeleton34`. Factory via `build_skeleton()`.
- **`motion_rep/`** — Motion encoding/decoding. `KimodoMotionRep` uses global root + global joint rotations in 6D continuous representation. Handles velocities, foot contacts, heading.
- **`constraints.py`** — Three constraint types: `Root2DConstraintSet` (ground plane trajectory), `FullBodyConstraintSet` (joint poses at keyframes), `EndEffectorConstraintSet` (hand/foot control). All serialize to/from JSON.
- **`exports/`** — Output format converters: BVH (SOMA), MuJoCo CSV (G1), AMASS NPZ (SMPL-X).
- **`postprocess.py`** — Foot skate reduction via IK, constraint enforcement and cleanup.
- **`demo/`** — Interactive web UI using Gradio + Viser. `app.py` manages sessions, `ui.py` (143KB) builds the Gradio interface, `generation.py` orchestrates async generation.
- **`viz/`** — 3D visualization with Viser. `constraint_ui.py` (46KB) provides interactive constraint editing.
- **`scripts/`** — CLI tools. `generate.py` handles batch generation via argparse.
- **`geometry.py`** — Rotation conversions (matrices, quaternions, axis-angle).
- **`tools.py`** — Shared utilities (validation, batching, JSON I/O, seeding, tensor ops).

### MotionCorrection (C++ Extension)
`MotionCorrection/` contains a C++ library with Python bindings, built via CMake through `setup.py`. Provides optimized motion correction operations.

### Text Encoder Architecture
The system supports two text encoding modes:
1. **Local**: LLM2Vec (McGill-NLP, Llama 3 8B with LoRA) loaded in-process
2. **Remote API**: Standalone Gradio service on port 9550 (`kimodo_textencoder`)

Auto-fallback: tries remote API first, falls back to local if unreachable. Set `TEXT_ENCODER_URL` env var to point to a custom encoder service.

### Model Registry
5 models identified by short keys: `kimodo-soma-rp`, `kimodo-soma-seed`, `kimodo-g1-rp`, `kimodo-g1-seed`, `kimodo-smplx-rp`. Default: `kimodo-soma-rp`. Each maps to a HuggingFace repo. Access via `kimodo.AVAILABLE_MODELS` and `kimodo.load_model()`.

## Code Style
- Line length: 120 characters
- Imports: sorted by ruff (first-party: `kimodo`, third-party: `torch`, `numpy`, etc.)
- Docstrings: Sphinx format
- All source files carry SPDX copyright/license headers
- Config management: Hydra + OmegaConf
- GPU requirement: ~17GB VRAM
