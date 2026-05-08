#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kimodo import DEFAULT_MODEL, load_model
from kimodo.constraints import load_constraints_lst
from kimodo.model.cfg import CFG_TYPES
from kimodo.model.kimodo_model import Kimodo
from kimodo.model.loading import instantiate_from_dict
from kimodo.model.registry import get_model_info
from kimodo.tools import seed_everything


MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def rot_mujoco_to_kimodo(rot_m: np.ndarray) -> np.ndarray:
    return MUJOCO_TO_KIMODO @ rot_m @ MUJOCO_TO_KIMODO.T


def yaw_from_rot_kimodo(rot_k: np.ndarray) -> float:
    forward = rot_k @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(np.arctan2(forward[0], forward[2]))


def extract_first_heading_angle_from_npz(npz_path: str | os.PathLike[str]) -> float:
    data = np.load(str(npz_path), allow_pickle=False)
    if "qpos" in data.files:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        if qpos.shape[1] < 7:
            raise ValueError(f"Expected qpos with at least 7 columns, got {tuple(qpos.shape)}.")
        root_rot_m = R.from_quat(qpos[0, 3:7], scalar_first=True).as_matrix()
        return yaw_from_rot_kimodo(rot_mujoco_to_kimodo(root_rot_m))

    if "root_global_6d" in data.files:
        root_global_6d = np.asarray(data["root_global_6d"], dtype=np.float64)
        if root_global_6d.ndim != 2 or root_global_6d.shape[1] < 6:
            raise ValueError(f"Expected root_global_6d shape [T, 6], got {tuple(root_global_6d.shape)}.")
        root_rot_m = R.from_euler("xyz", root_global_6d[0, 3:6]).as_matrix()
        return yaw_from_rot_kimodo(rot_mujoco_to_kimodo(root_rot_m))

    raise ValueError(f"NPZ {npz_path} does not contain qpos or root_global_6d for heading extraction.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate G1 motion while injecting the true first-frame heading.")
    parser.add_argument("prompt", type=str, help="Text prompt describing the motion to generate.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Name of the model (e.g. Kimodo-G1-RP-v1).",
    )
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of diffusion steps.")
    parser.add_argument("--num_transition_frames", type=int, default=5, help="Transition frames for multi-prompt mode.")
    parser.add_argument(
        "--hard_project_observed_motion",
        action="store_true",
        help="Enable per-step hard projection for constrained features during sampling.",
    )
    parser.add_argument(
        "--hard_project_prefix_frames",
        type=int,
        default=0,
        help="If >0, hard-project only the first K frames; <=0 means all constrained frames.",
    )
    parser.add_argument(
        "--hard_project_release_frames",
        type=int,
        default=0,
        help="If >0 and prefix hard projection is enabled, apply a decaying release for this many frames.",
    )
    parser.add_argument("--constraints", type=str, default=None, help="Path to constraints.json.")
    parser.add_argument("--heading_source_npz", type=str, required=True, help="NPZ path used to derive first heading.")
    parser.add_argument("--output", type=str, default="output", help="Output stem name.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible results.")
    parser.add_argument(
        "--cfg_type",
        type=str,
        default="separated",
        choices=CFG_TYPES,
        help="Classifier-free guidance mode.",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        nargs="*",
        default=[2.0, 2.0],
        help="CFG scale(s): one float for regular, or two floats [text_weight, constraint_weight] for separated.",
    )
    parser.add_argument(
        "--distill_config",
        type=str,
        default=None,
        help="Optional distillation YAML path. If provided with --distill_ckpt, build inference model from student config.",
    )
    parser.add_argument(
        "--distill_ckpt",
        type=str,
        default=None,
        help="Optional distillation checkpoint (.pt). Expects key 'student' for strict student loading.",
    )
    return parser.parse_args()


def _single_file_path(path: str, ext: str) -> str:
    if not path.endswith(ext):
        path = path.rstrip(os.sep) + ext
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _output_dir_and_path(path: str, default_base: str, ext: str):
    folder = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(folder.rstrip(os.sep))
    return folder, os.path.join(folder, default_base + ext), base_name


def _resolve_cfg_kwargs(args: argparse.Namespace) -> dict:
    if args.cfg_type == "nocfg":
        if args.cfg_weight:
            raise ValueError("--cfg_weight is not used with --cfg_type nocfg.")
        return {"cfg_type": "nocfg"}
    if args.cfg_type == "regular":
        if len(args.cfg_weight) != 1:
            raise ValueError("--cfg_type regular requires exactly one --cfg_weight value.")
        return {"cfg_type": "regular", "cfg_weight": float(args.cfg_weight[0])}
    if len(args.cfg_weight) != 2:
        raise ValueError("--cfg_type separated requires exactly two --cfg_weight values.")
    return {"cfg_type": "separated", "cfg_weight": [float(args.cfg_weight[0]), float(args.cfg_weight[1])]}


def resolve_num_frames(duration_sec: float, fps: float, constraints_path: str | None = None) -> tuple[int, int | None]:
    base_num_frames = int(float(duration_sec) * float(fps))
    max_constraint_frame = None

    if constraints_path:
        with open(constraints_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        frame_indices = []
        for item in items:
            frame_indices.extend(int(x) for x in item.get("frame_indices", []))
        if frame_indices:
            max_constraint_frame = max(frame_indices)
            base_num_frames = max(base_num_frames, max_constraint_frame + 1)

    return base_num_frames, max_constraint_frame


def _build_model_from_distill(
    *,
    distill_config_path: str,
    distill_ckpt_path: str,
    device: str,
) -> Kimodo:
    cfg = OmegaConf.load(distill_config_path)
    student_cfg = OmegaConf.to_container(cfg.model.student_denoiser, resolve=True)
    text_encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True)

    denoiser = instantiate_from_dict(student_cfg).to(device)
    text_encoder = instantiate_from_dict(text_encoder_cfg)
    text_encoder.to(device)
    text_encoder.eval()

    ckpt = torch.load(distill_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "student" not in ckpt:
        raise ValueError(
            f"Distill checkpoint must be a dict containing key 'student': {distill_ckpt_path}"
        )
    denoiser.load_state_dict(ckpt["student"], strict=True)
    denoiser.eval()

    num_base_steps = int(cfg.model.num_base_steps)
    model = Kimodo(
        denoiser=denoiser,
        text_encoder=text_encoder,
        num_base_steps=num_base_steps,
        device=device,
        cfg_type="separated",
    )
    return model


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if bool(args.distill_config) != bool(args.distill_ckpt):
        raise ValueError("--distill_config and --distill_ckpt must be provided together.")

    if args.distill_config and args.distill_ckpt:
        model = _build_model_from_distill(
            distill_config_path=args.distill_config,
            distill_ckpt_path=args.distill_ckpt,
            device=device,
        )
        resolved_model = "distill-student"
        print(f"Loaded distill student model from config={args.distill_config} ckpt={args.distill_ckpt}")
    else:
        model, resolved_model = load_model(
            args.model,
            device=device,
            default_family="Kimodo",
            return_resolved_name=True,
        )
        info = get_model_info(resolved_model)
        display = info.display_name if info else resolved_model
        print(f"Loaded model: {display} ({resolved_model})")

    if args.seed is not None:
        seed_everything(args.seed)

    num_frames, max_constraint_frame = resolve_num_frames(args.duration, model.fps, args.constraints)
    print(f"Prompt: {args.prompt!r}")
    print(f"Duration seconds: {args.duration:.4f}")
    print(f"Frames: {num_frames}")
    if max_constraint_frame is not None:
        print(f"Max constraint frame index: {max_constraint_frame}")

    constraint_lst = []
    if args.constraints:
        constraint_lst = load_constraints_lst(args.constraints, model.skeleton)
        print(f"Using {len(constraint_lst)} set of constraints")
        for constraint in constraint_lst:
            print(f"    {constraint}")

    first_heading_angle = extract_first_heading_angle_from_npz(args.heading_source_npz)
    print(f"Using first heading angle from {args.heading_source_npz}: {first_heading_angle:.6f} rad")

    cfg_kwargs = _resolve_cfg_kwargs(args)
    output = model(
        args.prompt,
        num_frames,
        constraint_lst=constraint_lst,
        num_denoising_steps=args.diffusion_steps,
        num_samples=args.num_samples,
        multi_prompt=False,
        num_transition_frames=args.num_transition_frames,
        post_processing=False,
        return_numpy=True,
        first_heading_angle=torch.tensor([first_heading_angle], dtype=torch.float32, device=device),
        hard_project_observed_motion=args.hard_project_observed_motion,
        hard_project_prefix_frames=args.hard_project_prefix_frames,
        hard_project_release_frames=args.hard_project_release_frames,
        **cfg_kwargs,
    )

    n_samples = int(output["posed_joints"].shape[0])
    output_base = args.output

    if n_samples == 1:
        npz_path = _single_file_path(output_base, ".npz")
        print(f"Saving the npz output to {npz_path}")
        single = {
            k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
            for k, v in output.items()
        }
        np.savez(npz_path, **single)
    else:
        out_dir, _, base_name = _output_dir_and_path(output_base, "motion", ".npz")
        print(f"Saving the npz output to {out_dir}/ ({base_name}_00.npz ...)")
        for i in range(n_samples):
            single = {
                k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                for k, v in output.items()
            }
            np.savez(os.path.join(out_dir, f"{base_name}_{i:02d}.npz"), **single)

    # Export MuJoCo qpos CSV whenever this is a G1 skeleton model.
    # Distill-student mode uses the same G1 skeleton but does not use the
    # canonical registry key "kimodo-g1-rp".
    if getattr(model.skeleton, "name", "").lower().startswith("g1"):
        from kimodo.exports.mujoco import MujocoQposConverter

        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(output, device)
        if n_samples == 1:
            csv_path = _single_file_path(output_base, ".csv")
            print(f"Saving the csv output to {csv_path}")
            converter.save_csv(qpos, csv_path)
        else:
            out_dir, _, base_name = _output_dir_and_path(output_base, "qpos", ".csv")
            print(f"Saving the csv output to {out_dir}/ ({base_name}_00.csv ...)")
            converter.save_csv(qpos, os.path.join(out_dir, base_name + ".csv"))


if __name__ == "__main__":
    main()
