#!/usr/bin/env python3
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kimodo import DEFAULT_MODEL
from kimodo.constraints import load_constraints_lst
from kimodo.model.cfg import CFG_TYPES
from kimodo.model.kimodo_model import Kimodo
from kimodo.model.loading import instantiate_from_dict
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
    parser = argparse.ArgumentParser(
        description="Generate G1 motion with distilled Flow-Matching student using discrete-time aligned sampling."
    )
    parser.add_argument("prompt", type=str, help="Text prompt describing the motion to generate.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Name of the model (e.g. Kimodo-G1-RP-v1).",
    )
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--inference_steps", type=int, default=20, help="Number of continuous ODE solver steps.")
    parser.add_argument(
        "--ode_solver",
        type=str,
        default="euler",
        choices=("euler", "heun"),
        help="Continuous ODE solver. Euler is faster; Heun uses two model evaluations per step.",
    )
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
        help="Distillation YAML path (required).",
    )
    parser.add_argument(
        "--distill_ckpt",
        type=str,
        default=None,
        help="Distillation checkpoint path (required). Supports student/ema.shadow/state_dict.",
    )
    args = parser.parse_args()
    if not args.distill_config or not args.distill_ckpt:
        raise ValueError("This script requires --distill_config and --distill_ckpt.")
    return args


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


def _patch_continuous_timestep_embedding(model: Kimodo, num_base_steps: int) -> None:
    for m in model.denoiser.modules():
        if not hasattr(m, "embed_timestep"):
            continue
        emb = m.embed_timestep
        if getattr(emb, "_continuous_patch_applied", False):
            continue
        orig_forward = emb.forward

        def _forward_cont(self, timesteps, _orig_forward=orig_forward):
            if torch.is_floating_point(timesteps):
                t = timesteps.clamp(0.0, float(num_base_steps - 1))
                t0 = torch.floor(t).to(torch.long)
                t1 = torch.clamp(t0 + 1, max=num_base_steps - 1)
                # Keep shape aligned with TimestepEmbedder output [B, 1, D].
                w = (t - t0.to(t.dtype)).view(-1, 1, 1)
                e0 = _orig_forward(t0)
                e1 = _orig_forward(t1)
                return e0 * (1.0 - w) + e1 * w
            return _orig_forward(timesteps.to(torch.long))

        emb.forward = types.MethodType(_forward_cont, emb)
        emb._continuous_patch_applied = True


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
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported distill checkpoint format: {distill_ckpt_path}")

    if "student" in ckpt and isinstance(ckpt["student"], dict):
        state = ckpt["student"]
        source = "student"
    elif "ema" in ckpt and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
        shadow = ckpt["ema"]["shadow"]
        if not isinstance(shadow, dict):
            raise ValueError(f"Invalid ema.shadow format in checkpoint: {distill_ckpt_path}")
        state = shadow
        source = "ema.shadow"
    elif all(isinstance(k, str) for k in ckpt.keys()) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
        source = "state_dict"
    else:
        raise ValueError(
            "Distill checkpoint must contain one of: {'student': state_dict}, "
            "{'ema': {'shadow': state_dict}}, or be a plain state_dict. "
            f"Got keys={list(ckpt.keys())[:10]} from {distill_ckpt_path}"
        )

    missing, unexpected = denoiser.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys when loading {source} from {distill_ckpt_path}: first 10={unexpected[:10]}"
        )
    if missing:
        raise ValueError(
            f"Missing keys when loading {source} from {distill_ckpt_path}: first 10={missing[:10]}"
        )
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


def _encode_text_for_sampling(model: Kimodo, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    text_feat, text_length = model.text_encoder(texts)
    text_feat = text_feat.to(model.device)
    empty_text_mask = [len(text.strip()) == 0 for text in texts]
    text_feat[empty_text_mask] = 0
    batch_size, maxlen = text_feat.shape[:2]
    tensor_text_length = torch.tensor(text_length, device=model.device)
    tensor_text_length[empty_text_mask] = 0
    text_pad_mask = torch.arange(maxlen, device=model.device).expand(batch_size, maxlen) < tensor_text_length[:, None]
    return text_feat, text_pad_mask


def _generate_flowmatch_motion_continuous(
    *,
    model: Kimodo,
    texts: list[str],
    max_frames: int,
    num_steps: int,
    pad_mask: torch.Tensor,
    first_heading_angle: torch.Tensor,
    motion_mask: torch.Tensor | None,
    observed_motion: torch.Tensor | None,
    cfg_weight: float | list[float],
    cfg_type: str,
    ode_solver: str,
    hard_project_observed_motion: bool,
    hard_project_prefix_frames: int,
    hard_project_release_frames: int,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("flowmatch sampling requires at least 2 steps.")

    text_feat, text_pad_mask = _encode_text_for_sampling(model, texts)
    if motion_mask is not None and motion_mask.dtype == torch.bool:
        motion_mask = motion_mask.to(dtype=text_feat.dtype)

    bsz = text_feat.shape[0]
    x = torch.randn((bsz, max_frames, model.motion_rep.motion_rep_dim), device=model.device)
    # Continuous time grid in [1 -> 0].
    t_grid = torch.linspace(1.0, 0.0, steps=num_steps, device=model.device, dtype=x.dtype)

    hard_mask = None
    hard_boundary_frame = None
    if hard_project_observed_motion and motion_mask is not None and observed_motion is not None:
        hard_mask = motion_mask > 0 if motion_mask.dtype != torch.bool else motion_mask.clone()
        if hard_project_prefix_frames > 0:
            n_frames = min(int(hard_project_prefix_frames), hard_mask.shape[1])
            prefix_mask = torch.zeros_like(hard_mask, dtype=torch.bool, device=hard_mask.device)
            prefix_mask[:, :n_frames, :] = True
            hard_mask = hard_mask & prefix_mask
            hard_boundary_frame = n_frames - 1 if n_frames > 0 else None
        if not bool(hard_mask.any().item()):
            hard_mask = None

    with torch.inference_mode():
        for k in tqdm(range(0, num_steps - 1)):
            if hard_mask is not None and observed_motion is not None:
                x_preproj = x
                x = torch.where(hard_mask, observed_motion, x)
                if (
                    hard_project_release_frames > 0
                    and hard_boundary_frame is not None
                    and hard_boundary_frame + 1 < x.shape[1]
                ):
                    boundary_mask = hard_mask[:, hard_boundary_frame, :]
                    if bool(boundary_mask.any().item()):
                        boundary_delta = observed_motion[:, hard_boundary_frame, :] - x_preproj[:, hard_boundary_frame, :]
                        total = int(hard_project_release_frames)
                        for rel_idx in range(1, total + 1):
                            frame_idx = hard_boundary_frame + rel_idx
                            if frame_idx >= x.shape[1]:
                                break
                            alpha = float(total - rel_idx + 1) / float(total + 1)
                            release_delta = boundary_delta * alpha
                            x[:, frame_idx, :] = torch.where(
                                boundary_mask,
                                x[:, frame_idx, :] + release_delta,
                                x[:, frame_idx, :],
                            )

            t_cur = t_grid[k]
            t_nxt_scalar = t_grid[k + 1]
            dt = (t_nxt_scalar - t_cur).view(1, 1, 1)
            t_cur_idx = torch.full((bsz,), float(t_cur.item() * float(model.diffusion.num_base_steps - 1)), device=model.device, dtype=x.dtype)
            v1 = model.denoiser(
                cfg_weight,
                x,
                pad_mask,
                text_feat,
                text_pad_mask,
                t_cur_idx,
                first_heading_angle,
                motion_mask,
                observed_motion,
                cfg_type=cfg_type,
            )
            if ode_solver == "euler":
                x = x + dt * v1
            elif ode_solver == "heun":
                x_euler = x + dt * v1
                t_nxt_idx = torch.full(
                    (bsz,),
                    float(t_nxt_scalar.item() * float(model.diffusion.num_base_steps - 1)),
                    device=model.device,
                    dtype=x.dtype,
                )
                v2 = model.denoiser(
                    cfg_weight,
                    x_euler,
                    pad_mask,
                    text_feat,
                    text_pad_mask,
                    t_nxt_idx,
                    first_heading_angle,
                    motion_mask,
                    observed_motion,
                    cfg_type=cfg_type,
                )
                x = x + 0.5 * dt * (v1 + v2)
            else:
                raise ValueError(f"Unsupported ode_solver={ode_solver!r}.")

    if hard_mask is not None and observed_motion is not None:
        x = torch.where(hard_mask, observed_motion, x)
    return x


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = _build_model_from_distill(
        distill_config_path=args.distill_config,
        distill_ckpt_path=args.distill_ckpt,
        device=device,
    )
    print(f"Loaded distill student model from config={args.distill_config} ckpt={args.distill_ckpt}")
    _patch_continuous_timestep_embedding(model, int(model.diffusion.num_base_steps))

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
    print("Sampler mode: flowmatch_continuous_ode")
    first_heading_tensor = torch.tensor([first_heading_angle], dtype=torch.float32, device=device)
    if args.num_samples != 1:
        raise ValueError("Flowmatch continuous sampler currently supports num_samples=1 in this script.")
    if args.cfg_type == "nocfg":
        raise ValueError("Flowmatch continuous sampler does not support cfg_type=nocfg.")

    lengths = torch.tensor([num_frames], device=device)
    max_frames = int(num_frames)
    motion_pad_mask = torch.arange(max_frames, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    observed_motion, motion_mask = None, None
    if constraint_lst:
        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraint_lst,
            lengths,
            to_normalize=True,
            device=device,
        )

    texts = [args.prompt]
    motion = _generate_flowmatch_motion_continuous(
        model=model,
        texts=texts,
        max_frames=max_frames,
        num_steps=args.inference_steps,
        pad_mask=motion_pad_mask,
        first_heading_angle=first_heading_tensor,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        cfg_weight=cfg_kwargs["cfg_weight"],
        cfg_type=cfg_kwargs["cfg_type"],
        ode_solver=args.ode_solver,
        hard_project_observed_motion=args.hard_project_observed_motion,
        hard_project_prefix_frames=args.hard_project_prefix_frames,
        hard_project_release_frames=args.hard_project_release_frames,
    )
    output = model.motion_rep.inverse(motion, is_normalized=True, return_numpy=True)

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
