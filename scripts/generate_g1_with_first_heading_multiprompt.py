#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kimodo import DEFAULT_MODEL, load_model
from kimodo.constraints import load_constraints_lst
from kimodo.model.cfg import CFG_TYPES
from kimodo.model.registry import get_model_info
from kimodo.tools import seed_everything
from kimodo.training.timeline_annotations import TimelineAnnotationIndex, normalize_clip_name

from scripts.generate_g1_with_first_heading import extract_first_heading_angle_from_npz, resolve_num_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate G1 motion with multi-prompt sequencing, timeline-derived segment text, and true first heading."
    )
    parser.add_argument(
        "--timeline_jsonl",
        type=str,
        required=True,
        help="Timeline JSONL used to derive event descriptions and event durations.",
    )
    parser.add_argument(
        "--heading_source_npz",
        type=str,
        required=True,
        help="NPZ path used to derive first heading and default clip name.",
    )
    parser.add_argument(
        "--clip_name",
        type=str,
        default=None,
        help="Timeline clip name override. Defaults to the stem of --heading_source_npz.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Name of the model (e.g. Kimodo-G1-RP-v1).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Total duration in seconds for the full clip.",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of diffusion steps.")
    parser.add_argument(
        "--hard_project_observed_motion",
        action="store_true",
        help="Enable per-step hard projection for constrained features during sampling.",
    )
    parser.add_argument(
        "--hard_project_prefix_frames",
        type=int,
        default=0,
        help="If >0, hard-project only the first K frames of the first segment; <=0 means all constrained frames.",
    )
    parser.add_argument(
        "--hard_project_release_frames",
        type=int,
        default=0,
        help="If >0 and prefix hard projection is enabled, apply a decaying release for this many frames.",
    )
    parser.add_argument(
        "--num_transition_frames",
        type=int,
        default=5,
        help="Number of shared transition frames between prompt segments.",
    )
    parser.add_argument(
        "--share_transition",
        action="store_true",
        default=True,
        help="Share transition frames between adjacent prompts (default: true).",
    )
    parser.add_argument(
        "--no-share-transition",
        dest="share_transition",
        action="store_false",
        help="Disable shared transition frames between adjacent prompts.",
    )
    parser.add_argument("--constraints", type=str, default=None, help="Path to constraints.json.")
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
        "--save_segments_json",
        type=str,
        default=None,
        help="Optional path to save the resolved segment prompts/frame counts as JSON for inspection.",
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


def _resolve_clip_name(args: argparse.Namespace) -> str:
    if args.clip_name:
        return normalize_clip_name(args.clip_name)
    return normalize_clip_name(args.heading_source_npz)


def build_multiprompt_segments(
    *,
    timeline_jsonl: str,
    clip_name: str,
    fps: float,
    total_duration_sec: float,
    constraints_path: str | None = None,
) -> dict:
    timeline_index = TimelineAnnotationIndex.from_jsonl(timeline_jsonl)
    record = timeline_index.get_record(clip_name)
    events = record["events"]
    if not events:
        raise ValueError(f"Timeline record {clip_name!r} does not contain any events for multi-prompt generation.")

    total_frames, max_constraint_frame = resolve_num_frames(total_duration_sec, fps, constraints_path)
    num_segments = len(events)
    if total_frames < num_segments:
        raise ValueError(
            f"Total frames {total_frames} is smaller than the number of events {num_segments}; "
            "cannot assign at least one frame per segment."
        )

    boundaries = [0]
    for event_idx, event in enumerate(events[:-1], start=1):
        candidate = int(round(float(event["end_time"]) * float(fps)))
        min_allowed = boundaries[-1] + 1
        remaining_segments = num_segments - event_idx
        max_allowed = total_frames - remaining_segments
        boundary = max(min_allowed, min(candidate, max_allowed))
        boundaries.append(boundary)
    boundaries.append(total_frames)

    prompts = [str(event["description"]).strip() for event in events]
    frame_counts = [int(boundaries[i + 1] - boundaries[i]) for i in range(num_segments)]
    durations_sec = [float(frames) / float(fps) for frames in frame_counts]

    if any(frames <= 0 for frames in frame_counts):
        raise ValueError(f"Non-positive frame count detected in multi-prompt segmentation: {frame_counts}")

    return {
        "clip_name": clip_name,
        "overview_description": str(record["overview_description"]),
        "prompts": prompts,
        "frame_counts": frame_counts,
        "durations_sec": durations_sec,
        "total_frames": int(total_frames),
        "total_duration_sec": float(total_duration_sec),
        "max_constraint_frame": max_constraint_frame,
        "boundaries": boundaries,
    }


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    clip_name = _resolve_clip_name(args)
    segments = build_multiprompt_segments(
        timeline_jsonl=args.timeline_jsonl,
        clip_name=clip_name,
        fps=float(model.fps),
        total_duration_sec=float(args.duration),
        constraints_path=args.constraints,
    )

    print(f"Clip: {segments['clip_name']}")
    print(f"Total duration seconds: {segments['total_duration_sec']:.4f}")
    print(f"Total frames: {segments['total_frames']}")
    if segments["max_constraint_frame"] is not None:
        print(f"Max constraint frame index: {segments['max_constraint_frame']}")
    print(f"Number of prompt segments: {len(segments['prompts'])}")
    for idx, (prompt, frames, seconds) in enumerate(
        zip(segments["prompts"], segments["frame_counts"], segments["durations_sec"]),
        start=1,
    ):
        print(f"  [{idx:02d}] frames={frames:3d} duration={seconds:.3f}s prompt={prompt!r}")

    if args.save_segments_json:
        save_path = Path(args.save_segments_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
        print(f"Saved segment plan: {save_path}")

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
        segments["prompts"],
        segments["frame_counts"],
        constraint_lst=constraint_lst,
        num_denoising_steps=args.diffusion_steps,
        num_samples=args.num_samples,
        multi_prompt=True,
        num_transition_frames=args.num_transition_frames,
        share_transition=args.share_transition,
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

    if resolved_model == "kimodo-g1-rp":
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
