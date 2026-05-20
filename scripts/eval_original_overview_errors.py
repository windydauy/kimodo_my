#!/usr/bin/env python3
"""Batch-evaluate original NPZ clips with overview prompts and report EE errors."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from kimodo import load_model
from kimodo.constraints import FullBodyConstraintSet, load_constraints_lst
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.kimodo_model import Kimodo
from kimodo.model.loading import instantiate_from_dict
from kimodo.skeleton import G1Skeleton34
from kimodo.training.custom_motion_ee_pose_npz import save_custom_motion_ee_pose_npz
from kimodo.training.custom_motion_npz import load_g1_npz_motion, resample_motion
from generate_g1_with_first_heading import (
    extract_first_heading_angle_from_npz,
    resolve_num_frames,
)
from npz_to_ee_pose_constraints import (
    EE_FIELD_MAP,
    mujoco_xyz_to_kimodo,
    rot_mujoco_to_kimodo,
    select_source_indices,
    yaw_from_rot_kimodo,
)
from resolve_timeline_overview_prompt import resolve_overview_prompt


FIELD_TO_BODY = {
    "left_hand_pose": "left_wrist_yaw_link",
    "right_hand_pose": "right_wrist_yaw_link",
    "left_foot_pose": "left_ankle_roll_link",
    "right_foot_pose": "right_ankle_roll_link",
}
HAND_FIELDS = {"left_hand_pose", "right_hand_pose"}
DEFAULT_G1_MODEL = "Kimodo-G1-RP-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all *_original.npz clips by generating with overview prompts and summarizing EE errors."
    )
    parser.add_argument(
        "--npz_glob",
        default="custom_motion/robot-object/sub10*_original.npz",
        help="Glob pattern for original clips.",
    )
    parser.add_argument(
        "--timeline_jsonl",
        default="custom_motion/timeline_sub10.jsonl",
        help="Timeline JSONL used to resolve overview prompts.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_G1_MODEL,
        help="Model name to load. For this evaluator, use a G1 model (default: Kimodo-G1-RP-v1).",
    )
    parser.add_argument("--target_fps", type=float, default=30.0, help="Target FPS for constraints.")
    parser.add_argument("--keyframe_step", type=int, default=30, help="Source keyframe step for constraints.")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Diffusion steps for generation.")
    parser.add_argument(
        "--num_runs_per_task",
        type=int,
        default=5,
        help="How many generation/eval runs to execute per clip, then average per-clip metrics (default: 5).",
    )
    parser.add_argument(
        "--distill_config",
        default=None,
        help="Optional distillation YAML. If provided with --distill_ckpt, evaluate the distilled student model.",
    )
    parser.add_argument(
        "--distill_ckpt",
        default=None,
        help="Optional distillation checkpoint (.pt). Supports both student checkpoints and ema_final.pt.",
    )
    parser.add_argument("--xml", default="kimodo/assets/skeletons/g1skel34/xml/g1.xml", help="MuJoCo XML path.")
    parser.add_argument(
        "--hard_project_observed_motion",
        action="store_true",
        help=(
            "Enable per-step hard projection using dense full-body prefix constraints. "
            "Only effective when --hard_project_prefix_frames > 0."
        ),
    )
    parser.add_argument(
        "--hard_project_prefix_frames",
        type=int,
        default=0,
        help="If >0, enforce dense full-body hard projection on the first K frames.",
    )
    parser.add_argument(
        "--output_json",
        default="scripts/eval_original_overview_errors_summary.json",
        help="Where to save detailed per-clip and overall stats.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately when one clip fails.",
    )
    parser.add_argument(
        "--include_unmatched_clips",
        action="store_true",
        help=(
            "By default, only evaluate clips that exist in timeline_jsonl. "
            "Set this flag to include unmatched clips too (they may fail with missing timeline annotation)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-clip metrics and model/load details.",
    )
    return parser.parse_args()


def _write_constraints_from_adapter(
    adapter_npz_path: Path,
    output_json_path: Path,
    target_fps: float,
    keyframe_step: int,
) -> None:
    data = np.load(adapter_npz_path, allow_pickle=True)
    required_keys = {"fps", "root_global_6d", "ee_root_relative_6d", "ee_names"}
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"Missing required keys in npz: {sorted(missing)}")

    src_fps = float(data["fps"])
    root_global_6d = np.asarray(data["root_global_6d"], dtype=np.float64)
    ee_rel_6d = np.asarray(data["ee_root_relative_6d"], dtype=np.float64)
    ee_names = [str(x) for x in data["ee_names"].tolist()]

    num_frames = root_global_6d.shape[0]
    src_idx = select_source_indices(num_frames=num_frames, step=max(1, keyframe_step))
    dst_idx = np.round(src_idx * target_fps / src_fps).astype(np.int64)

    uniq_mask = np.ones(len(dst_idx), dtype=bool)
    uniq_mask[1:] = dst_idx[1:] != dst_idx[:-1]
    src_idx = src_idx[uniq_mask]
    dst_idx = dst_idx[uniq_mask]

    item: dict[str, object] = {
        "type": "ee-pose",
        "frame_indices": dst_idx.tolist(),
        "root_xyzyaw": [],
    }
    for field in EE_FIELD_MAP.values():
        item[field] = []

    name_to_col = {n: i for i, n in enumerate(ee_names)}

    for t in src_idx:
        root_xyz_m = root_global_6d[t, :3]
        root_rpy_m = root_global_6d[t, 3:]
        root_rot_m = R.from_euler("xyz", root_rpy_m).as_matrix()
        root_rot_k = rot_mujoco_to_kimodo(root_rot_m)
        root_xyz_k = mujoco_xyz_to_kimodo(root_xyz_m)
        root_yaw_k = yaw_from_rot_kimodo(root_rot_k)
        item["root_xyzyaw"].append([float(root_xyz_k[0]), float(root_xyz_k[1]), float(root_xyz_k[2]), root_yaw_k])

        for ee_name, field in EE_FIELD_MAP.items():
            if ee_name not in name_to_col:
                continue
            c = name_to_col[ee_name]
            rel_xyz_m = ee_rel_6d[t, c, :3]
            rel_rpy_m = ee_rel_6d[t, c, 3:]
            rel_rot_m = R.from_euler("xyz", rel_rpy_m).as_matrix()

            ee_xyz_m = root_xyz_m + root_rot_m @ rel_xyz_m
            ee_rot_m = root_rot_m @ rel_rot_m
            ee_xyz_k = mujoco_xyz_to_kimodo(ee_xyz_m)
            ee_rot_k = rot_mujoco_to_kimodo(ee_rot_m)
            ee_rpy_k = R.from_matrix(ee_rot_k).as_euler("xyz")

            item[field].append(
                [
                    float(ee_xyz_k[0]),
                    float(ee_xyz_k[1]),
                    float(ee_xyz_k[2]),
                    float(ee_rpy_k[0]),
                    float(ee_rpy_k[1]),
                    float(ee_rpy_k[2]),
                ]
            )

    for field in list(EE_FIELD_MAP.values()):
        if len(item[field]) == 0:
            item.pop(field)

    output_json_path.write_text(json.dumps([item], indent=2), encoding="utf-8")


def _write_fullbody_prefix_constraints_from_npz(
    npz_path: Path,
    output_json_path: Path,
    prefix_frames: int,
    target_fps: float,
) -> int:
    if prefix_frames <= 0:
        raise ValueError("prefix_frames must be > 0.")

    motion = load_g1_npz_motion(npz_path, device="cpu", dtype=torch.float32)
    local_joint_rots = motion["local_joint_rots"]
    root_positions = motion["root_positions"]
    src_fps = float(motion["input_fps"])

    local_joint_rots, root_positions = resample_motion(
        local_joint_rots,
        root_positions,
        input_fps=src_fps,
        target_fps=float(target_fps),
    )
    total_frames = int(local_joint_rots.shape[0])
    used_frames = min(int(prefix_frames), total_frames)
    if used_frames <= 0:
        raise ValueError("No frames available after resampling for fullbody prefix constraints.")

    skeleton = G1Skeleton34()
    frame_indices = torch.arange(used_frames, dtype=torch.long)
    global_joints_rots, global_joints_positions, _ = skeleton.fk(
        local_joint_rots[:used_frames],
        root_positions[:used_frames],
    )
    constraint = FullBodyConstraintSet(
        skeleton=skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_joints_positions,
        global_joints_rots=global_joints_rots,
    )

    payload = [_tensor_to_list(constraint.get_save_info())]
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return used_frames


def _merge_constraints_json(paths: list[Path], output_path: Path) -> int:
    merged: list[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return len(merged)


def _tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _tensor_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_to_list(v) for v in obj]
    return obj


def _kimodo_to_mujoco_xyz(xyz_k: np.ndarray) -> np.ndarray:
    return np.asarray([xyz_k[2], xyz_k[0], xyz_k[1]], dtype=np.float64)


def _evaluate_qpos_errors(
    qpos: np.ndarray,
    constraints_json_path: Path,
    mj_model: mujoco.MjModel,
) -> tuple[list[float], list[float]]:
    with constraints_json_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    ee_items = [x for x in items if x.get("type") == "ee-pose"]
    if not ee_items:
        raise ValueError(f"No ee-pose constraints found in: {constraints_json_path}")

    mj_data = mujoco.MjData(mj_model)
    all_errors: list[float] = []
    hand_errors: list[float] = []

    for item in ee_items:
        frame_indices = item.get("frame_indices", [])
        for local_i, frame_idx in enumerate(frame_indices):
            if frame_idx < 0 or frame_idx >= len(qpos):
                continue
            mj_data.qpos[:] = qpos[frame_idx]
            mujoco.mj_forward(mj_model, mj_data)

            for field, body_name in FIELD_TO_BODY.items():
                poses = item.get(field)
                if not poses:
                    continue
                expected_k = np.asarray(poses[local_i][:3], dtype=np.float64)
                expected = _kimodo_to_mujoco_xyz(expected_k)
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                actual = np.asarray(mj_data.xpos[body_id], dtype=np.float64)
                err = float(np.linalg.norm(actual - expected))
                all_errors.append(err)
                if field in HAND_FIELDS:
                    hand_errors.append(err)

    return all_errors, hand_errors


def _safe_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_error_m": None, "max_error_m": None, "min_error_m": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_error_m": float(arr.mean()),
        "max_error_m": float(arr.max()),
        "min_error_m": float(arr.min()),
    }


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean())


def _fmt_metric(v: float | None) -> str:
    return "nan" if v is None else f"{v:.6f}"


def _no_progress(iterable):
    return iterable


def _load_timeline_filenames(timeline_jsonl: str | Path) -> set[str]:
    path = Path(timeline_jsonl)
    names: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = obj.get("filename")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def _load_eval_model(args: argparse.Namespace, device: str):
    if bool(args.distill_config) != bool(args.distill_ckpt):
        raise ValueError("--distill_config and --distill_ckpt must be provided together.")

    if args.distill_config and args.distill_ckpt:
        model = _build_model_from_distill_flexible(
            distill_config_path=args.distill_config,
            distill_ckpt_path=args.distill_ckpt,
            device=device,
        )
        return model, "distill-student"

    return load_model(args.model, device=device, default_family="Kimodo", return_resolved_name=True)


def _resolve_student_state_dict_from_ckpt(ckpt_path: str) -> tuple[dict[str, torch.Tensor], str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported distill checkpoint format: {ckpt_path}")

    if "student" in ckpt and isinstance(ckpt["student"], dict):
        return ckpt["student"], "student"

    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
        shadow = ckpt["ema"]["shadow"]
        if not isinstance(shadow, dict):
            raise ValueError(f"Invalid ema.shadow format in checkpoint: {ckpt_path}")
        return shadow, "ema.shadow"

    if all(isinstance(k, str) for k in ckpt.keys()) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, "state_dict"

    raise ValueError(
        "Distill checkpoint must contain one of: {'student': state_dict}, "
        "{'ema': {'shadow': state_dict}}, or be a plain state_dict. "
        f"Got keys={list(ckpt.keys())[:10]} from {ckpt_path}."
    )


def _build_model_from_distill_flexible(*, distill_config_path: str, distill_ckpt_path: str, device: str) -> Kimodo:
    cfg = OmegaConf.load(distill_config_path)
    student_cfg = OmegaConf.to_container(cfg.model.student_denoiser, resolve=True)
    text_encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True)

    denoiser = instantiate_from_dict(student_cfg).to(device)
    text_encoder = instantiate_from_dict(text_encoder_cfg)
    text_encoder.to(device)
    text_encoder.eval()

    loaded_state, source = _resolve_student_state_dict_from_ckpt(distill_ckpt_path)
    missing, unexpected = denoiser.load_state_dict(loaded_state, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys when loading {source} from {distill_ckpt_path}: first 10={unexpected[:10]}"
        )
    if missing:
        raise ValueError(
            f"Missing keys when loading {source} from {distill_ckpt_path}: first 10={missing[:10]}"
        )
    denoiser.eval()

    model = Kimodo(
        denoiser=denoiser,
        text_encoder=text_encoder,
        num_base_steps=int(cfg.model.num_base_steps),
        device=device,
        cfg_type="separated",
    )
    return model


def _validate_g1_model(model, resolved_model: str) -> None:
    skeleton_name = str(getattr(getattr(model, "skeleton", None), "name", "")).lower()
    if skeleton_name.startswith("g1"):
        return

    raise ValueError(
        "This evaluator expects a G1 model because it uses G1 MuJoCo export/check logic. "
        f"Resolved model was: {resolved_model!r}. "
        "Please pass --model Kimodo-G1-RP-v1."
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    all_glob_paths = sorted(Path().glob(args.npz_glob))
    npz_paths = list(all_glob_paths)
    if not npz_paths:
        raise FileNotFoundError(f"No files matched --npz_glob {args.npz_glob!r}")

    if not args.include_unmatched_clips:
        timeline_names = _load_timeline_filenames(args.timeline_jsonl)
        npz_paths = [p for p in npz_paths if p.stem in timeline_names]
        if args.verbose:
            print(
                f"Filtered by timeline: {len(npz_paths)}/{len(all_glob_paths)} "
                f"clips matched timeline annotations."
            )
        if not npz_paths:
            raise FileNotFoundError(
                "After timeline filtering, no clips remained. "
                "Check --npz_glob / --timeline_jsonl, or pass --include_unmatched_clips."
            )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, resolved_model = _load_eval_model(args, device=device)
    _validate_g1_model(model, resolved_model)
    if args.verbose:
        print(f"Loaded model: {resolved_model} on {device}")

    converter = MujocoQposConverter(model.skeleton)
    mj_model = mujoco.MjModel.from_xml_path(args.xml)

    all_clip_rows = []
    all_errors_all_clips: list[float] = []
    all_hand_errors_all_clips: list[float] = []
    failures: list[dict[str, str]] = []

    total = len(npz_paths)
    if args.verbose:
        print(f"Total original clips: {total}")

    with tempfile.TemporaryDirectory(prefix="kimodo_eval_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, npz_path in enumerate(npz_paths, start=1):
            clip_name = npz_path.stem
            print(f"[{idx}/{total}] {clip_name} ...", flush=True)
            try:
                prompt, _ = resolve_overview_prompt(timeline_jsonl=args.timeline_jsonl, clip=npz_path, fallback=None)

                adapter_npz = tmpdir_path / f"{clip_name}.adapter.npz"
                constraints_json_ee = tmpdir_path / f"{clip_name}.constraints.ee.json"
                constraints_json_gen = tmpdir_path / f"{clip_name}.constraints.gen.json"
                constraints_json_fullbody_prefix = tmpdir_path / f"{clip_name}.constraints.fullbody_prefix.json"

                save_custom_motion_ee_pose_npz(npz_path, adapter_npz)
                _write_constraints_from_adapter(
                    adapter_npz_path=adapter_npz,
                    output_json_path=constraints_json_ee,
                    target_fps=float(args.target_fps),
                    keyframe_step=int(args.keyframe_step),
                )
                hard_prefix_enabled = bool(args.hard_project_observed_motion) and int(args.hard_project_prefix_frames) > 0
                if hard_prefix_enabled:
                    _write_fullbody_prefix_constraints_from_npz(
                        npz_path=npz_path,
                        output_json_path=constraints_json_fullbody_prefix,
                        prefix_frames=int(args.hard_project_prefix_frames),
                        target_fps=float(args.target_fps),
                    )
                    _merge_constraints_json(
                        [constraints_json_ee, constraints_json_fullbody_prefix],
                        constraints_json_gen,
                    )
                else:
                    constraints_json_gen.write_text(constraints_json_ee.read_text(encoding="utf-8"), encoding="utf-8")

                adapter_data = np.load(adapter_npz, allow_pickle=False)
                duration = float(adapter_data["root_global_6d"].shape[0]) / float(adapter_data["fps"])
                num_frames, _ = resolve_num_frames(duration, float(model.fps), str(constraints_json_gen))
                constraint_lst = load_constraints_lst(str(constraints_json_gen), model.skeleton)
                first_heading = extract_first_heading_angle_from_npz(npz_path)

                run_means: list[float] = []
                run_maxes: list[float] = []
                run_mins: list[float] = []
                run_num_rows: list[int] = []
                clip_hand_errors_all_runs: list[float] = []
                clip_errors_all_runs: list[float] = []

                for _ in range(int(args.num_runs_per_task)):
                    output = model(
                        prompt,
                        num_frames,
                        constraint_lst=constraint_lst,
                        num_denoising_steps=int(args.diffusion_steps),
                        num_samples=1,
                        multi_prompt=False,
                        post_processing=False,
                        return_numpy=True,
                        first_heading_angle=torch.tensor([first_heading], dtype=torch.float32, device=device),
                        hard_project_observed_motion=bool(hard_prefix_enabled),
                        hard_project_prefix_frames=int(args.hard_project_prefix_frames) if hard_prefix_enabled else 0,
                        progress_bar=_no_progress,
                    )
                    qpos = converter.dict_to_qpos(output, device)
                    qpos_np = np.asarray(qpos, dtype=np.float64)
                    if qpos_np.ndim == 3:
                        qpos_np = qpos_np[0]

                    run_clip_errors, run_clip_hand_errors = _evaluate_qpos_errors(qpos_np, constraints_json_ee, mj_model)
                    run_stats = _safe_stats(run_clip_errors)
                    if run_stats["mean_error_m"] is not None:
                        run_means.append(float(run_stats["mean_error_m"]))
                    if run_stats["max_error_m"] is not None:
                        run_maxes.append(float(run_stats["max_error_m"]))
                    if run_stats["min_error_m"] is not None:
                        run_mins.append(float(run_stats["min_error_m"]))
                    run_num_rows.append(int(run_stats["count"]))
                    clip_errors_all_runs.extend(run_clip_errors)
                    clip_hand_errors_all_runs.extend(run_clip_hand_errors)

                clip_stats = {
                    "count": int(np.mean(run_num_rows)) if run_num_rows else 0,
                    "mean_error_m": _safe_mean(run_means),
                    "max_error_m": _safe_mean(run_maxes),
                    "min_error_m": _safe_mean(run_mins),
                }
                all_clip_rows.append(
                    {
                        "clip": clip_name,
                        "mean_error_m": clip_stats["mean_error_m"],
                        "max_error_m": clip_stats["max_error_m"],
                        "min_error_m": clip_stats["min_error_m"],
                        "num_rows": clip_stats["count"],
                        "num_runs": int(args.num_runs_per_task),
                    }
                )
                all_errors_all_clips.extend(clip_errors_all_runs)
                all_hand_errors_all_clips.extend(clip_hand_errors_all_runs)
                if args.verbose:
                    print(
                        f"  mean={_fmt_metric(clip_stats['mean_error_m'])} "
                        f"max={_fmt_metric(clip_stats['max_error_m'])} "
                        f"min={_fmt_metric(clip_stats['min_error_m'])}"
                    )
            except Exception as exc:
                failures.append({"clip": clip_name, "error": str(exc)})
                if args.verbose:
                    print(f"  FAILED: {exc}")
                else:
                    print(f"[{idx}/{total}] FAILED: {clip_name}")
                if args.fail_fast:
                    raise

    overall_stats = _safe_stats(all_errors_all_clips)
    overall_hand_stats = _safe_stats(all_hand_errors_all_clips)
    per_clip_means = [float(x["mean_error_m"]) for x in all_clip_rows if x.get("mean_error_m") is not None]
    overall_mean_of_task_means = _safe_mean(per_clip_means)
    result = {
        "settings": {
            "npz_glob": args.npz_glob,
            "timeline_jsonl": args.timeline_jsonl,
            "include_unmatched_clips": bool(args.include_unmatched_clips),
            "model": args.model,
            "distill_config": args.distill_config,
            "distill_ckpt": args.distill_ckpt,
            "target_fps": args.target_fps,
            "keyframe_step": args.keyframe_step,
            "diffusion_steps": args.diffusion_steps,
            "num_runs_per_task": int(args.num_runs_per_task),
            "hard_project_observed_motion": bool(args.hard_project_observed_motion),
            "hard_project_prefix_frames": int(args.hard_project_prefix_frames),
            "hard_project_mode": "dense_fullbody_prefix",
        },
        "total_clips": total,
        "success_clips": len(all_clip_rows),
        "failed_clips": len(failures),
        "per_clip": all_clip_rows,
        "overall_all_ee": overall_stats,
        "overall_hands_only": overall_hand_stats,
        "overall_mean_of_task_means_m": overall_mean_of_task_means,
        "failures": failures,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.verbose:
        print("")
        print("Final Summary")
        print(f"success: {len(all_clip_rows)}/{total}, failed: {len(failures)}")
        print(
            "all-ee  => "
            f"mean_error_m: {_fmt_metric(overall_stats['mean_error_m'])}  "
            f"max_error_m: {_fmt_metric(overall_stats['max_error_m'])}  "
            f"min_error_m: {_fmt_metric(overall_stats['min_error_m'])}"
        )
        print(
            "hands   => "
            f"mean_error_m: {_fmt_metric(overall_hand_stats['mean_error_m'])}  "
            f"max_error_m: {_fmt_metric(overall_hand_stats['max_error_m'])}  "
            f"min_error_m: {_fmt_metric(overall_hand_stats['min_error_m'])}"
        )
        print(f"mean(task_mean_error_m): {_fmt_metric(overall_mean_of_task_means)}")
        print(f"Saved summary JSON: {output_path}")
    else:
        print(f"Done: {len(all_clip_rows)}/{total} succeeded, {len(failures)} failed. JSON: {output_path}")


if __name__ == "__main__":
    main()
