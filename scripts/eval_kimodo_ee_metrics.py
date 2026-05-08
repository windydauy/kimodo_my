#!/usr/bin/env python3
"""Evaluate Kimodo-generated G1 motion against ee-pose constraints (and optional reference motion).

Outputs are written to:
  <output_root>/<run_name>/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np


EE_FIELDS = (
    ("left_hand_pose", "left_wrist_yaw_link", "left_hand"),
    ("right_hand_pose", "right_wrist_yaw_link", "right_hand"),
    ("left_foot_pose", "left_ankle_roll_link", "left_foot"),
    ("right_foot_pose", "right_ankle_roll_link", "right_foot"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_xml = repo_root / "kimodo/assets/skeletons/g1skel34/xml/g1.xml"
    parser = argparse.ArgumentParser(
        description=(
            "Compute ee metrics (mpjpe_g/mpjpe_l/mpjpe_pa) between generated motion and "
            "Kimodo ee constraints, with optional reference CSV comparison."
        )
    )
    parser.add_argument("--generated_csv", required=True, help="Generated MuJoCo qpos CSV path.")
    parser.add_argument("--constraints_json", required=True, help="Kimodo ee-pose constraints JSON path.")
    parser.add_argument("--run_name", required=True, help="Run/clip name, used for output sub-folder.")
    parser.add_argument("--output_root", default="kimodo_eval", help="Output root directory.")
    parser.add_argument("--xml", default=str(default_xml), help="MuJoCo XML path.")
    parser.add_argument("--generated_fps", type=float, default=30.0, help="Generated CSV FPS.")
    parser.add_argument("--constraints_fps", type=float, default=30.0, help="Constraints frame-index FPS.")
    parser.add_argument(
        "--reference_csv",
        default=None,
        help="Optional reference MuJoCo qpos CSV path for full-motion comparison.",
    )
    parser.add_argument("--reference_fps", type=float, default=30.0, help="Reference CSV FPS.")
    parser.add_argument("--pkl", default=None, help="Optional PKL artifact to copy into output folder.")
    parser.add_argument("--mp4", default=None, help="Optional MP4 artifact (file or dir) to copy into output.")
    return parser.parse_args()


def kimodo_xyz_to_mujoco(xyz_k: np.ndarray) -> np.ndarray:
    # Kimodo: y-up, z-forward; MuJoCo: z-up, x-forward
    # [x_m, y_m, z_m] = [z_k, x_k, y_k]
    return np.asarray([xyz_k[2], xyz_k[0], xyz_k[1]], dtype=np.float64)


def load_qpos_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_constraints_ee(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Constraints JSON must be a list: {path}")
    for item in payload:
        if item.get("type") == "ee-pose":
            return item
    raise ValueError(f"No ee-pose item found in constraints: {path}")


def forward_ee_positions(model: mujoco.MjModel, qpos: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    data = mujoco.MjData(model)
    body_ids = []
    joint_names = []
    for _, body_name, out_name in EE_FIELDS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found in XML: {body_name}")
        body_ids.append(body_id)
        joint_names.append(out_name)

    out = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float64)
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            out[i, j] = data.xpos[bid]
    return out, joint_names


def build_constraints_keyframes(ee_item: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    frame_indices = np.asarray(ee_item.get("frame_indices", []), dtype=np.int64)
    if frame_indices.size == 0:
        raise ValueError("ee-pose constraint has empty frame_indices.")

    names: List[str] = []
    coords = []
    for field, _, out_name in EE_FIELDS:
        arr = ee_item.get(field, None)
        if arr is None or len(arr) == 0:
            raise ValueError(f"Constraint field missing/empty: {field}")
        if len(arr) != len(frame_indices):
            raise ValueError(f"Constraint field length mismatch: {field} vs frame_indices")
        xyz_k = np.asarray(arr, dtype=np.float64)[:, :3]
        xyz_m = np.stack([kimodo_xyz_to_mujoco(v) for v in xyz_k], axis=0)
        names.append(out_name)
        coords.append(xyz_m)

    ref = np.stack(coords, axis=1)  # [K, J, 3]
    return frame_indices, ref, names


def align_by_time_nn(
    src_xyz: np.ndarray,
    src_fps: float,
    tgt_frame_indices: np.ndarray,
    tgt_fps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t = tgt_frame_indices.astype(np.float64) / float(tgt_fps)
    src_idx = np.rint(t * float(src_fps)).astype(np.int64)
    src_idx = np.clip(src_idx, 0, src_xyz.shape[0] - 1)
    return src_xyz[src_idx], src_idx


def interpolate_reference_at_times(
    ref_times: np.ndarray,
    ref_xyz: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    # ref_xyz: [N, J, 3]
    j = ref_xyz.shape[1]
    out = np.zeros((query_times.shape[0], j, 3), dtype=np.float64)
    for ji in range(j):
        for ai in range(3):
            out[:, ji, ai] = np.interp(query_times, ref_times, ref_xyz[:, ji, ai])
    return out


def _similarity_align_frame(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # pred/ref: [J, 3], aligns pred to ref with similarity transform.
    pred_mean = pred.mean(axis=0, keepdims=True)
    ref_mean = ref.mean(axis=0, keepdims=True)
    pred_c = pred - pred_mean
    ref_c = ref - ref_mean

    denom = float((pred_c**2).sum())
    if denom < 1e-12:
        return np.repeat(ref_mean, pred.shape[0], axis=0)

    h = pred_c.T @ ref_c
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    scale = float(s.sum() / denom)
    aligned = scale * (pred_c @ r) + ref_mean
    return aligned


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict:
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected [T,J,3], got {pred.shape}")

    err_g = np.linalg.norm(pred - ref, axis=-1)  # [T, J]
    mpjpe_g_frame = err_g.mean(axis=1)

    pred_local = pred - pred.mean(axis=1, keepdims=True)
    ref_local = ref - ref.mean(axis=1, keepdims=True)
    err_l = np.linalg.norm(pred_local - ref_local, axis=-1)
    mpjpe_l_frame = err_l.mean(axis=1)

    aligned = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        aligned[i] = _similarity_align_frame(pred[i], ref[i])
    err_pa = np.linalg.norm(aligned - ref, axis=-1)
    mpjpe_pa_frame = err_pa.mean(axis=1)

    out = {
        "frame_count": int(pred.shape[0]),
        "joint_count": int(pred.shape[1]),
        "mpjpe_g_per_frame_m": mpjpe_g_frame,
        "mpjpe_l_per_frame_m": mpjpe_l_frame,
        "mpjpe_pa_per_frame_m": mpjpe_pa_frame,
        "ee_err_per_frame_m": err_g,
    }
    return out


def summarize_metrics(metrics: Dict, joint_names: List[str]) -> Dict:
    g = metrics["mpjpe_g_per_frame_m"]
    l = metrics["mpjpe_l_per_frame_m"]
    pa = metrics["mpjpe_pa_per_frame_m"]
    ee_err = metrics["ee_err_per_frame_m"]  # [T, J]

    ee_mean = {joint_names[j]: float(ee_err[:, j].mean()) for j in range(ee_err.shape[1])}

    summary = {
        "frame_count": int(metrics["frame_count"]),
        "joint_count": int(metrics["joint_count"]),
        "mpjpe_g_mean_m": float(g.mean()),
        "mpjpe_l_mean_m": float(l.mean()),
        "mpjpe_pa_mean_m": float(pa.mean()),
        "mpjpe_g_mean_mm": float(g.mean() * 1000.0),
        "mpjpe_l_mean_mm": float(l.mean() * 1000.0),
        "mpjpe_pa_mean_mm": float(pa.mean() * 1000.0),
        "mpjpe_g_max_m": float(g.max()),
        "mpjpe_l_max_m": float(l.max()),
        "mpjpe_pa_max_m": float(pa.max()),
        "mpjpe_g_min_m": float(g.min()),
        "mpjpe_l_min_m": float(l.min()),
        "mpjpe_pa_min_m": float(pa.min()),
        "ee_mean_error_m": ee_mean,
        "ee_mean_error_mm": {k: float(v * 1000.0) for k, v in ee_mean.items()},
    }
    return summary


def write_frame_csv(
    path: Path,
    frame_idx: np.ndarray,
    time_sec: np.ndarray,
    metrics: Dict,
    joint_names: List[str],
    extra_cols: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ee = metrics["ee_err_per_frame_m"]
    headers = ["frame_index", "time_sec", "mpjpe_g_m", "mpjpe_l_m", "mpjpe_pa_m"]
    for name in joint_names:
        headers.append(f"{name}_err_m")
    if extra_cols:
        headers.extend(extra_cols.keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(len(frame_idx)):
            row = [
                int(frame_idx[i]),
                float(time_sec[i]),
                float(metrics["mpjpe_g_per_frame_m"][i]),
                float(metrics["mpjpe_l_per_frame_m"][i]),
                float(metrics["mpjpe_pa_per_frame_m"][i]),
            ]
            row.extend(float(v) for v in ee[i])
            if extra_cols:
                row.extend(float(extra_cols[k][i]) for k in extra_cols.keys())
            w.writerow(row)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_optional_artifact(path_like: Optional[str], dest_dir: Path) -> List[str]:
    copied = []
    if not path_like:
        return copied
    src = Path(path_like)
    if not src.exists():
        return copied

    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
        return copied

    # Directory: copy all mp4 files inside recursively.
    for mp4 in sorted(src.rglob("*.mp4")):
        rel = mp4.relative_to(src)
        dst = dest_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mp4, dst)
        copied.append(str(dst))
    return copied


def main() -> None:
    args = parse_args()
    generated_csv = Path(args.generated_csv).resolve()
    constraints_json = Path(args.constraints_json).resolve()
    xml_path = Path(args.xml).resolve()

    out_dir = Path(args.output_root).resolve() / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not generated_csv.exists():
        raise FileNotFoundError(f"generated_csv not found: {generated_csv}")
    if not constraints_json.exists():
        raise FileNotFoundError(f"constraints_json not found: {constraints_json}")
    if not xml_path.exists():
        raise FileNotFoundError(f"xml not found: {xml_path}")

    qpos_gen = load_qpos_csv(generated_csv)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gen_xyz, joint_names = forward_ee_positions(model, qpos_gen)

    ee_item = load_constraints_ee(constraints_json)
    c_frames, c_xyz, c_joint_names = build_constraints_keyframes(ee_item)
    if c_joint_names != joint_names:
        raise ValueError(f"Joint order mismatch: constraints={c_joint_names}, model={joint_names}")

    # A) Generated vs sparse constraints keyframes (time-aligned, nearest generated frame).
    pred_kf, mapped_gen_idx = align_by_time_nn(
        src_xyz=gen_xyz,
        src_fps=float(args.generated_fps),
        tgt_frame_indices=c_frames,
        tgt_fps=float(args.constraints_fps),
    )
    m_kf = compute_metrics(pred_kf, c_xyz)
    s_kf = summarize_metrics(m_kf, joint_names)

    kf_time = c_frames.astype(np.float64) / float(args.constraints_fps)
    write_frame_csv(
        out_dir / "errors_vs_constraints_keyframes.csv",
        frame_idx=c_frames,
        time_sec=kf_time,
        metrics=m_kf,
        joint_names=joint_names,
        extra_cols={"mapped_generated_frame": mapped_gen_idx.astype(np.float64)},
    )
    save_json(out_dir / "errors_vs_constraints_keyframes_summary.json", s_kf)

    # B) Generated vs dense interpolation from sparse constraints, at generated timeline.
    t_constraints = c_frames.astype(np.float64) / float(args.constraints_fps)
    t_gen = np.arange(gen_xyz.shape[0], dtype=np.float64) / float(args.generated_fps)
    ref_dense = interpolate_reference_at_times(t_constraints, c_xyz, t_gen)
    m_dense = compute_metrics(gen_xyz, ref_dense)
    s_dense = summarize_metrics(m_dense, joint_names)

    write_frame_csv(
        out_dir / "errors_vs_constraints_dense_interp.csv",
        frame_idx=np.arange(gen_xyz.shape[0], dtype=np.int64),
        time_sec=t_gen,
        metrics=m_dense,
        joint_names=joint_names,
    )
    save_json(out_dir / "errors_vs_constraints_dense_interp_summary.json", s_dense)

    summary = {
        "run_name": args.run_name,
        "generated_csv": str(generated_csv),
        "constraints_json": str(constraints_json),
        "xml": str(xml_path),
        "generated_frames": int(gen_xyz.shape[0]),
        "generated_fps": float(args.generated_fps),
        "constraints_keyframes": int(c_frames.shape[0]),
        "constraints_fps": float(args.constraints_fps),
        "metrics": {
            "vs_constraints_keyframes": s_kf,
            "vs_constraints_dense_interp": s_dense,
        },
    }

    # C) Optional: generated vs full reference motion.
    if args.reference_csv:
        ref_csv = Path(args.reference_csv).resolve()
        if not ref_csv.exists():
            raise FileNotFoundError(f"reference_csv not found: {ref_csv}")
        qpos_ref = load_qpos_csv(ref_csv)
        ref_xyz_raw, ref_joint_names = forward_ee_positions(model, qpos_ref)
        if ref_joint_names != joint_names:
            raise ValueError(f"Joint order mismatch: reference={ref_joint_names}, model={joint_names}")

        t_ref = np.arange(ref_xyz_raw.shape[0], dtype=np.float64) / float(args.reference_fps)
        ref_xyz_on_gen = interpolate_reference_at_times(t_ref, ref_xyz_raw, t_gen)
        m_ref = compute_metrics(gen_xyz, ref_xyz_on_gen)
        s_ref = summarize_metrics(m_ref, joint_names)

        write_frame_csv(
            out_dir / "errors_vs_reference_full.csv",
            frame_idx=np.arange(gen_xyz.shape[0], dtype=np.int64),
            time_sec=t_gen,
            metrics=m_ref,
            joint_names=joint_names,
        )
        save_json(out_dir / "errors_vs_reference_full_summary.json", s_ref)
        summary["reference_csv"] = str(ref_csv)
        summary["reference_fps"] = float(args.reference_fps)
        summary["metrics"]["vs_reference_full"] = s_ref

    artifacts_dir = out_dir / "artifacts"
    copied = {
        "pkl": copy_optional_artifact(args.pkl, artifacts_dir / "pkl"),
        "mp4": copy_optional_artifact(args.mp4, artifacts_dir / "mp4"),
    }
    summary["copied_artifacts"] = copied

    save_json(out_dir / "summary.json", summary)

    print(f"[OK] Evaluation finished: {out_dir}")
    print(f"  - constraints keyframes mpjpe_g/mm: {s_kf['mpjpe_g_mean_mm']:.3f}")
    print(f"  - dense interp   mpjpe_g/mm: {s_dense['mpjpe_g_mean_mm']:.3f}")
    if "vs_reference_full" in summary["metrics"]:
        print(
            "  - reference full mpjpe_g/mm: "
            f"{summary['metrics']['vs_reference_full']['mpjpe_g_mean_mm']:.3f}"
        )


if __name__ == "__main__":
    main()
