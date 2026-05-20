#!/usr/bin/env python3
"""Viser comparison: raw HUMI end-effector trajectory vs generated G1 robot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from kimodo.assets import skeleton_asset_path
from kimodo.skeleton import G1Skeleton34
from kimodo.viz.scene import Character
from scripts.humi_json_to_ee_pose_npz import canonicalize_humi_positions


MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

HUMI_FIELDS = (
    "root_pose",
    "left_hand_pose",
    "right_hand_pose",
    "left_foot_pose",
    "right_foot_pose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Viser comparison: HUMI keypoint ghost vs generated G1 robot.")
    parser.add_argument("--original-json", required=True, help="Original HUMI recording JSON.")
    parser.add_argument(
        "--original-ik-json",
        default=None,
        help="Optional HUMI ik_recomputed JSON. If provided, draw its full-body q trajectory as GT skeleton.",
    )
    parser.add_argument("--generated-npz", required=True, help="Generated Kimodo NPZ fallback.")
    parser.add_argument("--generated-csv", required=True, help="Generated MuJoCo qpos CSV.")
    parser.add_argument("--generated-fps", type=float, default=30.0, help="FPS for generated CSV playback.")
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=None,
        help="Optional cap for displayed original/generated trajectories.",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument("--original-color", type=str, default="0,200,0", help="Original ghost RGB, e.g. 0,200,0.")
    parser.add_argument("--generated-color", type=str, default="255,120,0", help="Generated robot RGB.")
    parser.add_argument("--ghost-opacity", type=float, default=0.35, help="Opacity for HUMI ghost markers.")
    return parser.parse_args()


def _parse_rgb(text: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Invalid RGB string: {text}")
    vals = [max(0, min(255, v)) for v in vals]
    return (vals[0], vals[1], vals[2])


def _xyz_mujoco_to_kimodo(xyz: np.ndarray) -> np.ndarray:
    return np.asarray([xyz[1], xyz[2], xyz[0]], dtype=np.float64)


def _rot_mujoco_to_kimodo(rot: np.ndarray) -> np.ndarray:
    return MUJOCO_TO_KIMODO @ rot @ MUJOCO_TO_KIMODO.T


def _rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    h = src_centered.T @ dst_centered
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = dst_centroid - rot @ src_centroid
    return rot, trans


def _load_humi_pose(json_path: Path) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    episode = data.get("episode")
    if not isinstance(episode, list) or not episode:
        raise ValueError(f"Expected non-empty episode in {json_path}.")

    timestamps = np.asarray([float(frame["timestamp"]) for frame in episode], dtype=np.float64)
    dt = np.diff(timestamps)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError("Cannot infer HUMI FPS from timestamps.")
    fps = float(1.0 / np.mean(dt))

    pos = np.zeros((len(episode), len(HUMI_FIELDS), 3), dtype=np.float64)
    rot = np.zeros((len(episode), len(HUMI_FIELDS), 3, 3), dtype=np.float64)
    for t, frame in enumerate(episode):
        for j, field in enumerate(HUMI_FIELDS):
            pose = frame[field]
            xyz_m = np.asarray(pose["position"], dtype=np.float64)
            quat_wxyz = np.asarray(pose["quaternion_wxyz"], dtype=np.float64)
            rot_m = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
            pos[t, j] = xyz_m
            rot[t, j] = _rot_mujoco_to_kimodo(rot_m)

    pos, _origin_xy, _floor_z = canonicalize_humi_positions(pos)
    pos = np.asarray([[_xyz_mujoco_to_kimodo(p) for p in frame] for frame in pos], dtype=np.float64)
    return pos, rot, fps, timestamps


def _load_ik_fullbody_pose(
    ik_json_path: Path,
) -> tuple[np.ndarray, float, str]:
    with ik_json_path.open("r", encoding="utf-8") as f:
        data_json = json.load(f)
    episode = data_json.get("episode")
    if not isinstance(episode, list) or not episode:
        raise ValueError(f"Expected non-empty episode in {ik_json_path}.")

    mjcf_path = str(data_json.get("mjcf_path", "")).strip()
    if not mjcf_path:
        raise ValueError(f"IK JSON missing mjcf_path: {ik_json_path}")
    if not Path(mjcf_path).exists():
        raise FileNotFoundError(f"IK mjcf_path not found: {mjcf_path}")

    timestamps = np.asarray([float(frame["timestamp"]) for frame in episode], dtype=np.float64)
    dt = np.diff(timestamps)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError("Cannot infer IK FPS from timestamps.")
    fps = float(1.0 / np.mean(dt))

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    q0 = np.asarray(episode[0]["q"], dtype=np.float64)
    if q0.shape[0] != model.nq:
        raise ValueError(f"IK q length {q0.shape[0]} does not match MuJoCo model.nq {model.nq}: {mjcf_path}")

    body_names = [
        "pelvis",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_roll_link",
    ]
    body_ids = []
    kept_names = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_ids.append(bid)
            kept_names.append(name)
    if not body_ids:
        raise ValueError(f"No expected G1 body names found in IK model: {mjcf_path}")

    pos = np.zeros((len(episode), len(body_ids), 3), dtype=np.float64)
    align_fields = ("root_pose", "left_hand_pose", "right_hand_pose", "left_foot_pose", "right_foot_pose")
    for t, frame in enumerate(episode):
        data.qpos[:] = np.asarray(frame["q"], dtype=np.float64)
        mujoco.mj_forward(model, data)
        src = np.asarray(
            [frame["transformed_target"][field]["position"] for field in align_fields],
            dtype=np.float64,
        )
        dst = np.asarray([frame[field]["position"] for field in align_fields], dtype=np.float64)
        align_rot, align_trans = _rigid_transform(src, dst)
        for j, bid in enumerate(body_ids):
            p_ik = np.asarray(data.xpos[bid], dtype=np.float64)
            p_raw = align_rot @ p_ik + align_trans
            pos[t, j] = _xyz_mujoco_to_kimodo(p_raw)

    return pos, fps, ",".join(kept_names)


def _joint_to_body_name(joint_name: str) -> str | None:
    if joint_name == "pelvis_skel":
        return "pelvis"
    if joint_name == "waist_pitch_skel":
        return "torso_link"
    if joint_name.endswith("_skel"):
        return joint_name[: -len("_skel")] + "_link"
    return None


def _qpos_to_g1_pose_in_kimodo(qpos: np.ndarray, skeleton: G1Skeleton34, xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    n = qpos.shape[0]
    j = skeleton.nbjoints
    joints_pos = np.zeros((n, j, 3), dtype=np.float64)
    joints_rot = np.zeros((n, j, 3, 3), dtype=np.float64)

    body_id_cache = {}
    for name in skeleton.bone_order_names:
        bname = _joint_to_body_name(name)
        if bname is None:
            body_id_cache[name] = -1
            continue
        try:
            body_id_cache[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        except Exception:
            body_id_cache[name] = -1

    for t in range(n):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for idx, jname in enumerate(skeleton.bone_order_names):
            bid = body_id_cache[jname]
            if bid >= 0:
                p_m = np.asarray(data.xpos[bid], dtype=np.float64)
                r_m = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
                joints_pos[t, idx] = _xyz_mujoco_to_kimodo(p_m)
                joints_rot[t, idx] = _rot_mujoco_to_kimodo(r_m)
            else:
                parent = int(skeleton.joint_parents[idx].item())
                if parent >= 0:
                    joints_pos[t, idx] = joints_pos[t, parent]
                    joints_rot[t, idx] = joints_rot[t, parent]
                else:
                    joints_pos[t, idx] = np.zeros(3, dtype=np.float64)
                    joints_rot[t, idx] = np.eye(3, dtype=np.float64)

        min_y = joints_pos[t, :, 1].min()
        joints_pos[t, :, 1] -= min_y

    return joints_pos, joints_rot


def _load_generated_pose(args: argparse.Namespace, skeleton: G1Skeleton34, xml_path: str):
    gen_csv = Path(args.generated_csv)
    if gen_csv.exists():
        qpos = np.loadtxt(str(gen_csv), delimiter=",")
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        pos, rot = _qpos_to_g1_pose_in_kimodo(qpos, skeleton, xml_path)
        return pos, rot, float(args.generated_fps), f"generated-csv: {gen_csv}"

    gen_npz = Path(args.generated_npz)
    if not gen_npz.exists():
        raise FileNotFoundError("Neither generated csv nor generated npz exists.")
    data = np.load(str(gen_npz), allow_pickle=True)
    pos = np.asarray(data["posed_joints"], dtype=np.float64)
    rot = np.asarray(data["global_rot_mats"], dtype=np.float64)
    if pos.ndim == 4:
        pos = pos[0]
    if rot.ndim == 5:
        rot = rot[0]
    return pos, rot, float(args.generated_fps), f"generated-npz: {gen_npz}"


def main() -> None:
    args = parse_args()
    original_color = _parse_rgb(args.original_color)
    generated_color = _parse_rgb(args.generated_color)
    xml_path = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    skeleton = G1Skeleton34()

    original_pos, _original_rot, original_fps, _original_timestamps = _load_humi_pose(Path(args.original_json))
    if args.max_duration_sec is not None:
        keep = max(1, min(original_pos.shape[0], int(np.floor(float(args.max_duration_sec) * original_fps))))
        original_pos = original_pos[:keep]
    ik_pos = None
    ik_fps = None
    ik_src = None
    if args.original_ik_json:
        ik_pos, ik_fps, ik_src = _load_ik_fullbody_pose(Path(args.original_ik_json))
        if args.max_duration_sec is not None:
            keep = max(1, min(ik_pos.shape[0], int(np.floor(float(args.max_duration_sec) * ik_fps))))
            ik_pos = ik_pos[:keep]
    generated_pos, generated_rot, generated_fps, generated_src = _load_generated_pose(args, skeleton, xml_path)
    if args.max_duration_sec is not None:
        keep = max(1, min(generated_pos.shape[0], int(np.floor(float(args.max_duration_sec) * generated_fps))))
        generated_pos = generated_pos[:keep]
        generated_rot = generated_rot[:keep]

    original_len = original_pos.shape[0]
    ik_len = ik_pos.shape[0] if ik_pos is not None else 0
    generated_len = generated_pos.shape[0]
    duration_sec = max(
        original_len / original_fps,
        generated_len / generated_fps,
        (ik_len / ik_fps) if ik_pos is not None and ik_fps else 0.0,
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")
    server.gui.add_markdown(
        f"### HUMI Overlay Viewer\n"
        f"- original EE: `{args.original_json}` (keypoint ghost)\n"
        f"- original IK: `{args.original_ik_json or 'not provided'}`"
        f"{' (full-body skeleton)' if ik_src else ''}\n"
        f"- {generated_src} (solid robot)\n"
        f"- open: `http://127.0.0.1:{args.port}`"
    )
    play = server.gui.add_checkbox("Play", initial_value=True)
    speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    t_slider = server.gui.add_slider("Time (s)", min=0.0, max=max(duration_sec, 0.01), step=0.001, initial_value=0.0)

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.035)
    markers = []
    for i, field in enumerate(HUMI_FIELDS):
        marker = server.scene.add_mesh_simple(
            name=f"/humi_ghost/{field}",
            vertices=sphere.vertices,
            faces=sphere.faces,
            color=original_color,
        )
        marker.opacity = float(args.ghost_opacity)
        markers.append(marker)

    lines = server.scene.add_line_segments(
        name="/humi_ghost/segments",
        points=np.zeros((4, 2, 3), dtype=np.float64),
        colors=original_color,
        line_width=4.0,
    )
    ik_markers = []
    ik_lines = None
    if ik_pos is not None:
        small_sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.025)
        for j in range(ik_pos.shape[1]):
            marker = server.scene.add_mesh_simple(
                name=f"/humi_ik_gt/body_{j:02d}",
                vertices=small_sphere.vertices,
                faces=small_sphere.faces,
                color=original_color,
            )
            marker.opacity = float(args.ghost_opacity)
            ik_markers.append(marker)
        ik_lines = server.scene.add_line_segments(
            name="/humi_ik_gt/segments",
            points=np.zeros((max(ik_pos.shape[1] - 1, 1), 2, 3), dtype=np.float64),
            colors=original_color,
            line_width=3.0,
        )

    robot = Character(
        "generated_robot",
        server,
        skeleton,
        create_skeleton_mesh=False,
        create_skinned_mesh=True,
        visible_skinned_mesh=True,
        mesh_mode="g1_stl",
    )
    robot.set_skinned_mesh_opacity(1.0)
    robot.set_skinned_mesh_wireframe(False)
    if robot.g1_mesh_rig is not None:
        robot.g1_mesh_rig.set_color(generated_color)

    start_wall = time.time()
    base_t = 0.0

    while True:
        if play.value:
            elapsed = (time.time() - start_wall) * speed.value
            cur_t = (base_t + elapsed) % duration_sec
            t_slider.value = cur_t
        else:
            base_t = t_slider.value
            start_wall = time.time()
            cur_t = t_slider.value

        i0 = min(int(round(cur_t * original_fps)), original_len - 1)
        iik = min(int(round(cur_t * ik_fps)), ik_len - 1) if ik_pos is not None and ik_fps else 0
        i1 = min(int(round(cur_t * generated_fps)), generated_len - 1)

        humi_points = original_pos[i0]
        for marker, point in zip(markers, humi_points):
            marker.position = point

        line_points = np.zeros((4, 2, 3), dtype=np.float64)
        line_points[:, 0, :] = humi_points[0]
        line_points[:, 1, :] = humi_points[1:]
        lines.points = line_points

        if ik_pos is not None and ik_lines is not None:
            body_points = ik_pos[iik]
            for marker, point in zip(ik_markers, body_points):
                marker.position = point
            segs = np.zeros((max(body_points.shape[0] - 1, 1), 2, 3), dtype=np.float64)
            if body_points.shape[0] >= 2:
                segs[:, 0, :] = body_points[:-1]
                segs[:, 1, :] = body_points[1:]
            else:
                segs[0, :, :] = body_points[0]
            ik_lines.points = segs

        robot.set_pose(
            torch.from_numpy(generated_pos[i1]).float(),
            torch.from_numpy(generated_rot[i1]).float(),
        )
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
