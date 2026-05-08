#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser
from scipy.spatial.transform import Rotation as R

from kimodo.assets import skeleton_asset_path
from kimodo.skeleton import G1Skeleton34
from kimodo.viz.scene import Character

MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Viser comparison: raw custom-motion ghost vs generated G1 robot.")
    parser.add_argument(
        "--original-npz",
        default="custom_motion/robot-object/sub10_largebox_000_original.npz",
        help="Original raw custom-motion qpos npz.",
    )
    parser.add_argument(
        "--generated-npz",
        default="scripts/pipeline_outputs/sub10_largebox_000/g1_generated.npz",
        help="Generated Kimodo npz path (fallback when --generated-csv is absent).",
    )
    parser.add_argument(
        "--generated-csv",
        default="scripts/pipeline_outputs/sub10_largebox_000/g1_generated.csv",
        help="Generated MuJoCo qpos csv (preferred for 'actual robot' playback).",
    )
    parser.add_argument("--generated-fps", type=float, default=30.0, help="FPS for generated CSV playback.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--original-color",
        type=str,
        default="0,200,0",
        help="Original ghost RGB, e.g. 0,200,0",
    )
    parser.add_argument(
        "--generated-color",
        type=str,
        default="255,120,0",
        help="Generated robot RGB, e.g. 255,120,0",
    )
    parser.add_argument(
        "--ghost-opacity",
        type=float,
        default=0.2,
        help="Opacity for original ghost mesh (0~1).",
    )
    return parser.parse_args()


def _parse_rgb(text: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Invalid RGB string: {text}")
    vals = [max(0, min(255, v)) for v in vals]
    return (vals[0], vals[1], vals[2])


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
                p_k = np.array([p_m[1], p_m[2], p_m[0]], dtype=np.float64)
                r_k = MUJOCO_TO_KIMODO @ r_m @ MUJOCO_TO_KIMODO.T
                joints_pos[t, idx] = p_k
                joints_rot[t, idx] = r_k
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


def _build_qpos_from_custom_motion_npz(npz_path: Path) -> tuple[np.ndarray, float]:
    data = np.load(str(npz_path), allow_pickle=True)
    fps = float(data["fps"])
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.shape[1] < 36:
        raise ValueError(f"Expected raw custom-motion qpos with at least 36 columns, got {tuple(qpos.shape)}.")
    return qpos[:, :36], fps


def _load_generated_pose(args, skeleton: G1Skeleton34, xml_path: str):
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


def main():
    args = parse_args()
    original_color = _parse_rgb(args.original_color)
    generated_color = _parse_rgb(args.generated_color)
    xml_path = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    skeleton = G1Skeleton34()

    original_qpos, original_fps = _build_qpos_from_custom_motion_npz(Path(args.original_npz))
    original_pos, original_rot = _qpos_to_g1_pose_in_kimodo(original_qpos, skeleton, xml_path)

    generated_pos, generated_rot, generated_fps, generated_src = _load_generated_pose(args, skeleton, xml_path)

    original_len = original_pos.shape[0]
    generated_len = generated_pos.shape[0]
    duration_sec = max(original_len / original_fps, generated_len / generated_fps)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")

    server.gui.add_markdown(
        f"### G1 Overlay Viewer\n"
        f"- original: `{args.original_npz}` (ghost)\n"
        f"- {generated_src} (solid robot)\n"
        f"- open: `http://127.0.0.1:{args.port}`"
    )
    play = server.gui.add_checkbox("Play", initial_value=True)
    speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    t_slider = server.gui.add_slider("Time (s)", min=0.0, max=max(duration_sec, 0.01), step=0.001, initial_value=0.0)

    ghost = Character(
        "original_ghost",
        server,
        skeleton,
        create_skeleton_mesh=False,
        create_skinned_mesh=True,
        visible_skinned_mesh=True,
        mesh_mode="g1_stl",
    )
    ghost.set_skinned_mesh_opacity(args.ghost_opacity)
    ghost.set_skinned_mesh_wireframe(True)
    if ghost.g1_mesh_rig is not None:
        ghost.g1_mesh_rig.set_color(original_color)

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
        i1 = min(int(round(cur_t * generated_fps)), generated_len - 1)

        ghost.set_pose(
            torch.from_numpy(original_pos[i0]).float(),
            torch.from_numpy(original_rot[i0]).float(),
        )
        robot.set_pose(
            torch.from_numpy(generated_pos[i1]).float(),
            torch.from_numpy(generated_rot[i1]).float(),
        )
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
