# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a Kimodo .npz motion file to an MP4 video (headless, no display needed)."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Parent tables derived from bone_order_names_with_parents in skeleton/definitions.py.
# -1 means root (no parent).

SKELETON_DEFS = {
    77: {  # SOMASkeleton77
        "parents": [
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 7,   # Hips..RightEye
            7, 3, 11, 12, 13,                   # LeftShoulder..LeftHand
            14, 15, 16, 17,                     # LeftHandThumb1..End
            14, 19, 20, 21, 22,                 # LeftHandIndex1..End
            14, 24, 25, 26, 27,                 # LeftHandMiddle1..End
            14, 29, 30, 31, 32,                 # LeftHandRing1..End
            14, 34, 35, 36, 37,                 # LeftHandPinky1..End
            3, 39, 40, 41,                      # RightShoulder..RightHand
            42, 43, 44, 45,                     # RightHandThumb1..End
            42, 47, 48, 49, 50,                 # RightHandIndex1..End
            42, 52, 53, 54, 55,                 # RightHandMiddle1..End
            42, 57, 58, 59, 60,                 # RightHandRing1..End
            42, 62, 63, 64, 65,                 # RightHandPinky1..End
            0, 67, 68, 69, 70,                  # LeftLeg..LeftToeEnd
            0, 72, 73, 74, 75,                  # RightLeg..RightToeEnd
        ],
        "key_joints": {0: "Hips", 5: "Head", 14: "LHand", 42: "RHand", 68: "LFoot", 73: "RFoot"},
        "finger_toe": set(range(15, 39)) | set(range(43, 67)) | {70, 75},
    },
    30: {  # SOMASkeleton30
        "parents": [
            -1, 0, 1, 2, 3, 4, 5,              # Hips..Head
            6, 6, 6,                             # Jaw, LeftEye, RightEye
            3, 10, 11, 12, 13, 13,              # LeftShoulder..LeftHandMiddleEnd
            3, 16, 17, 18, 19, 19,              # RightShoulder..RightHandMiddleEnd
            0, 22, 23, 24,                       # LeftLeg..LeftToeBase
            0, 26, 27, 28,                       # RightLeg..RightToeBase
        ],
        "key_joints": {0: "Hips", 6: "Head", 13: "LHand", 19: "RHand", 24: "LFoot", 28: "RFoot"},
        "finger_toe": set(),
    },
    34: {  # G1Skeleton34
        "parents": [
            -1,                                  # pelvis
            0, 1, 2, 3, 4, 5, 6,                # left leg chain (hip_pitch..toe_base)
            0, 8, 9, 10, 11, 12, 13,            # right leg chain
            0, 15, 16,                           # torso..head
            16, 18, 19, 20, 21, 22,             # left arm chain (shoulder_pitch..hand_roll)
            16, 24, 25, 26, 27, 28,             # right arm chain
            17, 17,                              # left_logo, right_logo
        ],
        "key_joints": {0: "Pelvis", 17: "Head", 23: "LHand", 29: "RHand", 6: "LFoot", 13: "RFoot"},
        "finger_toe": set(),
    },
}


def get_skeleton_def(num_joints):
    """Look up skeleton definition by joint count, or return a generic fallback."""
    if num_joints in SKELETON_DEFS:
        return SKELETON_DEFS[num_joints]
    return {"parents": None, "key_joints": {0: "Root"}, "finger_toe": set()}


def build_bones(parents, finger_toe, skip_fingers=True):
    """Return list of (child, parent) pairs for bone lines."""
    bones = []
    for child, parent in enumerate(parents):
        if parent == -1:
            continue
        if skip_fingers and child in finger_toe:
            continue
        bones.append((child, parent))
    return bones


def render(npz_path, output_path, fps=30, skip_fingers=True, figsize=(8, 8)):
    data = np.load(npz_path)
    joints = data["posed_joints"]

    # Handle batched (B, T, J, 3) vs unbatched (T, J, 3)
    if joints.ndim == 4:
        joints = joints[0]

    num_frames, num_joints, _ = joints.shape
    print(f"Loaded {npz_path}: {num_frames} frames, {num_joints} joints")

    skel = get_skeleton_def(num_joints)
    parents = skel["parents"]
    key_joints = skel["key_joints"]
    finger_toe = skel["finger_toe"]

    bones = build_bones(parents, finger_toe, skip_fingers) if parents else []

    # Compute bounding box across all frames for stable camera
    all_x, all_y, all_z = joints[:, :, 0], joints[:, :, 2], joints[:, :, 1]
    margin = 0.3
    x_range = (all_x.min() - margin, all_x.max() + margin)
    y_range = (all_y.min() - margin, all_y.max() + margin)
    z_range = (0, max(all_z.max() + margin, 2.0))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()
        j = joints[frame]
        x, y, z = j[:, 0], j[:, 2], j[:, 1]  # swap Y/Z for upright display

        # Draw bones
        for child, parent in bones:
            ax.plot(
                [x[child], x[parent]],
                [y[child], y[parent]],
                [z[child], z[parent]],
                color="steelblue", linewidth=1.5, zorder=1,
            )

        # Draw joints (skip fingers for cleanliness)
        if skip_fingers and finger_toe:
            mask = np.array([i not in finger_toe for i in range(num_joints)])
            ax.scatter(x[mask], y[mask], z[mask], s=8, c="steelblue", zorder=2)
        else:
            ax.scatter(x, y, z, s=5, c="steelblue", zorder=2)

        # Highlight key joints
        for idx, name in key_joints.items():
            if idx < num_joints:
                ax.scatter(x[idx], y[idx], z[idx], s=30, c="red", zorder=3)

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y (up)")
        ax.set_title(f"Frame {frame}/{num_frames-1}")
        ax.view_init(elev=15, azim=-60)

    print(f"Rendering {num_frames} frames to {output_path} ...")
    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps)

    import shutil
    if shutil.which("ffmpeg"):
        ani.save(output_path, writer="ffmpeg", fps=fps, dpi=100)
    else:
        # Fallback to GIF via Pillow
        if output_path.endswith(".mp4"):
            output_path = output_path.replace(".mp4", ".gif")
        print(f"ffmpeg not found, saving as GIF: {output_path}")
        ani.save(output_path, writer="pillow", fps=fps, dpi=100)

    plt.close(fig)
    print(f"Done: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render Kimodo .npz motion to MP4 video")
    parser.add_argument("input", help="Path to .npz file (output of kimodo_gen)")
    parser.add_argument("-o", "--output", default=None, help="Output .mp4 path (default: same name as input)")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--show-fingers", action="store_true", help="Include finger/toe joints")
    args = parser.parse_args()

    output = args.output or args.input.replace(".npz", ".mp4")
    render(args.input, output, fps=args.fps, skip_fingers=not args.show_fingers)


if __name__ == "__main__":
    main()
