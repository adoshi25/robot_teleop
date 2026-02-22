#!/usr/bin/env python3
"""Sweep each actuator from 0 -> pi -> 0 and record a video."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "teleop" / "robots" / "tesollo_hand" / "robot_scene_combined.xml"
)


def _build_camera(model):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.75, 0.0, 0.35]
    cam.distance = 2.0
    cam.azimuth = 180
    cam.elevation = -30
    return cam


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep each actuator from 0 -> pi -> 0 and record a video."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "view_joints.mp4",
        help="Output video path.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Video FPS.")
    parser.add_argument(
        "--steps-per-half",
        type=int,
        default=60,
        help="Steps for each half of the sweep (0->pi and pi->0).",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=54,
        help="Number of action dimensions to sweep.",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    joint_qpos_indices = []
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            joint_qpos_indices.append(int(model.jnt_qposadr[jid]))

    if len(joint_qpos_indices) < args.num_actions:
        raise ValueError(
            f"Model has {len(joint_qpos_indices)} single-DoF joints, "
            f"expected at least {args.num_actions}."
        )

    joint_qpos_indices = joint_qpos_indices[: args.num_actions]
    data.qpos[joint_qpos_indices] = 0.0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = _build_camera(model)
    frames = []

    steps = int(args.steps_per_half)
    sweep = np.concatenate(
        [
            np.linspace(0.0, np.pi, steps, endpoint=False),
            np.linspace(np.pi, 0.0, steps, endpoint=True),
        ]
    )

    try:
        for action_idx, qpos_idx in enumerate(joint_qpos_indices):
            for val in sweep:
                data.qpos[qpos_idx] = float(val)
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
    finally:
        renderer.close()

    imageio.mimsave(str(args.output), frames, fps=args.fps)
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
