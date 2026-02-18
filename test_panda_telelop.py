#!/usr/bin/env python3
"""Test real-time bimanual teleop trajectory processing for dual Panda arms.

Generates non-trivial figure-eight wrist trajectories for both arms,
streams them point-by-point at 60 Hz, and records the result to video.
"""

import time
import numpy as np
from panda_telelop import PandaArmTrajectoryProcessor

# ---------------------------------------------------------------------------
# Generate bimanual wrist trajectories in world frame.
#
# Scene layout (from robot_scene_combined.xml):
#   Left  arm base: (0.7, -0.4, 0)
#   Right arm base: (0.7,  0.4, 0)
#   Table center:   (0.7,  0.0, 0.2)
#
# Each trajectory traces a figure-eight above the table, with the two arms
# sweeping toward the center in complementary phase.
# ---------------------------------------------------------------------------
N = 300
t = np.linspace(0, 2 * np.pi, N)

# Left arm: figure-eight with height variation
#   base-frame center ≈ (0.30, 0.0, 0.48) → well within Panda reach
left_poses = np.column_stack([
    0.7 + 0.30 + 0.10 * np.sin(t),
    -0.4 + 0.18 * np.sin(2 * t),
    0.48 + 0.13 * np.cos(t),
])

# Right arm: complementary figure-eight (phase-shifted)
#   base-frame center ≈ (0.30, 0.0, 0.48)
right_poses = np.column_stack([
    0.7 + 0.30 + 0.10 * np.cos(t),
    0.4 - 0.18 * np.sin(2 * t),
    0.48 + 0.13 * np.sin(t),
])

print(f"Generated {N} wrist poses per arm")
for label, poses in [("left", left_poses), ("right", right_poses)]:
    print(
        f"  {label}  "
        f"x:[{poses[:, 0].min():.3f}, {poses[:, 0].max():.3f}]  "
        f"y:[{poses[:, 1].min():.3f}, {poses[:, 1].max():.3f}]  "
        f"z:[{poses[:, 2].min():.3f}, {poses[:, 2].max():.3f}]"
    )

# ---------------------------------------------------------------------------
# Create processor and warm up IK (triggers JAX JIT compilation)
# ---------------------------------------------------------------------------
control_hz = 60.0
proc = PandaArmTrajectoryProcessor(control_hz=control_hz)

# print("\nWarming up IK solver (JAX JIT)...")
# t0 = time.time()
# proc.add_point(left_poses[0], "left")
# proc.add_point(right_poses[0], "right")
# print(f"  warmup done in {time.time() - t0:.1f}s")

proc.start_trajectory()

# ---------------------------------------------------------------------------
# Stream points one-by-one at control_hz (simulating real-time teleop)
# ---------------------------------------------------------------------------
dt = 1.0 / control_hz
print(f"\nStreaming {N} bimanual poses at {control_hz:.1f} Hz...")
t_start = time.time()
for i in range(N):
    proc.add_point(left_poses[i], "left")
    proc.add_point(right_poses[i], "right")
    elapsed = time.time() - t_start
    target = (i + 1) * dt
    if target > elapsed:
        time.sleep(target - elapsed)

elapsed = time.time() - t_start
print(f"Streaming done: {elapsed:.2f}s ({N / elapsed:.1f} Hz effective)")

for side in ("left", "right"):
    ts, qs = proc.get_trajectory(side)
    print(f"  {side}: {len(qs)} recorded waypoints")

duration = proc.get_trajectory_duration()
print(f"  logical duration: {duration:.2f}s (expected {N / control_hz:.2f}s)")

# ---------------------------------------------------------------------------
# Record optimized trajectory to video and save EE poses
# ---------------------------------------------------------------------------
print("\nRecording trajectory to video...")
result = proc.record_trajectory_to_video(
    video_path="trajectory_video.mp4",
    save_ee_poses_path="ee_poses.npz",
)
for side, qs in result.items():
    print(f"  {side}: {qs.shape[0]} optimized waypoints")
