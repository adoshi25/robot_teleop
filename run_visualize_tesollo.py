#!/usr/bin/env python3
"""Run the hand-tracking visualizer with the tesollo_hand_2 scene. Use from repo root."""
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TESOLLO_DIR = REPO_ROOT / "teleop" / "robots" / "tesollo_hand"
MJCF = TESOLLO_DIR / "robot_scene_combined.xml"
VISUALIZER = REPO_ROOT / "teleop" / "visualize_hand_tracking.py"

if not MJCF.exists():
    print(f"Error: {MJCF} not found.", file=sys.stderr)
    sys.exit(1)

os.chdir(TESOLLO_DIR)
os.environ["CHIMERA_MJCF_PATH"] = str(MJCF.resolve())

# Pass-through of extra CLI args to the visualizer so you can toggle
# features like collision checking when running this helper script.
EXTRA_ARGS = sys.argv[1:]

def get_mjpython():
    if sys.platform != "darwin":
        return None
    exe_dir = Path(sys.executable).resolve().parent
    cand = exe_dir / "mjpython"
    if cand.exists():
        return str(cand)
    return shutil.which("mjpython")

mjpython = get_mjpython()
if mjpython:
    args = [
        mjpython,
        str(VISUALIZER),
        "--mjcf-path",
        str(MJCF.resolve()),
        "--no-stream-frames",
    ]
    args.extend(EXTRA_ARGS)
    os.execv(mjpython, args)
else:
    if sys.platform == "darwin":
        print("On macOS the viewer needs mjpython. Install and run:", file=sys.stderr)
        print("  pip install mujoco", file=sys.stderr)
        print("  venv/bin/mjpython run_visualize_tesollo.py", file=sys.stderr)
        sys.exit(1)
args = [sys.executable, str(VISUALIZER)]
args.extend(EXTRA_ARGS)
os.execv(sys.executable, args)
