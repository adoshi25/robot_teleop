#!/usr/bin/env python3
"""
MuJoCo Hand Tracking Visualizer

This script visualizes hand tracking data from the WebXR hand tracking server
by updating mocap bodies in a MuJoCo simulation.

Usage:
    1. Start the hand tracking server: python tools/start_hand_tracker.py
    2. Connect VR headset and start hand tracking in browser
    3. Run this script: python tools/visualize_hand_tracking.py
"""

import sys
import mujoco
import mujoco.viewer
import requests
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import PandaCalibrator


# Configuration
HAND_TRACKER_URL = "http://localhost:9002/get_hand_data"
MJCF_PATH = Path(__file__).parent.parent / "teleop" / "robots" / "dg5f_dual_panda.mjcf.xml"
UPDATE_FREQUENCY = 60  # Hz

# Mapping from WebXR joint names to MuJoCo mocap body names
JOINT_TO_MOCAP_MAPPING = {
    'left': {
        'wrist': 'left-wrist',
        'thumb-tip': 'left-thumb-tip',
        'index-finger-tip': 'left-index-finger-tip',
        'middle-finger-tip': 'left-middle-finger-tip',
        'ring-finger-tip': 'left-ring-finger-tip',
        'pinky-finger-tip': 'left-pinky-finger-tip',
    },
    'right': {
        'wrist': 'right-wrist',
        'thumb-tip': 'right-thumb-tip',
        'index-finger-tip': 'right-index-finger-tip',
        'middle-finger-tip': 'right-middle-finger-tip',
        'ring-finger-tip': 'right-ring-finger-tip',
        'pinky-finger-tip': 'right-pinky-finger-tip',
    }
}

ROTATION_X90 = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rotate_position_x90(position):
    x, y, z = position
    return np.array([x, -z, y])


def fetch_hand_data():
    """Fetch hand tracking data from the server."""
    try:
        response = requests.get(HAND_TRACKER_URL, timeout=0.1)
        if response.status_code == 200:
            data = response.json()
            if data and 'hands' in data:
                return data['hands']
    except (requests.RequestException, ValueError):
        pass
    return None


def update_mocap_from_hand_data(model, data, hand_data, calibrator):
    """Update MuJoCo mocap bodies with hand tracking data."""
    if not hand_data:
        return

    for hand_name, joints in hand_data.items():
        if hand_name not in JOINT_TO_MOCAP_MAPPING:
            continue

        joint_mapping = JOINT_TO_MOCAP_MAPPING[hand_name]

        # On the first frame of a trajectory, capture all initial positions at once
        if not calibrator.is_initial_captured(hand_name):
            initial_positions = {}
            for joint_name in joint_mapping:
                if joint_name in joints:
                    jd = joints[joint_name]
                    if jd and 'position' in jd:
                        pos = jd['position']
                        initial_positions[joint_name] = rotate_position_x90(
                            np.array([pos['x'], pos['y'], pos['z']])
                        )
            if initial_positions:
                calibrator.capture_initial_positions(hand_name, initial_positions)

        for joint_name, mocap_name in joint_mapping.items():
            if joint_name not in joints:
                continue

            joint_data = joints[joint_name]
            if not joint_data or 'position' not in joint_data:
                continue

            try:
                mocap_id = model.body(mocap_name).mocapid[0]
                if mocap_id < 0:
                    continue
            except KeyError:
                continue

            pos = joint_data['position']
            position = rotate_position_x90(np.array([pos['x'], pos['y'], pos['z']]))
            position = calibrator.transform_position(hand_name, joint_name, position)

            data.mocap_pos[mocap_id] = position

            if 'orientation' in joint_data:
                quat = joint_data['orientation']
                quat_wxyz = np.array([quat['w'], quat['x'], quat['y'], quat['z']])
                data.mocap_quat[mocap_id] = quat_wxyz


def main():
    """Main visualization loop."""
    if not MJCF_PATH.exists():
        print(f"Error: MJCF file not found at {MJCF_PATH}")
        return

    print(f"Loading MuJoCo model from {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    calibrator = PandaCalibrator()
    calibrator.init_simulator(model, data)

    def key_callback(key):
        calibrator.handle_key(key)

    print(f"Model loaded successfully")
    print(f"Connecting to hand tracking server at {HAND_TRACKER_URL}")
    print(f"Update frequency: {UPDATE_FREQUENCY} Hz")
    print("")
    print("Controls:")
    print("  - Press SPACE to start/stop trajectory")
    print("  - Right-click and drag to rotate camera")
    print("  - Scroll to zoom")
    print("  - Press ESC or close window to exit")
    print("")

    test_data = fetch_hand_data()
    if test_data is None:
        print("Warning: Could not connect to hand tracking server.")
        print("Make sure the server is running: python tools/start_hand_tracker.py")
        print("Continuing anyway - will keep trying to connect...")
    else:
        print(f"Successfully connected! Currently tracking: {list(test_data.keys())}")

    print("")

    last_update_time = 0
    update_interval = 1.0 / UPDATE_FREQUENCY

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            current_time = time.time()

            if calibrator.trajectory_active:
                if current_time - last_update_time >= update_interval:
                    hand_data = fetch_hand_data()
                    update_mocap_from_hand_data(model, data, hand_data, calibrator)
                    last_update_time = current_time
                data.ctrl[:] = 0
                mujoco.mj_step(model, data)
            else:
                calibrator.hold_canonical_pose()
                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
