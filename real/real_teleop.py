#!/usr/bin/env python3
"""
Minimal script to execute a sequence of joint positions on the bimanual Franka robot.

Usage:
    python test.py --trajectory path/to/trajectory.npy --hz 2.0
    python test.py --hand left   # test left arm+hand only
    python test.py --hand right  # test right arm+hand only

Trajectory format: (num_steps, 54) numpy array where:
    - [0:7]   = left arm joints
    - [7:27]  = left hand joints  
    - [27:34] = right arm joints
    - [34:54] = right hand joints
"""

import argparse
import sys
import os
from queue import Queue, Empty

import pyrallis
from pynput import keyboard
# Add tesollo-control to path
sys.path.insert(0, "/scr/satvik/tesollo-control")

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from threading import Thread

from environment.bimanual_franka_env import BimanualFrankaEnv
from environment.bimanual_franka_env import BimanualFrankaEnvConfig
from common_utils import FreqGuard
from teleop_client import TeleopClient
import time
from teleop_config import TeleopConfig

# Global queue for keyboard events
keyboard_queue = Queue()


def setup_keyboard_listener():
    """Set up pynput keyboard listener to capture 'a', 'b', and 'c' key presses."""
    def on_press(key):
        try:
            if key.char == 'a':
                keyboard_queue.put_nowait('a')
            elif key.char == 'b':
                keyboard_queue.put_nowait('b')
            elif key.char == 'c':
                keyboard_queue.put_nowait('c')
        except AttributeError:
            pass  # Special key, ignore

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def main():
    """
    Execute a joint trajectory on the robot.
    
    Args:
        trajectory_path: Path to .npy file with shape (num_steps, 54), or None for all zeros
        hz: Control frequency in Hz
        hand: "left", "right", or "both" - which side(s) to command
    """
    # Initialize ROS
    cfg = "/scr/satvik/robot_teleop/real/bimanual_franka_env.yaml"
    teleop_cfg_path = "/scr/satvik/robot_teleop/real/teleop.yaml"
    teleop_cfg = pyrallis.load(TeleopConfig, open(teleop_cfg_path, "r"))
    rclpy.init()
    
    # Initialize environment
    print(f"Loading environment config from: {cfg}")
    env = BimanualFrankaEnv(config_path=cfg)
    env.reset()
    
    # Load env config to get home hand joints
    env_cfg = pyrallis.load(BimanualFrankaEnvConfig, open(cfg, "r"))
    
    # Initialize hands to home positions
    left_home_hand = np.array(env_cfg.left_home_hand_joints) if env_cfg.left_home_hand_joints and env.use_left_hand else None
    right_home_hand = np.array(env_cfg.right_home_hand_joints) if env_cfg.right_home_hand_joints and env.use_right_hand else None
    
    if left_home_hand is not None or right_home_hand is not None:
        env.initialize_to_first_target(
            left_arm=None,
            right_arm=None,
            left_hand=left_home_hand,
            right_hand=right_home_hand
        )
    
    # Spin the node in a separate thread
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    input("Press Enter to start execution...")
    tracker = TeleopClient()

    while not tracker.is_ready():
        time.sleep(1)
        print("Waiting for teleop tracker to be ready...")
        continue

    # Start keyboard listener for interrupt keys (a, b, c)
    listener = setup_keyboard_listener()
    try:
        # Execute trajectory
        for i in range(teleop_cfg.max_horizon):
            # Check for interrupt key (a, b, or c)
            try:
                key = keyboard_queue.get_nowait()
                print(f"\nInterrupt key '{key}' pressed, stopping execution.")
                break
            except Empty:
                pass

            action = tracker.get_qpos(order="genesis")
            print(action)
            action = {
                'left_arm': action[:7],
                'left_hand': action[7:27],
                'right_arm': action[27:34],
                'right_hand': action[34:54],
            }
            env.step(
                left_arm=action['left_arm'],
                right_arm=action['right_arm'],
                left_hand=action['left_hand'],
                right_hand=action['right_hand'],
            )
    finally:
        listener.stop()

    tracker.stop()
    print("Done!")
    
    # Cleanup
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=str, default=None, help="Path to .npy trajectory file (default: all zeros)")
    parser.add_argument("--hz", type=float, default=2.0, help="Control frequency (Hz)")
    parser.add_argument("--hand", type=str, choices=("left", "right", "both"), default="both",
                        help="Which side to command: left, right, or both (default: both)")
    args = parser.parse_args()
    
    main()