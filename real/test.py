#!/usr/bin/env python3
"""
Minimal script to execute a sequence of joint positions on the bimanual Franka robot.

Usage:
    python test.py --trajectory path/to/trajectory.npy --hz 2.0

Trajectory format: (num_steps, 54) numpy array where:
    - [0:7]   = left arm joints
    - [7:27]  = left hand joints  
    - [27:34] = right arm joints
    - [34:54] = right hand joints
"""

import argparse
import sys
import os

# Add tesollo-control to path
sys.path.insert(0, "/scr/satvik/tesollo-control")

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from threading import Thread

from environment.bimanual_franka_env import BimanualFrankaEnv
from common_utils import FreqGuard

# Default test joint sequence (54-dim)
# [0:7] left arm, [7:27] left hand, [27:34] right arm, [34:54] right hand
DEFAULT_JOINTS = np.array([
    # Left arm (7)
    0.128, -0.738, 0.017, -2.530, 1.341, 1.823, 0.715,
    # Left hand (20)
    -0.400, 0.140, 0.080, 0.100, -0.430, 0.070, 0.440, 0.520, 0.770,
    0.030, -0.810, 0.310, 0.510, -0.330, -0.120, 0.230, -0.290, -1.030,
    -0.560, 1.210,
    # Right arm (7)
    -0.305, -0.792, 0.068, -2.584, -1.353, 2.039, 0.629,
    # Right hand (20)
    0.300, -0.080, -0.070, -0.040, 0.000, -0.040, 0.230, 0.080, 0.280,
    0.000, 0.600, 0.250, 0.690, -0.020, 0.710, -0.020, -0.220, -0.820,
    -0.170, -1.330,
])


def main(trajectory_path: str = None, hz: float = 2.0):
    """
    Execute a joint trajectory on the robot.
    
    Args:
        trajectory_path: Path to .npy file with shape (num_steps, 54), or None for all zeros
        hz: Control frequency in Hz
    """
    # Load trajectory or use default test joints
    if trajectory_path is not None:
        trajectory = np.load(trajectory_path)
        print(f"Loaded trajectory: {trajectory.shape}")
    else:
        # Default: single step with test joint positions
        trajectory = DEFAULT_JOINTS.reshape(1, 54)
        print("No trajectory provided, using default test joints")
    
    # Initialize ROS
    rclpy.init()
    
    config_path = "/scr/satvik/chimera/real/bimanual_franka_env.yaml"
    env = BimanualFrankaEnv(config_path=config_path)
    
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # Parse trajectory into actions
    actions = []
    for step_data in trajectory:
        action = {
            'left_arm': step_data[0:7] if env.use_left_arm else None,
            'left_hand': step_data[7:27] if env.use_left_hand else None,
            'right_arm': step_data[27:34] if env.use_right_arm else None,
            'right_hand': step_data[34:54] if env.use_right_hand else None,
        }
        actions.append(action)
    
    print(f"Parsed {len(actions)} steps")
    
    # Initialize to first position
    first = actions[0]
    env.initialize_to_first_target(
        left_arm=first['left_arm'],
        right_arm=first['right_arm'],
        left_hand=first['left_hand'],
        right_hand=first['right_hand'],
    )
    
    input("Press Enter to start execution...")
    
    # Execute trajectory
    for i, action in enumerate(actions):
        with FreqGuard(hz):
            print(f"Step {i+1}/{len(actions)}")
            env.move_joint_to(
                left_arm=action['left_arm'],
                right_arm=action['right_arm'],
                left_hand=action['left_hand'],
                right_hand=action['right_hand'],
                control_frequency=hz,
            )
    
    print("Done!")
    
    # Cleanup
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=str, default=None, help="Path to .npy trajectory file (default: all zeros)")
    parser.add_argument("--hz", type=float, default=2.0, help="Control frequency (Hz)")
    args = parser.parse_args()
    
    main(args.trajectory, args.hz)

