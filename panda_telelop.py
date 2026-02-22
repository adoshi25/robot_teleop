import os
import threading
import time
from pathlib import Path

import imageio
import numpy as np
from scipy.interpolate import CubicSpline
from sys import platform
# os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["MUJOCO_GL"] = "glfw" if platform == "darwin" else "egl"

import mujoco
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

from teleop_utils import (
    _get_robot_base_poses_from_model,
    _slerp_wxyz,
    _smooth_trajectory,
    _solve_ik_pyroki,
    _solve_ik_pyroki_batch,
    _spline_interpolate,
)


class PandaArmTrajectoryProcessor:
    """Real-time teleop trajectory processor for dual Panda arms.

    Receives wrist poses from teleop, runs IK, applies exponential
    smoothing, and builds a trajectory where each accepted wrist pose
    advances time by 1 / control_hz seconds.
    """

    def __init__(
        self,
        scene_path=None,
        control_hz=60.0,
        min_point_distance=0.005,
        max_point_distance=0.08,
        interpolation_points=3,
        smoothing_alpha=0.4,
        smoothing_sigma=2.0,
        ee_orientation=None,
        pos_only=False,
    ):
        self.callbacks = []
        self.control_hz = float(control_hz)
        self.control_dt = 1.0 / self.control_hz
        self.min_point_distance = min_point_distance
        self.max_point_distance = max_point_distance
        self.interpolation_points = interpolation_points
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_sigma = smoothing_sigma
        self.pos_only = pos_only
        self.ee_orientation = (
            ee_orientation if ee_orientation is not None
            else np.array([0.0, 0.0, 1.0, 0.0])
        )

        self._last_point = {"left": None, "right": None}
        self._last_quat = {"left": None, "right": None}
        self._last_joint = {"left": None, "right": None}
        self._smoothed_joint = {"left": None, "right": None}
        self._logical_time = {"left": None, "right": None}
        self._trajectory = {"left": [], "right": []}
        self._lock = threading.Lock()
        self.last_ik_raw = {"left": None, "right": None}
        self.last_ik_target_base = {"left": None, "right": None}

        if scene_path is None:
            scene_path = (
                Path(__file__).resolve().parent
                / "teleop" / "robots" / "tesollo_hand"
                / "robot_scene_combined.xml"
            )
        self.scene_path = Path(scene_path)

        self._mj_model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._base_poses = _get_robot_base_poses_from_model(self._mj_model)
        self._left_joint_indices = self._get_panda_joint_indices("left")
        self._right_joint_indices = self._get_panda_joint_indices("right")

        self._default_start_joints = self._read_keyframe_joints()

        urdf = load_robot_description("panda_description")
        self._robot = pk.Robot.from_urdf(urdf)
        self._target_link = "panda_hand"

        self._warmup_ik()

    # -- internals ---------------------------------------------------------

    def _get_panda_joint_indices(self, side):
        if side == "left":
            joint_names = (
                [f"panda_left_joint{i}" for i in range(1, 7)]
                + ["panda_joint7"]
            )
        else:
            joint_names = [f"panda_right_joint{i}" for i in range(1, 8)]
        indices = []
        for name in joint_names:
            jid = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name,
            )
            qposadr = self._mj_model.jnt_qposadr[jid]
            nq = (
                1
                if self._mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
                else 4
            )
            for i in range(nq):
                indices.append(int(qposadr) + i)
        return indices

    def _read_keyframe_joints(self):
        """Read canonical joint angles from keyframe 0."""
        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, 0)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        return {
            "left": self._mj_data.qpos[self._left_joint_indices].copy(),
            "right": self._mj_data.qpos[self._right_joint_indices].copy(),
        }

    def _warmup_ik(self):
        """Run a dummy IK solve to trigger JAX JIT compilation at init time."""
        dummy_pos = np.array([0.3, 0.0, 0.3])
        dummy_q = self._default_start_joints["right"]
        self._run_ik(
            dummy_pos, "right",
            initial_q=dummy_q, target_wxyz=self.ee_orientation,
        )

    def _world_to_base(self, world_pos, side):
        pose = self._base_poses[side]
        pos = np.array(world_pos, dtype=float) - pose["pos"]
        quat = pose["quat"]
        if quat is None:
            return pos
        w, x, y, z = quat
        rot = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        return rot.T @ pos

    def _run_ik(self, target_pos, side, initial_q=None, target_wxyz=None):
        wxyz = target_wxyz if target_wxyz is not None else self.ee_orientation
        return np.array(
            _solve_ik_pyroki(
                self._robot, self._target_link, target_pos, wxyz,
                initial_q=initial_q,
                pos_only=self.pos_only,
            )
        )

    def _parse_pose(self, wrist_pose):
        arr = np.asarray(wrist_pose, dtype=float).flatten()
        if len(arr) == 3:
            return arr, None
        if len(arr) >= 7:
            return arr[:3], arr[3:7]
        raise ValueError("wrist_pose must be (3,) xyz or (7,) xyz+wxyz")

    def _update_ema(self, side, q_raw):
        """Feed a joint config into the EMA filter (no recording)."""
        self.last_ik_raw[side] = q_raw.copy()
        if self._smoothed_joint[side] is None:
            self._smoothed_joint[side] = q_raw.copy()
        else:
            a = self.smoothing_alpha
            self._smoothed_joint[side] = (
                a * q_raw + (1 - a) * self._smoothed_joint[side]
            )

    def _record(self, side, t):
        """Append the current smoothed joint to the trajectory at time t."""
        q_out = self._smoothed_joint[side].copy()
        self._trajectory[side].append((t, q_out))
        for cb in self.callbacks:
            cb(side, q_out, t)

    # -- public API --------------------------------------------------------

    def register_callback(self, callback):
        self.callbacks.append(callback)

    def start_trajectory(self):
        self.clear_trajectory()

    def add_point(self, wrist_pose, side):
        """Add a wrist pose. Always records exactly one waypoint at the next control tick."""
        pos, wxyz = self._parse_pose(wrist_pose)
        base_pos = self._world_to_base(pos, side)
        self.last_ik_target_base[side] = base_pos.copy()
        ori = wxyz if wxyz is not None else self.ee_orientation

        with self._lock:
            last = self._last_point[side]
            last_quat = self._last_quat[side]
            last_q = self._last_joint[side]
            prev_t = self._logical_time[side]

            new_t = 0.0 if prev_t is None else prev_t + self.control_dt

            if prev_t is None:
                self._update_ema(side, self._default_start_joints[side])

            skip_ik = False
            if last is not None:
                dist = np.linalg.norm(base_pos - last)
                ori_changed = (
                    last_quat is not None
                    and wxyz is not None
                    and (1.0 - abs(float(np.dot(ori, last_quat)))) > 0.01
                )
                if dist < self.min_point_distance and not ori_changed:
                    skip_ik = True
                elif dist > self.max_point_distance and self.interpolation_points > 0:
                    last_ori = last_quat if last_quat is not None else ori
                    for i in range(1, self.interpolation_points + 1):
                        alpha = i / (self.interpolation_points + 1)
                        interp_pos = (1 - alpha) * last + alpha * base_pos
                        interp_ori = _slerp_wxyz(last_ori, ori, alpha)
                        q = self._run_ik(
                            interp_pos, side,
                            initial_q=last_q, target_wxyz=interp_ori,
                        )
                        if q is not None:
                            self._update_ema(side, q)
                            last_q = q

            if not skip_ik:
                init_q = last_q if last_q is not None else self._default_start_joints[side]
                q = self._run_ik(base_pos, side, initial_q=init_q, target_wxyz=ori)
                if q is not None:
                    self._update_ema(side, q)
                    self._last_point[side] = base_pos.copy()
                    self._last_quat[side] = ori.copy()
                    self._last_joint[side] = q.copy()

            self._record(side, new_t)
            self._logical_time[side] = new_t

    def add_points_batch(self, wrist_poses, side):
        """Add multiple wrist poses using batched GPU IK. Bypasses distance filtering."""
        wrist_poses = np.asarray(wrist_poses, dtype=float)
        if wrist_poses.ndim == 1:
            wrist_poses = wrist_poses.reshape(1, -1)
        base_positions = np.array([
            self._world_to_base(p, side) for p in wrist_poses[:, :3]
        ])
        if wrist_poses.shape[1] == 7:
            qs = _solve_ik_pyroki_batch(
                self._robot, self._target_link, base_positions,
                self.ee_orientation,
                target_wxyz_per_point=wrist_poses[:, 3:7],
            )
        else:
            qs = _solve_ik_pyroki_batch(
                self._robot, self._target_link, base_positions,
                self.ee_orientation,
            )
        with self._lock:
            traj = self._trajectory[side]
            prev_t = self._logical_time[side]
            for i, q in enumerate(qs):
                if prev_t is None and i == 0:
                    t = 0.0
                else:
                    t = (0.0 if prev_t is None else prev_t) + (i + 1) * self.control_dt
                traj.append((t, q))
                for cb in self.callbacks:
                    cb(side, q, t)
                prev_t = t
            self._last_point[side] = base_positions[-1].copy()
            self._last_joint[side] = qs[-1].copy()
            self._last_quat[side] = (
                wrist_poses[-1, 3:7].copy()
                if wrist_poses.shape[1] == 7 else None
            )
            self._logical_time[side] = prev_t

    def get_trajectory(self, side):
        with self._lock:
            traj = self._trajectory[side]
            if not traj:
                return np.array([]), np.zeros((0, 7))
            ts = np.array([x[0] for x in traj])
            qs = np.array([x[1] for x in traj])
            return ts, qs

    def get_smoothed_trajectory(self, side, num_samples=100):
        ts, qs = self.get_trajectory(side)
        if len(ts) < 2:
            return ts, qs
        samples, sample_times = _spline_interpolate(qs, ts, num_samples)
        smoothed = _smooth_trajectory(samples, self.smoothing_sigma)
        return sample_times, smoothed

    def clear_trajectory(self, side=None):
        with self._lock:
            sides = ["left", "right"] if side is None else [side]
            for s in sides:
                self._trajectory[s] = []
                self._last_point[s] = None
                self._last_quat[s] = None
                self._last_joint[s] = None
                self._smoothed_joint[s] = None
                self._logical_time[s] = None

    def get_mujoco_qpos_indices(self, side):
        return (
            self._left_joint_indices if side == "left"
            else self._right_joint_indices
        )

    def end_trajectory(self):
        result = {}
        for side in ("left", "right"):
            _, qs = self.get_trajectory(side)
            if len(qs) > 0:
                result[side] = qs
        return result

    def get_trajectory_duration(self):
        """Return the logical duration (in seconds) of the current trajectory."""
        duration = 0.0
        for side in ("left", "right"):
            ts, _ = self.get_trajectory(side)
            if len(ts) >= 2:
                duration = max(duration, ts[-1] - ts[0])
        return duration

    def record_trajectory_to_video(
        self,
        video_path="trajectory_video.mp4",
        fps=30.0,
        save_ee_poses_path=None,
        **process_kwargs,
    ):
        duration = self.get_trajectory_duration()
        traj = self.end_trajectory()
        if not traj:
            return {}
        return self.process_offline_trajectory(
            traj,
            record_video=True,
            video_path=video_path,
            fps=fps,
            save_ee_poses_path=save_ee_poses_path,
            trajectory_duration=duration,
            **process_kwargs,
        )

    # -- offline processing ------------------------------------------------

    def process_offline_trajectory(
        self,
        trajectory,
        record_video=False,
        video_path="trajectory_video.mp4",
        fps=30.0,
        use_trajopt=True,
        timesteps_per_segment=1,
        max_waypoints=10000,
        save_ee_poses_path=None,
        trajectory_duration=None,
    ):
        if isinstance(trajectory, np.ndarray):
            trajectory = {"right": trajectory}

        result = {}
        for side, qs in trajectory.items():
            qs = np.asarray(qs)
            if len(qs) < 2:
                result[side] = qs
                continue
            if trajectory_duration is None and len(qs) > max_waypoints:
                idx = np.linspace(0, len(qs) - 1, max_waypoints, dtype=int)
                qs = qs[idx]
            if use_trajopt:
                tps = 1 if trajectory_duration is not None else timesteps_per_segment
                result[side] = self._optimize_trajectory_segments(qs, tps)
            else:
                result[side] = qs

        if (record_video or save_ee_poses_path) and self.scene_path.exists():
            self._record_trajectory_video(
                result, video_path, fps,
                record_video=record_video,
                save_ee_poses_path=save_ee_poses_path,
                trajectory_duration=trajectory_duration,
            )
        return result

    def _optimize_trajectory_segments(self, waypoints, timesteps):
        if len(waypoints) < 2:
            return waypoints
        n_out = (
            (len(waypoints) - 1) * timesteps if timesteps > 1
            else len(waypoints)
        )
        ts = np.linspace(0, 1, len(waypoints))
        sample_ts = np.linspace(0, 1, n_out)
        cs = CubicSpline(ts, waypoints, bc_type="clamped")
        return _smooth_trajectory(cs(sample_ts), self.smoothing_sigma)

    def _record_trajectory_video(
        self, trajectory, video_path, fps,
        record_video=True, save_ee_poses_path=None,
        trajectory_duration=None,
    ):
        video_path = Path(video_path).resolve()
        model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        data = mujoco.MjData(model)

        left_idx = self._left_joint_indices
        right_idx = self._right_joint_indices

        left_ee_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "panda_left_link8",
        )
        right_ee_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "panda_link8",
        )

        max_len = max(
            len(trajectory.get("left", [])),
            len(trajectory.get("right", [])),
        )
        if max_len < 2:
            return

        if trajectory_duration is not None and trajectory_duration > 0:
            n_frames = max(int(round(trajectory_duration * fps)), 2)
        else:
            n_frames = max_len
        n_frames = max(n_frames, 2)

        left_q = trajectory.get("left")
        right_q = trajectory.get("right")
        if left_q is None:
            left_q = np.tile(data.qpos[left_idx].copy(), (n_frames, 1))
        else:
            left_q = np.asarray(left_q)
        if right_q is None:
            right_q = np.tile(data.qpos[right_idx].copy(), (n_frames, 1))
        else:
            right_q = np.asarray(right_q)

        def _resample(q, n):
            if len(q) == n:
                return q
            return np.array([
                np.interp(
                    np.linspace(0, len(q) - 1, n),
                    np.arange(len(q)), q[:, j],
                )
                for j in range(7)
            ]).T

        left_q = _resample(left_q, n_frames)
        right_q = _resample(right_q, n_frames)

        if trajectory_duration is not None and trajectory_duration > 0:
            timesteps = np.linspace(0, trajectory_duration, n_frames)
        else:
            timesteps = np.arange(n_frames) / fps
        left_pos = np.zeros((n_frames, 3))
        left_quat = np.zeros((n_frames, 4))
        right_pos = np.zeros((n_frames, 3))
        right_quat = np.zeros((n_frames, 4))

        has_left = trajectory.get("left") is not None
        has_right = trajectory.get("right") is not None

        renderer = (
            mujoco.Renderer(model, height=480, width=640)
            if record_video else None
        )
        frames = [] if record_video else None

        cam = None
        if record_video:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.lookat[:] = [0.75, 0.0, 0.35]
            cam.distance = 2.0
            cam.azimuth = 180
            cam.elevation = -30

        try:
            for i in range(n_frames):
                data.qpos[left_idx] = left_q[i]
                data.qpos[right_idx] = right_q[i]
                mujoco.mj_forward(model, data)
                if has_left:
                    left_pos[i] = data.xpos[left_ee_bid].copy()
                    left_quat[i] = data.xquat[left_ee_bid].copy()
                if has_right:
                    right_pos[i] = data.xpos[right_ee_bid].copy()
                    right_quat[i] = data.xquat[right_ee_bid].copy()
                if record_video and renderer is not None:
                    renderer.update_scene(data, camera=cam)
                    frames.append(renderer.render())
        finally:
            if renderer is not None:
                renderer.close()

        if record_video and frames:
            imageio.mimsave(str(video_path), frames, fps=fps)
            print(f"Video saved to {video_path}")

        if save_ee_poses_path:
            out = {"timesteps": timesteps, "fps": fps}
            if has_left:
                out["left_position"] = left_pos
                out["left_quat_wxyz"] = left_quat
            if has_right:
                out["right_position"] = right_pos
                out["right_quat_wxyz"] = right_quat
            out_path = Path(save_ee_poses_path).resolve()
            np.savez(out_path, **out)
            print(f"EE poses saved to {out_path}")
