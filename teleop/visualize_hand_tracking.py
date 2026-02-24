#!/usr/bin/env python3
"""WebXR hand tracking → MuJoCo IK visualizer.

Press SPACE to start. After a 2.5s warmup (hold hands steady), the offset
between your WebXR wrist and the robot's canonical pose is locked in.
All subsequent hand motion is 1:1 relative to that anchor.

Usage:
    1. python teleop/start_hand_tracker.py
    2. Connect VR headset, start hand tracking in browser
    3. mjpython run_visualize_tesollo.py   (or run this directly)
"""

import argparse
import io
import json
import socket
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from panda_telelop import PandaArmTrajectoryProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HAND_TRACKER_URL = "http://localhost:9002/get_hand_data"
FRAME_POST_URL = "http://localhost:9002/mujoco_frame"
FRAME_POST_HZ = 15
OFFSCREEN_WIDTH = 640
OFFSCREEN_HEIGHT = 480
DEFAULT_MJCF = (
    Path(__file__).parent.parent / "teleop" / "robots" / "tesollo_hand" / "robot_scene_combined.xml"
)
CONTROL_HZ = 60
WARMUP_SECONDS = 2.5
MAX_JOINT_DELTA = 1000

JOINT_NAMES = [
    "wrist",
    "thumb-tip",
    "index-finger-tip",
    "middle-finger-tip",
    "ring-finger-tip",
    "pinky-finger-tip",
]

MOCAP_BODIES = {
    side: {j: f"{side}-{j}" for j in JOINT_NAMES} for side in ("left", "right")
}

ROBOT_SITES = {
    "left": {
        "wrist": "left_palm_site",
        "thumb-tip": "left_thumb_tip_site",
        "index-finger-tip": "left_index_tip_site",
        "middle-finger-tip": "left_middle_tip_site",
        "ring-finger-tip": "left_ring_tip_site",
        "pinky-finger-tip": "left_pinky_tip_site",
    },
    "right": {
        "wrist": "right_palm_site",
        "thumb-tip": "right_thumb_tip_site",
        "index-finger-tip": "right_index_tip_site",
        "middle-finger-tip": "right_middle_tip_site",
        "ring-finger-tip": "right_ring_tip_site",
        "pinky-finger-tip": "right_pinky_tip_site",
    },
}

FINGER_KEYPOINTS = {
    "thumb": [
        "thumb-metacarpal", "thumb-phalanx-proximal",
        "thumb-phalanx-distal", "thumb-tip",
    ],
    "index": [
        "index-finger-metacarpal", "index-finger-phalanx-proximal",
        "index-finger-phalanx-intermediate", "index-finger-phalanx-distal",
        "index-finger-tip",
    ],
    "middle": [
        "middle-finger-metacarpal", "middle-finger-phalanx-proximal",
        "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal",
        "middle-finger-tip",
    ],
    "ring": [
        "ring-finger-metacarpal", "ring-finger-phalanx-proximal",
        "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal",
        "ring-finger-tip",
    ],
    "pinky": [
        "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal",
        "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal",
        "pinky-finger-tip",
    ],
}

FINGER_JOINT_NAMES = {
    "left": {
        "thumb":  ["lj_dg_1_1", "lj_dg_1_2", "lj_dg_1_3", "lj_dg_1_4"],
        "index":  ["lj_dg_2_1", "lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4"],
        "middle": ["lj_dg_3_1", "lj_dg_3_2", "lj_dg_3_3", "lj_dg_3_4"],
        "ring":   ["lj_dg_4_1", "lj_dg_4_2", "lj_dg_4_3", "lj_dg_4_4"],
        "pinky":  ["lj_dg_5_1", "lj_dg_5_2", "lj_dg_5_3", "lj_dg_5_4"],
    },
    "right": {
        "thumb":  ["rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4"],
        "index":  ["rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4"],
        "middle": ["rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4"],
        "ring":   ["rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4"],
        "pinky":  ["rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"],
    },
}

# Maps inter-bone-angle index → robot joint index (0-based within the 4 joints).
# Thumb: 4 keypoints → 3 bones → 2 angles → joints _3,_4 (flexion only)
# Index/middle/ring: 5 kp → 4 bones → 3 angles → joints _2,_3,_4
# Pinky: 5 kp → 4 bones → 3 angles → skip MCP, joints _3,_4
FINGER_ANGLE_TO_JOINT = {
    "thumb":  {0: 2, 1: 3},
    "index":  {0: 1, 1: 2, 2: 3},
    "middle": {0: 1, 1: 2, 2: 3},
    "ring":   {0: 1, 1: 2, 2: 3},
    "pinky":  {1: 2, 2: 3},
}

# Abduction: proximal → intermediate bone direction, projected onto palm plane.
_ABD_FINGERS = [
    ("index",  "index-finger-phalanx-proximal",  "index-finger-phalanx-intermediate",  0),
    ("middle", "middle-finger-phalanx-proximal",  "middle-finger-phalanx-intermediate", 0),
    ("ring",   "ring-finger-phalanx-proximal",    "ring-finger-phalanx-intermediate",   0),
    ("pinky",  "pinky-finger-phalanx-proximal",   "pinky-finger-phalanx-intermediate",  1),
]

# ---------------------------------------------------------------------------
# Coordinate / quaternion helpers
# ---------------------------------------------------------------------------


def webxr_to_mujoco(pos_dict):
    """WebXR {x,y,z} → MuJoCo frame (-z, x, y)."""
    return np.array([-pos_dict["x"], pos_dict["z"], pos_dict["y"]])


def webxr_quat_to_mujoco_wxyz(ori_dict):
    """WebXR orientation {x,y,z,w} (xyzw) → MuJoCo wxyz quaternion.

    Rotates via q_m = q_map ⊗ q_web ⊗ q_map⁻¹.
    """
    q_web = np.array(
        [ori_dict["w"], ori_dict["x"], ori_dict["y"], ori_dict["z"]],
        dtype=float,
    )
    q_map = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    return _qmul(q_map, _qmul(q_web, _qinv(q_map)))


def _qmul(q1, q2):
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qinv(q):
    """Conjugate (inverse for unit quaternions) in wxyz format."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _xmat_to_wxyz(xmat):
    """Convert a 3x3 rotation matrix (flattened 9-vec from MuJoCo) to wxyz quaternion."""
    mat = np.array(xmat, dtype=float).reshape(3, 3)
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(mat)
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _get_pos(hand_joints, name):
    kp = hand_joints.get(name)
    if kp and "position" in kp:
        return webxr_to_mujoco(kp["position"])
    return None


# ---------------------------------------------------------------------------
# Pure-function helpers (no state dependency)
# ---------------------------------------------------------------------------


def fetch_hands():
    try:
        r = requests.get(HAND_TRACKER_URL, timeout=0.1)
        if r.ok:
            d = r.json()
            if d and "hands" in d:
                return d["hands"]
    except (requests.RequestException, ValueError):
        pass
    return None


def read_canonical_positions(model, data):
    """FK positions and wrist orientations of every tracked site at the current qpos."""
    out = {}
    for side, joints in ROBOT_SITES.items():
        out[side] = {}
        for jname, sname in joints.items():
            sid = model.site(sname).id
            out[side][jname] = data.site_xpos[sid].copy()
            if jname == "wrist":
                out[side]["_wrist_quat"] = _xmat_to_wxyz(data.site_xmat[sid])
    return out


def compute_offsets(hand_data, canonical):
    """Per-joint offset so that initial WebXR pose maps onto canonical FK."""
    offsets = {}
    for side in ("left", "right"):
        joints = hand_data.get(side)
        if not joints:
            continue
        offsets[side] = {}
        for jname in JOINT_NAMES:
            jd = joints.get(jname)
            canon = canonical.get(side, {}).get(jname)
            if jd and "position" in jd and canon is not None:
                offsets[side][jname] = canon - webxr_to_mujoco(jd["position"])
        wrist = joints.get("wrist")
        if wrist and "orientation" in wrist:
            offsets[side]["_wrist_quat_cal"] = webxr_quat_to_mujoco_wxyz(
                wrist["orientation"]
            )
        canon_wrist_q = canonical.get(side, {}).get("_wrist_quat")
        if canon_wrist_q is not None:
            offsets[side]["_wrist_quat_robot"] = canon_wrist_q.copy()
    return offsets


def get_offset_wrist(hand_data, side, offsets):
    """Offset-corrected wrist position in world frame, or None."""
    joints = hand_data.get(side)
    off = offsets.get(side, {}).get("wrist")
    if joints is None or off is None:
        return None
    w = joints.get("wrist")
    if w is None or "position" not in w:
        return None
    return webxr_to_mujoco(w["position"]) + off


def get_offset_wrist_pose(hand_data, side, offsets):
    """Offset-corrected wrist pose [x, y, z, w, qx, qy, qz] in world frame, or None."""
    pos = get_offset_wrist(hand_data, side, offsets)
    if pos is None:
        return None

    side_off = offsets.get(side, {})
    q_cal = side_off.get("_wrist_quat_cal")
    q_robot = side_off.get("_wrist_quat_robot")

    joints = hand_data.get(side, {})
    wrist = joints.get("wrist", {})

    if q_cal is None or q_robot is None or "orientation" not in wrist:
        return pos

    q_cur = webxr_quat_to_mujoco_wxyz(wrist["orientation"])
    q_delta = _qmul(q_cur, _qinv(q_cal))
    q_target = _qmul(q_delta, q_robot)

    return np.concatenate([pos, q_target])


def build_finger_index_map(model):
    """Build {(side, finger, joint_idx): (qpos_idx, lo, hi)} for all finger joints."""
    result = {}
    for side, fingers in FINGER_JOINT_NAMES.items():
        for finger_name, joint_names in fingers.items():
            for j_idx, jname in enumerate(joint_names):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                qpos_idx = int(model.jnt_qposadr[jid])
                lo = float(model.jnt_range[jid, 0])
                hi = float(model.jnt_range[jid, 1])
                result[(side, finger_name, j_idx)] = (qpos_idx, lo, hi)
    return result


def _build_palm_frame(hand_joints):
    """Build an orthonormal palm frame (forward, lateral, normal) from keypoints."""
    wrist_pos = _get_pos(hand_joints, "wrist")
    mid_meta = _get_pos(hand_joints, "middle-finger-metacarpal")
    idx_meta = _get_pos(hand_joints, "index-finger-metacarpal")
    pinky_meta = _get_pos(hand_joints, "pinky-finger-metacarpal")

    if any(p is None for p in (wrist_pos, mid_meta, idx_meta, pinky_meta)):
        return None

    fwd = mid_meta - wrist_pos
    fwd_n = np.linalg.norm(fwd)
    if fwd_n < 1e-6:
        return None
    fwd = fwd / fwd_n

    lat = pinky_meta - idx_meta
    lat -= np.dot(lat, fwd) * fwd
    lat_n = np.linalg.norm(lat)
    if lat_n < 1e-6:
        return None
    lat = lat / lat_n

    nrm = np.cross(fwd, lat)
    return fwd, lat, nrm


def calibrate_abduction_rest(hand_data):
    """Record per-finger lateral angles at calibration (rest) pose."""
    rest = {}
    for side in ("left", "right"):
        joints = hand_data.get(side)
        if not joints:
            continue
        frame = _build_palm_frame(joints)
        if frame is None:
            continue
        fwd, lat, nrm = frame

        for fname, prox_name, inter_name, _j_idx in _ABD_FINGERS:
            prox = _get_pos(joints, prox_name)
            inter = _get_pos(joints, inter_name)
            if prox is None or inter is None:
                continue
            bone = inter - prox
            bone -= np.dot(bone, nrm) * nrm
            bn = np.linalg.norm(bone)
            if bn < 1e-6:
                continue
            bone = bone / bn
            rest[(side, fname)] = float(np.arctan2(
                np.dot(bone, lat), np.dot(bone, fwd)
            ))
    return rest


def _compute_abduction(hand_joints, side, finger_map, result, abd_rest):
    """Compute abduction from lateral deviation of proximal phalanx."""
    frame = _build_palm_frame(hand_joints)
    if frame is None:
        return
    fwd, lat, nrm = frame

    for fname, prox_name, inter_name, j_idx in _ABD_FINGERS:
        prox = _get_pos(hand_joints, prox_name)
        inter = _get_pos(hand_joints, inter_name)
        if prox is None or inter is None:
            continue
        bone = inter - prox
        bone -= np.dot(bone, nrm) * nrm
        bn = np.linalg.norm(bone)
        if bn < 1e-6:
            continue
        bone = bone / bn

        angle = float(np.arctan2(np.dot(bone, lat), np.dot(bone, fwd)))
        rest_angle = abd_rest.get((side, fname), angle)
        side_sign = -1.0 if side == "left" else 1.0
        abd = side_sign * (angle - rest_angle)

        key = (side, fname, j_idx)
        if key not in finger_map:
            continue
        qpos_idx, lo, hi = finger_map[key]
        result[qpos_idx] = float(np.clip(abd, lo, hi))


def retarget_fingers(hand_joints, side, finger_map, abd_rest):
    """Compute finger joint angles from WebXR keypoint positions.

    Returns {qpos_index: angle} for all 20 finger joints on one hand.
    """
    result = {}

    for finger_name, kp_names in FINGER_KEYPOINTS.items():
        positions = []
        for kp_name in kp_names:
            kp = hand_joints.get(kp_name)
            if kp is None or "position" not in kp:
                break
            positions.append(webxr_to_mujoco(kp["position"]))

        if len(positions) < len(kp_names):
            continue

        bones = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

        flexion_angles = []
        for i in range(len(bones) - 1):
            n1 = bones[i] / (np.linalg.norm(bones[i]) + 1e-8)
            n2 = bones[i + 1] / (np.linalg.norm(bones[i + 1]) + 1e-8)
            cos_a = np.clip(np.dot(n1, n2), -1.0, 1.0)
            flexion_angles.append(float(np.arccos(cos_a)))

        angle_map = FINGER_ANGLE_TO_JOINT.get(finger_name, {})

        for angle_idx, joint_idx in angle_map.items():
            if angle_idx >= len(flexion_angles):
                continue
            key = (side, finger_name, joint_idx)
            if key not in finger_map:
                continue
            qpos_idx, lo, hi = finger_map[key]
            angle = flexion_angles[angle_idx]
            if hi <= 0 or (finger_name == "thumb" and side == "left"):
                angle = -angle
            result[qpos_idx] = float(np.clip(angle, lo, hi))

        for j_idx in range(4):
            if j_idx not in angle_map.values():
                key = (side, finger_name, j_idx)
                if key in finger_map:
                    qpos_idx, lo, hi = finger_map[key]
                    if qpos_idx not in result:
                        result[qpos_idx] = float(np.clip(0.0, lo, hi))

    _compute_abduction(hand_joints, side, finger_map, result, abd_rest)

    return result


def sync_mocap_to_sites(model, data):
    """Position red-dot mocap bodies at the robot's current FK site positions."""
    for side in ("left", "right"):
        for jname, site_name in ROBOT_SITES[side].items():
            mocap_name = MOCAP_BODIES[side].get(jname)
            if mocap_name is None:
                continue
            try:
                mid = model.body(mocap_name).mocapid[0]
                sid = model.site(site_name).id
                if mid >= 0:
                    data.mocap_pos[mid] = data.site_xpos[sid].copy()
            except (KeyError, IndexError):
                pass


def draw_webxr_keypoints(viewer, hand_data, offsets=None):
    """Render WebXR keypoints as colored spheres in the viewer."""
    viewer.user_scn.ngeom = 0
    if not hand_data:
        return

    mat = np.eye(3).ravel()

    def add_sphere(pos, sz, color):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        gid = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[gid],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            sz, pos, mat, color,
        )
        viewer.user_scn.ngeom += 1

    blue_sz = np.array([0.008, 0.008, 0.008])
    blue_rgba = np.array([0.2, 0.4, 1.0, 0.9])
    red_sz = np.array([0.012, 0.012, 0.012])
    red_rgba = np.array([1.0, 0.2, 0.2, 0.9])

    for side in ("left", "right"):
        joints = hand_data.get(side)
        if not joints:
            continue
        side_offsets = offsets.get(side, {}) if offsets else {}
        for name, kp in joints.items():
            if kp is None or "position" not in kp:
                continue
            pos = webxr_to_mujoco(kp["position"])
            add_sphere(pos, blue_sz, blue_rgba)
            off = side_offsets.get(name)
            if off is not None:
                add_sphere(pos + off, red_sz, red_rgba)


# ---------------------------------------------------------------------------
# TeleopTracker
# ---------------------------------------------------------------------------


class TeleopTracker:
    """Encapsulates the full WebXR → MuJoCo IK teleop loop.

    Manages model loading, IK processing, finger retargeting, frame streaming,
    and the warmup/calibration/tracking state machine.
    """

    def __init__(
        self,
        mjcf_path=None,
        reanchor=True,
        raw_ik=False,
        stream_frames=True,
        enable_collision_check=True,
        log_npy_path="hand_teleop_log.npz",
        socket_host="127.0.0.1",
        socket_port=9004,
    ):
        self.mjcf_path = Path(mjcf_path) if mjcf_path else DEFAULT_MJCF
        self.reanchor_enabled = reanchor
        self.stream_frames = stream_frames
        self.collision_check_enabled = enable_collision_check
        self.log_npy_path = log_npy_path
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.dt = 1.0 / CONTROL_HZ

        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"{self.mjcf_path} not found")

        # MuJoCo model & data
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.free_nq = 0
        self.free_nv = 0
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_nq += 7
                self.free_nv += 6

        # Offscreen renderer for VR headset streaming
        self._offscreen = mujoco.Renderer(
            self.model, height=OFFSCREEN_HEIGHT, width=OFFSCREEN_WIDTH,
        )
        self._offscreen_cam = mujoco.MjvCamera()
        self._offscreen_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._offscreen_cam.distance = 2.0
        self._offscreen_cam.azimuth = 270
        self._offscreen_cam.elevation = -30
        self._offscreen_cam.lookat[:] = [0, 0, 1]
        self._last_frame_post = 0.0

        # Canonical (home) FK positions
        self.canonical = read_canonical_positions(self.model, self.data)

        # IK processor
        if raw_ik:
            self.proc = PandaArmTrajectoryProcessor(
                scene_path=self.mjcf_path,
                control_hz=CONTROL_HZ,
                min_point_distance=0.0,
                max_point_distance=1e9,
                interpolation_points=0,
                smoothing_alpha=1.0,
                smoothing_sigma=0.0,
                ori_weight=5.0,
            )
        else:
            self.proc = PandaArmTrajectoryProcessor(
                scene_path=self.mjcf_path, control_hz=CONTROL_HZ, ori_weight=5.0,
            )

        self.arm_idx = {
            s: self.proc.get_mujoco_qpos_indices(s)
            for s in ("left", "right")
        }
        self.canonical_q = {
            s: self.data.qpos[self.arm_idx[s]].copy()
            for s in ("left", "right")
        }

        self.proc.register_callback(self._on_ik)

        # Finger retargeting
        self.finger_map = build_finger_index_map(self.model)

        # Collision checking (pre-compute once at init)
        self._geom_is_left = np.zeros(self.model.ngeom, dtype=bool)
        self._geom_is_right = np.zeros(self.model.ngeom, dtype=bool)
        self._build_collision_geom_groups()

        # Baseline: rest-pose contact body pairs are structural mesh overlaps
        self._baseline_body_pairs: set[tuple[int, int]] = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist < 0:
                b1 = int(self.model.geom_bodyid[int(c.geom[0])])
                b2 = int(self.model.geom_bodyid[int(c.geom[1])])
                self._baseline_body_pairs.add((min(b1, b2), max(b1, b2)))
        print(f"Baseline contact pairs (rest pose): {len(self._baseline_body_pairs)}")

        # Tracking state (reset in start/stop)
        self.active = False
        self._t_start = 0.0
        self.offsets = {}
        self._warmup_done = False
        self._warmup_data = None
        self._latest_raw_hand_data = None
        self._last_fetch = 0.0
        self._ik_q = {"left": None, "right": None}
        self._prev_ik_q = {"left": None, "right": None}
        self._finger_qpos = {"left": {}, "right": {}}
        self._abd_rest = {}
        self._reanchor = {"left": False, "right": False}
        self._latest_webxr_wrist = {"left": None, "right": None}
        self._ik_diag = {"left": {}, "right": {}}
        self._collision_info = {
            "left_self_collision": False, "right_self_collision": False,
            "inter_collision": False, "n_contacts": 0, "rejected": False,
        }
        self._safe_robot_qpos = self.data.qpos[self.free_nq:].copy()
        self._log_samples = []

        # Thread-safety for non-blocking mode
        self._qpos_lock = threading.Lock()
        self._latest_qpos = self.data.qpos[self.free_nq:].copy()
        self._thread = None
        self._stop_event = threading.Event()

        self._socket_server_sock = None
        self._socket_server_thread = None

        self.is_ready = False

    # -- Collision checking ------------------------------------------------

    def _build_collision_geom_groups(self):
        """Pre-compute boolean arrays mapping geom IDs to left/right robot.

        Walks the body tree from each side's joints upward, removes shared
        ancestor bodies, then collects all descendant bodies.  The result is
        two boolean arrays (one per side) indexed by geom ID for O(1) lookup.
        """
        left_joint_bodies: set[int] = set()
        right_joint_bodies: set[int] = set()

        for side in ("left", "right"):
            bset = left_joint_bodies if side == "left" else right_joint_bodies
            qpos_set = set(int(qi) for qi in self.arm_idx[side])
            for j in range(self.model.njnt):
                if int(self.model.jnt_qposadr[j]) in qpos_set:
                    bset.add(int(self.model.jnt_bodyid[j]))

        for side, fingers in FINGER_JOINT_NAMES.items():
            bset = left_joint_bodies if side == "left" else right_joint_bodies
            for joint_names in fingers.values():
                for jname in joint_names:
                    jid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, jname,
                    )
                    if jid >= 0:
                        bset.add(int(self.model.jnt_bodyid[jid]))

        left_all: set[int] = set()
        right_all: set[int] = set()
        for bid in left_joint_bodies:
            b = bid
            while b > 0:
                left_all.add(b)
                b = int(self.model.body_parentid[b])
        for bid in right_joint_bodies:
            b = bid
            while b > 0:
                right_all.add(b)
                b = int(self.model.body_parentid[b])

        shared = left_all & right_all
        left_bodies = left_all - shared
        right_bodies = right_all - shared

        # Add descendant bodies (MuJoCo guarantees parent ID < child ID)
        for b in range(self.model.nbody):
            if b in left_bodies or b in right_bodies:
                continue
            parent = int(self.model.body_parentid[b])
            if parent in left_bodies:
                left_bodies.add(b)
            elif parent in right_bodies:
                right_bodies.add(b)

        ngeom = self.model.ngeom
        self._geom_is_left = np.zeros(ngeom, dtype=bool)
        self._geom_is_right = np.zeros(ngeom, dtype=bool)
        for g in range(ngeom):
            bid = int(self.model.geom_bodyid[g])
            if bid in left_bodies:
                self._geom_is_left[g] = True
            if bid in right_bodies:
                self._geom_is_right[g] = True

        print(
            f"Collision groups: {int(np.sum(self._geom_is_left))} left geoms, "
            f"{int(np.sum(self._geom_is_right))} right geoms"
        )

    def _check_collisions(self):
        """Classify contacts already computed on self.data by mj_forward.

        These are the exact same contacts the MuJoCo viewer visualises
        (toggle with backtick).  Contacts between body pairs that exist
        at the rest pose are treated as structural mesh overlaps and
        skipped.  When collision checking is disabled, this is a cheap
        no-op.
        """
        if not self.collision_check_enabled:
            return {
                "left_self_collision": False, "right_self_collision": False,
                "inter_collision": False, "n_contacts": 0, "rejected": False,
            }

        ncon = self.data.ncon
        if ncon == 0:
            return {
                "left_self_collision": False, "right_self_collision": False,
                "inter_collision": False, "n_contacts": 0, "rejected": False,
            }

        left_self = False
        right_self = False
        inter = False
        n_pen = 0
        geom_is_left = self._geom_is_left
        geom_is_right = self._geom_is_right
        geom_bodyid = self.model.geom_bodyid
        baseline = self._baseline_body_pairs

        for i in range(ncon):
            c = self.data.contact[i]
            if c.dist >= 0:
                continue
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            b1, b2 = int(geom_bodyid[g1]), int(geom_bodyid[g2])
            if (min(b1, b2), max(b1, b2)) in baseline:
                continue
            n_pen += 1
            g1l = geom_is_left[g1]
            g1r = geom_is_right[g1]
            g2l = geom_is_left[g2]
            g2r = geom_is_right[g2]
            left_self = left_self or (g1l and g2l)
            right_self = right_self or (g1r and g2r)
            inter = inter or ((g1l and g2r) or (g1r and g2l))
            if left_self and right_self and inter:
                break

        rejected = left_self or right_self or inter
        return {
            "left_self_collision": left_self,
            "right_self_collision": right_self,
            "inter_collision": inter,
            "n_contacts": n_pen,
            "rejected": rejected,
        }

    # -- IK callback -------------------------------------------------------

    def _on_ik(self, side, q, _t):
        prev = self._prev_ik_q[side]
        if prev is not None:
            max_delta = float(np.max(np.abs(q - prev)))
            rejected = max_delta > MAX_JOINT_DELTA
            self._ik_diag[side] = {
                "q_smoothed": q.copy(),
                "max_delta": max_delta,
                "rejected": rejected,
            }
            if rejected:
                if self.reanchor_enabled:
                    self._reanchor[side] = True
                return
        else:
            self._ik_diag[side] = {
                "q_smoothed": q.copy(),
                "max_delta": 0.0,
                "rejected": False,
            }
        self._ik_q[side] = q.copy()
        self._prev_ik_q[side] = q.copy()

    # -- Frame streaming ---------------------------------------------------

    def _maybe_post_frame(self, t):
        if not self.stream_frames:
            return
        if t - self._last_frame_post < 1.0 / FRAME_POST_HZ:
            return
        try:
            self._offscreen.update_scene(self.data, camera=self._offscreen_cam)
            pixels = self._offscreen.render()
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format="JPEG", quality=70)
            requests.post(
                FRAME_POST_URL,
                data=buf.getvalue(),
                headers={"Content-Type": "image/jpeg"},
                timeout=0.05,
            )
        except Exception:
            pass
        self._last_frame_post = t

    # -- Public API --------------------------------------------------------

    def get_qpos(self, order='mujoco'):
        """Return the current robot joint positions (excluding free joints).

        This is the primary output for downstream controllers: a 1-D numpy
        array of all non-free-body joint angles in the MuJoCo model.
        Thread-safe — can be called from any thread while the loop is running.
        """
        with self._qpos_lock:
            qpos = self._latest_qpos.copy()
        
        if order == 'mujoco':
            return qpos

        left_robot_qpos = qpos[:27]
        right_robot_qpos = qpos[27:]

        if order == "genesis":
            assert left_robot_qpos.shape == (27, )
            assert right_robot_qpos.shape == (27, )
            MUJOCO_TO_GENESIS_ORDER = np.array([
                0, 1, 2, 3, 4, 5, 6,
                7, 11, 15, 19, 23,
                8, 12, 16, 20, 24,
                9, 13, 17, 21, 25,
                10, 14, 18, 22, 26,
            ])
            left_robot_qpos = left_robot_qpos[MUJOCO_TO_GENESIS_ORDER]
            right_robot_qpos = right_robot_qpos[MUJOCO_TO_GENESIS_ORDER]

            return np.concatenate([left_robot_qpos, right_robot_qpos], axis=0)
        
    def start(self):
        """Begin a new tracking session (warmup phase starts immediately)."""
        self.active = True
        self._t_start = time.time()
        self.offsets = {}
        self._warmup_done = False
        self._warmup_data = None
        self.proc.start_trajectory()
        self._ik_q = {"left": None, "right": None}
        self._prev_ik_q = {
            "left": self.canonical_q["left"].copy(),
            "right": self.canonical_q["right"].copy(),
        }
        self._finger_qpos = {"left": {}, "right": {}}
        self._abd_rest = {}
        self._reanchor = {"left": False, "right": False}
        self._latest_webxr_wrist = {"left": None, "right": None}
        self._collision_info = {
            "left_self_collision": False, "right_self_collision": False,
            "inter_collision": False, "n_contacts": 0, "rejected": False,
        }
        self._safe_robot_qpos = self.data.qpos[self.free_nq:].copy()
        self._log_samples.clear()
        print(f"Warmup {WARMUP_SECONDS}s — hold hands steady...")

    def stop(self):
        """End the current tracking session and optionally save logs."""
        self.active = False
        self.proc.clear_trajectory()
        self._ik_q = {"left": None, "right": None}
        self._prev_ik_q = {"left": None, "right": None}
        self._finger_qpos = {"left": {}, "right": {}}
        self._abd_rest = {}
        self._reanchor = {"left": False, "right": False}
        self._latest_webxr_wrist = {"left": None, "right": None}
        print("Stopped.")
        self.is_ready = False
        if self._log_samples:
            np.savez_compressed(
                self.log_npy_path,
                samples=np.array(self._log_samples, dtype=object),
            )
            print(f"Saved log: {self.log_npy_path} ({len(self._log_samples)} samples)")

    # -- Per-frame tick phases ---------------------------------------------

    def _tick_idle(self, now):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        sync_mocap_to_sites(self.model, self.data)
        self._maybe_post_frame(now)

    def _tick_warmup(self, now):
        hd = fetch_hands()
        if hd:
            self._warmup_data = hd
            self._latest_raw_hand_data = hd
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        sync_mocap_to_sites(self.model, self.data)
        self._maybe_post_frame(now)

    def _finalize_warmup(self):
        self._warmup_done = True
        hd = self._warmup_data or fetch_hands()
        if hd:
            self.offsets = compute_offsets(hd, self.canonical)
            for side, so in self.offsets.items():
                w = so.get("wrist")
                if w is not None:
                    print(f"  {side} wrist offset: {w}")
            for side in ("left", "right"):
                wrist_target = get_offset_wrist(hd, side, self.offsets)
                canon_w = self.canonical.get(side, {}).get("wrist")
                if wrist_target is not None and canon_w is not None:
                    err = np.linalg.norm(wrist_target - canon_w)
                    print(f"  {side} calibration verify: target_err={err:.4f}")
            self._abd_rest = calibrate_abduction_rest(hd)
            print(f"  abduction rest angles: {self._abd_rest}")
        print("Tracking active!")
        self.is_ready = True

    def _tick_tracking(self, now):
        if now - self._last_fetch >= self.dt:
            hand_data = fetch_hands()
            if hand_data:
                self._latest_raw_hand_data = hand_data
            if hand_data and self.offsets:
                for side in ("left", "right"):
                    joints = hand_data.get(side)
                    if joints:
                        w = joints.get("wrist")
                        if w and "position" in w:
                            self._latest_webxr_wrist[side] = w["position"]

                    if (self._reanchor[side]
                            and self._latest_webxr_wrist[side] is not None
                            and side in self.offsets):
                        site_name = ROBOT_SITES[side]["wrist"]
                        robot_pos = self.data.site_xpos[
                            self.model.site(site_name).id
                        ].copy()
                        new_off = robot_pos - webxr_to_mujoco(
                            self._latest_webxr_wrist[side]
                        )
                        self.offsets[side]["wrist"] = new_off
                        self._prev_ik_q[side] = self.data.qpos[self.arm_idx[side]].copy()
                        self._reanchor[side] = False

                    wrist_pose = get_offset_wrist_pose(hand_data, side, self.offsets)
                    if wrist_pose is not None:
                        self.proc.add_point(wrist_pose, side)
                    if joints:
                        self._finger_qpos[side] = retarget_fingers(
                            joints, side, self.finger_map, self._abd_rest,
                        )
            self._last_fetch = now

        # Apply IK + finger results to qpos
        for side in ("left", "right"):
            q = self._ik_q[side]
            if q is not None:
                self.data.qpos[self.arm_idx[side]] = q
            for qidx, angle in self._finger_qpos[side].items():
                self.data.qpos[qidx] = angle

        robot_qpos = self.data.qpos[self.free_nq:].copy()
        mujoco.mj_step(self.model, self.data)
        self.data.qpos[self.free_nq:] = robot_qpos
        self.data.qvel[self.free_nv:] = 0
        mujoco.mj_forward(self.model, self.data)

        # Collision check on the contacts mj_forward just computed
        # (same contacts the viewer shows when pressing backtick)
        self._collision_info = self._check_collisions()
        if self._collision_info["rejected"]:
            self.data.qpos[self.free_nq:] = self._safe_robot_qpos
            for side in ("left", "right"):
                self._prev_ik_q[side] = self.data.qpos[self.arm_idx[side]].copy()
                for qidx in self._finger_qpos[side]:
                    self._finger_qpos[side][qidx] = float(self.data.qpos[qidx])
            mujoco.mj_forward(self.model, self.data)
        else:
            self._safe_robot_qpos = self.data.qpos[self.free_nq:].copy()

        sync_mocap_to_sites(self.model, self.data)
        self._maybe_post_frame(now)

        self._record_log_sample(now)

    def _record_log_sample(self, now):
        if not self._latest_raw_hand_data:
            return
        robot_sites = {}
        for side in ("left", "right"):
            robot_sites[side] = {}
            for jname, site_name in ROBOT_SITES[side].items():
                sid = self.model.site(site_name).id
                robot_sites[side][jname] = self.data.site_xpos[sid].copy()

        offsets_copy = {
            s: {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in d.items()}
            for s, d in self.offsets.items()
        } if self.offsets else {}

        ik_diag_copy = {}
        for side in ("left", "right"):
            d = self._ik_diag.get(side, {})
            ik_diag_copy[side] = {
                k: (v.copy() if hasattr(v, 'copy') else v) for k, v in d.items()
            }

        ik_target_world = {}
        for side in ("left", "right"):
            raw = self._latest_raw_hand_data.get(side, {})
            raw_wrist = raw.get("wrist", {})
            off = self.offsets.get(side, {}).get("wrist")
            if "position" in raw_wrist and off is not None:
                pos_mj = webxr_to_mujoco(raw_wrist["position"])
                ik_target_world[side] = (pos_mj + off).copy()
            else:
                ik_target_world[side] = None

        ik_target_base = {}
        ik_raw_q = {}
        for side in ("left", "right"):
            tb = self.proc.last_ik_target_base.get(side)
            ik_target_base[side] = tb.copy() if tb is not None else None
            rq = self.proc.last_ik_raw.get(side)
            ik_raw_q[side] = rq.copy() if rq is not None else None

        self._log_samples.append({
            "time": now,
            "webxr_raw": self._latest_raw_hand_data,
            "robot_sites": robot_sites,
            "robot_qpos": self.data.qpos.copy(),
            "offsets": offsets_copy,
            "ik_diag": ik_diag_copy,
            "ik_target_world": ik_target_world,
            "ik_target_base": ik_target_base,
            "ik_raw_q": ik_raw_q,
            "canonical_q": {s: self.canonical_q[s].copy() for s in ("left", "right")},
            "arm_qpos": {s: self.data.qpos[self.arm_idx[s]].copy() for s in ("left", "right")},
            "collision_info": self._collision_info.copy(),
        })

    # -- Viewer key callback -----------------------------------------------

    def _on_key(self, key):
        if key == ord("f"):
            self.stream_frames = not self.stream_frames
            state = "enabled" if self.stream_frames else "disabled"
            print(f"Frame streaming {state}.")
            return
        if key != 32:
            return
        if not self.active:
            self.start()
        else:
            self.stop()

    # -- Socket server -----------------------------------------------------

    def _handle_socket_request(self, conn, addr):
        """Handle one request from a connected client. Protocol: one line per request."""
        try:
            data = conn.recv(4096).decode("utf-8").strip()
            if not data:
                return
            parts = data.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "is_ready":
                resp = json.dumps({"ready": self.is_ready})
            elif cmd == "stop":
                self.stop()
                resp = json.dumps({"ok": True})
            elif cmd == "get_qpos":
                order = arg if arg in ("mujoco", "genesis") else "genesis"
                # if not self.is_ready:
                #     resp = json.dumps({"error": "not ready"})
                # else:
                qpos = self.get_qpos(order=order)
                resp = json.dumps({"qpos": qpos.tolist()})
            else:
                resp = json.dumps({"error": f"unknown command: {cmd}"})
            conn.sendall((resp + "\n").encode("utf-8"))
        except Exception as e:
            try:
                conn.sendall(
                    (json.dumps({"error": str(e)}) + "\n").encode("utf-8")
                )
            except Exception:
                pass
        finally:
            conn.close()

    def _run_socket_server(self):
        """Run the TCP socket server in a loop. Accepts connections and handles requests."""
        self._socket_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._socket_server_sock.bind((self.socket_host, self.socket_port))
        except OSError as e:
            print(f"Socket server bind failed: {e}")
            return
        self._socket_server_sock.listen(5)
        self._socket_server_sock.settimeout(0.5)
        print(f"Teleop socket server listening on {self.socket_host}:{self.socket_port}")

        while not self._stop_event.is_set():
            try:
                conn, addr = self._socket_server_sock.accept()
                threading.Thread(
                    target=self._handle_socket_request,
                    args=(conn, addr),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break
        try:
            self._socket_server_sock.close()
        except Exception:
            pass
        self._socket_server_sock = None

    def _start_socket_server(self):
        """Start the socket server in a background thread."""
        if self._socket_server_thread is not None:
            return
        self._socket_server_thread = threading.Thread(
            target=self._run_socket_server,
            daemon=True,
            name="TeleopSocketServer",
        )
        self._socket_server_thread.start()

    def _stop_socket_server(self):
        """Stop the socket server."""
        self._stop_event.set()
        if self._socket_server_sock:
            try:
                self._socket_server_sock.close()
            except Exception:
                pass
        if self._socket_server_thread is not None:
            self._socket_server_thread.join(timeout=2.0)
            self._socket_server_thread = None

    # -- Step & run --------------------------------------------------------

    def step(self, viewer=None):
        """Execute one tick of the teleop loop. Returns the current robot qpos.

        If *viewer* is provided, also renders keypoints and syncs the viewer.
        """
        now = time.time()

        if not self.active:
            self._tick_idle(now)
        elif not self._warmup_done:
            elapsed = now - self._t_start
            if elapsed < WARMUP_SECONDS:
                self._tick_warmup(now)
            else:
                self._finalize_warmup()
                self._tick_tracking(now)
        else:
            self._tick_tracking(now)

        with self._qpos_lock:
            self._latest_qpos = self.data.qpos[self.free_nq:].copy()

        if viewer is not None:
            draw_webxr_keypoints(viewer, self._latest_raw_hand_data, self.offsets)
            viewer.sync()

        return self.get_qpos()

    def _run_loop(self):
        """Internal loop body — launched directly or on a background thread."""
        self._stop_event.clear()
        self._start_socket_server()

        for side in ("left", "right"):
            for jn, pos in self.canonical[side].items():
                print(f"  canonical {side} {jn}: {pos}")

        if fetch_hands() is None:
            print("Warning: hand tracker not reachable.")
        else:
            print("Hand tracker connected.")
        print("\nSPACE = start/stop  |  F = toggle streaming  |  ESC = quit\n")

        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._on_key,
        ) as v:
            while v.is_running() and not self._stop_event.is_set():
                self.step(viewer=v)
                time.sleep(0.001)

    def run(self, blocking=True):
        """Launch the passive MuJoCo viewer and run the teleop loop.

        Parameters
        ----------
        blocking : bool
            If True (default), blocks the calling thread until the viewer
            is closed.  If False, the loop runs on a daemon background
            thread so the caller can continue — e.g. to poll ``get_qpos()``
            from the main thread.
        """
        self._stop_event.clear()
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(
                target=self._run_loop, daemon=True, name="TeleopTracker",
            )
            self._thread.start()

    @property
    def is_running(self):
        """True while the background loop thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def shutdown(self, timeout=5.0):
        """Signal the background loop to stop and wait for it to finish.

        Safe to call even if ``run()`` was called with ``blocking=True``
        or has already exited.
        """
        self._stop_event.set()
        if self._socket_server_sock:
            try:
                self._socket_server_sock.close()
            except Exception:
                pass
        if self._socket_server_thread is not None:
            self._socket_server_thread.join(timeout=2.0)
            self._socket_server_thread = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf-path", type=str, default=None)
    parser.add_argument(
        "--reanchor", action="store_true", default=True,
        help="Re-anchor wrist offset on large IK jumps instead of freezing",
    )
    parser.add_argument(
        "--raw-ik", action="store_true", default=False,
        help="Disable smoothing/interpolation and run plain per-frame IK",
    )
    parser.add_argument(
        "--no-stream-frames", action="store_true", default=False,
        help="Disable streaming MuJoCo frames to the WebXR server",
    )
    parser.add_argument(
        "--log-npy-path", type=str, default="hand_teleop_log.npz",
        help="Path to save hand/end-effector log (npz, pickled objects)",
    )
    parser.add_argument(
        "--socket-host", type=str, default="127.0.0.1",
        help="Host for the teleop socket server",
    )
    parser.add_argument(
        "--socket-port", type=int, default=9004,
        help="Port for the teleop socket server",
    )
    parser.add_argument(
        "--no-collision-check", action="store_true", default=False,
        help="Disable robot self/arm-arm collision rejection",
    )
    args = parser.parse_args()

    tracker = TeleopTracker(
        mjcf_path=args.mjcf_path,
        reanchor=args.reanchor,
        raw_ik=args.raw_ik,
        stream_frames=not args.no_stream_frames,
        enable_collision_check=not args.no_collision_check,
        log_npy_path=args.log_npy_path,
        socket_host=args.socket_host,
        socket_port=args.socket_port,
    )
    tracker.run(blocking=True)
    tracker.shutdown()


if __name__ == "__main__":
    main()
