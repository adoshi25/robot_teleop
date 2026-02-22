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
import sys
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
    Path(__file__).parent.parent / "teleop" / "robots" / "dg5f_dual_panda.mjcf.xml"
)
CONTROL_HZ = 60
WARMUP_SECONDS = 2.5
MAX_JOINT_DELTA = 0.5  # radians — reject IK solutions that jump more than this per frame

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
# j_idx is the joint index within the 4-joint chain that controls abduction.
# Index/middle/ring: joint 0 (axis X). Pinky: joint 1 (lj_dg_5_2, axis X).
_ABD_FINGERS = [
    ("index",  "index-finger-phalanx-proximal",  "index-finger-phalanx-intermediate",  0),
    ("middle", "middle-finger-phalanx-proximal",  "middle-finger-phalanx-intermediate", 0),
    ("ring",   "ring-finger-phalanx-proximal",    "ring-finger-phalanx-intermediate",   0),
    ("pinky",  "pinky-finger-phalanx-proximal",   "pinky-finger-phalanx-intermediate",  1),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def webxr_to_mujoco(pos_dict):
    """WebXR {x,y,z} → MuJoCo frame (-z, x, y).

    WebXR: X=right, Y=up, Z=toward user.
    MuJoCo scene: X=forward, Y=left/right, Z=up.
    """
    return np.array([-pos_dict["x"], pos_dict["z"], pos_dict["y"]])


def webxr_quat_to_mujoco_wxyz(ori_dict):
    """WebXR orientation {x,y,z,w} (xyzw) → MuJoCo wxyz quaternion.

    The position axes remap as (-X, Z, Y) which is a proper rotation.
    We rotate the WebXR orientation into the MuJoCo basis via
    q_m = q_map ⊗ q_web ⊗ q_map^{-1}.
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


def _xmat_to_wxyz(xmat):
    """Convert a 3x3 rotation matrix (flattened 9-vec from MuJoCo) to wxyz quaternion."""
    mat = np.array(xmat, dtype=float).reshape(3, 3)
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(mat)
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


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
    """Offset-corrected wrist pose [x, y, z, w, qx, qy, qz] in world frame, or None.

    Position: same as get_offset_wrist.
    Orientation: q_delta * q_cal_robot, where q_delta = q_cur * q_cal^{-1}.
    When the user's hand hasn't rotated, the target orientation equals the
    canonical robot wrist orientation, ensuring zero IK error at rest.
    """
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


def _get_pos(hand_joints, name):
    kp = hand_joints.get(name)
    if kp and "position" in kp:
        return webxr_to_mujoco(kp["position"])
    return None


def _build_palm_frame(hand_joints):
    """Build an orthonormal palm frame (forward, lateral, normal) from keypoints.

    forward  = wrist → middle-finger-metacarpal  (along fingers)
    lateral  = index-metacarpal → pinky-metacarpal (orthogonalised)
    normal   = cross(forward, lateral)             (out of palm)

    Returns (forward, lateral, normal) unit vectors, or None.
    """
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
    """Record per-finger lateral angles at calibration (rest) pose.

    Returns {(side, finger_name): rest_angle}.
    """
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
    """Compute abduction from lateral deviation of proximal phalanx.

    For each finger, projects the proximal bone onto the palm plane and
    measures its lateral angle.  The abduction is the change from the
    calibration-time rest angle, which naturally handles the default
    finger splay.
    """
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
    """Render WebXR keypoints: raw positions as blue dots, offset-corrected as red dots."""
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf-path", type=str, default=None)
    parser.add_argument("--reanchor", action="store_true", default=True,
                        help="Re-anchor wrist offset on large IK jumps instead of freezing")
    parser.add_argument(
        "--raw-ik",
        action="store_true",
        default=False,
        help="Disable smoothing/interpolation and run plain per-frame IK",
    )
    parser.add_argument(
        "--no-stream-frames",
        action="store_true",
        default=False,
        help="Disable streaming MuJoCo frames to the WebXR server",
    )
    parser.add_argument(
        "--log-npy-path",
        type=str,
        default="hand_teleop_log.npz",
        help="Path to save hand/end-effector log (npz, pickled objects)",
    )
    args = parser.parse_args()

    mjcf_path = Path(args.mjcf_path) if args.mjcf_path else DEFAULT_MJCF
    if not mjcf_path.exists():
        print(f"Error: {mjcf_path} not found")
        return

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    free_nq = 0
    free_nv = 0
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_nq += 7
            free_nv += 6

    # Offscreen renderer for streaming to VR headset
    offscreen = mujoco.Renderer(model, height=OFFSCREEN_HEIGHT, width=OFFSCREEN_WIDTH)
    offscreen_cam = mujoco.MjvCamera()
    offscreen_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    offscreen_cam.distance = 2.0
    offscreen_cam.azimuth = 270
    offscreen_cam.elevation = -30
    offscreen_cam.lookat[:] = [-0, 0, 1]
    last_frame_post = [0.0]
    stream_frames = [not args.no_stream_frames]

    def maybe_post_frame(t):
        if not stream_frames[0]:
            return
        if t - last_frame_post[0] < 1.0 / FRAME_POST_HZ:
            return
        try:
            offscreen.update_scene(data, camera=offscreen_cam)
            pixels = offscreen.render()
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
        last_frame_post[0] = t

    canonical = read_canonical_positions(model, data)
    for side in ("left", "right"):
        for jn, pos in canonical[side].items():
            print(f"  canonical {side} {jn}: {pos}")

    # IK processor
    if args.raw_ik:
        proc = PandaArmTrajectoryProcessor(
            scene_path=mjcf_path,
            control_hz=CONTROL_HZ,
            min_point_distance=0.0,
            max_point_distance=1e9,
            interpolation_points=0,
            smoothing_alpha=1.0,
            smoothing_sigma=0.0,
            ori_weight=5.0,
        )
    else:
        proc = PandaArmTrajectoryProcessor(
            scene_path=mjcf_path, control_hz=CONTROL_HZ, ori_weight=5.0,
        )
    arm_idx = {s: proc.get_mujoco_qpos_indices(s) for s in ("left", "right")}
    canonical_q = {s: data.qpos[arm_idx[s]].copy() for s in ("left", "right")}
    ik_q = {"left": None, "right": None}
    prev_ik_q = {"left": None, "right": None}

    # Finger retargeting
    finger_map = build_finger_index_map(model)
    finger_qpos = {"left": {}, "right": {}}
    abd_rest = [{}]

    reanchor = {"left": False, "right": False}
    latest_webxr_wrist = {"left": None, "right": None}

    ik_diag = {"left": {}, "right": {}}

    def on_ik(side, q, _t):
        prev = prev_ik_q[side]
        if prev is not None:
            max_delta = float(np.max(np.abs(q - prev)))
            rejected = max_delta > MAX_JOINT_DELTA
            ik_diag[side] = {
                "q_smoothed": q.copy(),
                "max_delta": max_delta,
                "rejected": rejected,
            }
            if rejected:
                if args.reanchor:
                    reanchor[side] = True
                return
        else:
            ik_diag[side] = {
                "q_smoothed": q.copy(),
                "max_delta": 0.0,
                "rejected": False,
            }
        ik_q[side] = q.copy()
        prev_ik_q[side] = q.copy()

    proc.register_callback(on_ik)

    # State -----------------------------------------------------------------
    active = [False]
    t_start = [0.0]
    offsets = [{}]
    warmup_done = [False]
    warmup_data = [None]
    latest_raw_hand_data = [None]
    last_fetch = [0.0]
    dt = 1.0 / CONTROL_HZ
    log_samples = []

    def on_key(key):
        if key == ord("f"):
            stream_frames[0] = not stream_frames[0]
            state = "enabled" if stream_frames[0] else "disabled"
            print(f"Frame streaming {state}.")
            return
        if key != 32:
            return
        if not active[0]:
            active[0] = True
            t_start[0] = time.time()
            offsets[0] = {}
            warmup_done[0] = False
            warmup_data[0] = None
            proc.start_trajectory()
            ik_q["left"] = ik_q["right"] = None
            prev_ik_q["left"] = canonical_q["left"].copy()
            prev_ik_q["right"] = canonical_q["right"].copy()
            finger_qpos["left"] = finger_qpos["right"] = {}
            abd_rest[0] = {}
            reanchor["left"] = reanchor["right"] = False
            latest_webxr_wrist["left"] = latest_webxr_wrist["right"] = None
            log_samples.clear()
            print(f"Warmup {WARMUP_SECONDS}s — hold hands steady...")
        else:
            active[0] = False
            proc.clear_trajectory()
            ik_q["left"] = ik_q["right"] = None
            prev_ik_q["left"] = prev_ik_q["right"] = None
            finger_qpos["left"] = finger_qpos["right"] = {}
            abd_rest[0] = {}
            reanchor["left"] = reanchor["right"] = False
            latest_webxr_wrist["left"] = latest_webxr_wrist["right"] = None
            print("Stopped.")
            if log_samples:
                np.savez_compressed(
                    args.log_npy_path,
                    samples=np.array(log_samples, dtype=object),
                )
                print(f"Saved log: {args.log_npy_path} ({len(log_samples)} samples)")

    if fetch_hands() is None:
        print("Warning: hand tracker not reachable.")
    else:
        print("Hand tracker connected.")
    print("\nSPACE = start/stop  |  F = toggle streaming  |  ESC = quit\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        while v.is_running():
            now = time.time()

            # ------ idle ---------------------------------------------------
            if not active[0]:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                sync_mocap_to_sites(model, data)
                maybe_post_frame(now)
                draw_webxr_keypoints(v, latest_raw_hand_data[0], offsets[0])
                v.sync()
                time.sleep(0.001)
                continue

            elapsed = now - t_start[0]

            # ------ warmup -------------------------------------------------
            if elapsed < WARMUP_SECONDS:
                hd = fetch_hands()
                if hd:
                    warmup_data[0] = hd
                    latest_raw_hand_data[0] = hd
                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                sync_mocap_to_sites(model, data)
                maybe_post_frame(now)
                draw_webxr_keypoints(v, latest_raw_hand_data[0], offsets[0])
                v.sync()
                time.sleep(0.001)
                continue

            # ------ warmup → tracking transition --------------------------
            if not warmup_done[0]:
                warmup_done[0] = True
                hd = warmup_data[0] or fetch_hands()
                if hd:
                    offsets[0] = compute_offsets(hd, canonical)
                    for side, so in offsets[0].items():
                        w = so.get("wrist")
                        if w is not None:
                            print(f"  {side} wrist offset: {w}")
                    for side in ("left", "right"):
                        wrist_target = get_offset_wrist(hd, side, offsets[0])
                        canon_w = canonical.get(side, {}).get("wrist")
                        if wrist_target is not None and canon_w is not None:
                            err = np.linalg.norm(wrist_target - canon_w)
                            print(f"  {side} calibration verify: target_err={err:.4f}")
                    abd_rest[0] = calibrate_abduction_rest(hd)
                    print(f"  abduction rest angles: {abd_rest[0]}")
                print("Tracking active!")

            # ------ tracking -----------------------------------------------
            if now - last_fetch[0] >= dt:
                hand_data = fetch_hands()
                if hand_data:
                    latest_raw_hand_data[0] = hand_data
                if hand_data and offsets[0]:
                    for side in ("left", "right"):
                        joints = hand_data.get(side)
                        if joints:
                            w = joints.get("wrist")
                            if w and "position" in w:
                                latest_webxr_wrist[side] = w["position"]

                        if (reanchor[side]
                                and latest_webxr_wrist[side] is not None
                                and side in offsets[0]):
                            site_name = ROBOT_SITES[side]["wrist"]
                            robot_pos = data.site_xpos[
                                model.site(site_name).id
                            ].copy()
                            new_off = robot_pos - webxr_to_mujoco(
                                latest_webxr_wrist[side]
                            )
                            offsets[0][side]["wrist"] = new_off
                            prev_ik_q[side] = data.qpos[arm_idx[side]].copy()
                            reanchor[side] = False

                        wrist_pose = get_offset_wrist_pose(hand_data, side, offsets[0])
                        if wrist_pose is not None:
                            proc.add_point(wrist_pose, side)
                        if joints:
                            finger_qpos[side] = retarget_fingers(
                                joints, side, finger_map, abd_rest[0]
                            )
                last_fetch[0] = now

            for side in ("left", "right"):
                q = ik_q[side]
                if q is not None:
                    data.qpos[arm_idx[side]] = q
                for qidx, angle in finger_qpos[side].items():
                    data.qpos[qidx] = angle

            robot_qpos = data.qpos[free_nq:].copy()
            mujoco.mj_step(model, data)
            data.qpos[free_nq:] = robot_qpos
            data.qvel[free_nv:] = 0
            mujoco.mj_forward(model, data)
            sync_mocap_to_sites(model, data)
            maybe_post_frame(now)
            draw_webxr_keypoints(v, latest_raw_hand_data[0], offsets[0])

            if latest_raw_hand_data[0]:
                robot_sites = {}
                for side in ("left", "right"):
                    robot_sites[side] = {}
                    for jname, site_name in ROBOT_SITES[side].items():
                        sid = model.site(site_name).id
                        robot_sites[side][jname] = data.site_xpos[sid].copy()
                offsets_copy = {
                    s: {k: (v.copy() if hasattr(v, 'copy') else v)
                        for k, v in d.items()}
                    for s, d in offsets[0].items()
                } if offsets[0] else {}
                ik_diag_copy = {}
                for side in ("left", "right"):
                    d = ik_diag.get(side, {})
                    ik_diag_copy[side] = {
                        k: (v.copy() if hasattr(v, 'copy') else v)
                        for k, v in d.items()
                    }
                ik_target_world = {}
                for side in ("left", "right"):
                    raw = latest_raw_hand_data[0].get(side, {})
                    raw_wrist = raw.get("wrist", {})
                    off = offsets[0].get(side, {}).get("wrist")
                    if "position" in raw_wrist and off is not None:
                        pos_mj = webxr_to_mujoco(raw_wrist["position"])
                        ik_target_world[side] = (pos_mj + off).copy()
                    else:
                        ik_target_world[side] = None
                ik_target_base = {}
                ik_raw_q = {}
                for side in ("left", "right"):
                    tb = proc.last_ik_target_base.get(side)
                    ik_target_base[side] = tb.copy() if tb is not None else None
                    rq = proc.last_ik_raw.get(side)
                    ik_raw_q[side] = rq.copy() if rq is not None else None
                log_samples.append({
                    "time": now,
                    "webxr_raw": latest_raw_hand_data[0],
                    "robot_sites": robot_sites,
                    "robot_qpos": data.qpos.copy(),
                    "offsets": offsets_copy,
                    "ik_diag": ik_diag_copy,
                    "ik_target_world": ik_target_world,
                    "ik_target_base": ik_target_base,
                    "ik_raw_q": ik_raw_q,
                    "canonical_q": {s: canonical_q[s].copy() for s in ("left", "right")},
                    "arm_qpos": {s: data.qpos[arm_idx[s]].copy() for s in ("left", "right")},
                })

            v.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
