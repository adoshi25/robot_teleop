import numpy as np
import mujoco

CANONICAL_QPOS = np.array([
    0.12851247, -0.74926734,  0.01523919, -2.5314567,   1.3384545,
    1.8232623,   0.72626555, -0.4,         0.14,         0.08,
    0.1,        -0.43,        0.07,         0.44,         0.52,
    0.77,        0.03,       -0.81,         0.31,         0.51,
   -0.33,       -0.12,        0.23,        -0.29,        -1.03,
   -0.56,        1.21,       -0.30540678,  -0.79184073,   0.06812683,
   -2.5840569,  -1.3527942,   2.0385659,    0.62929636,   0.3,
   -0.08,       -0.07,       -0.04,         0.,          -0.04,
    0.23,        0.08,        0.28,          0.,           0.6,
    0.25,        0.69,       -0.02,          0.71,        -0.02,
   -0.22,       -0.82,       -0.17,         -1.33
])

SPACE_KEY = 32

# Mapping from (hand, WebXR joint) to the robot FK site used for calibration
JOINT_TO_ROBOT_SITE = {
    'left': {
        'wrist':             'left_palm_site',
        'thumb-tip':         'left_thumb_tip_site',
        'index-finger-tip':  'left_index_tip_site',
        'middle-finger-tip': 'left_middle_tip_site',
        'ring-finger-tip':   'left_ring_tip_site',
        'pinky-finger-tip':  'left_pinky_tip_site',
    },
    'right': {
        'wrist':             'right_palm_site',
        'thumb-tip':         'right_thumb_tip_site',
        'index-finger-tip':  'right_index_tip_site',
        'middle-finger-tip': 'right_middle_tip_site',
        'ring-finger-tip':   'right_ring_tip_site',
        'pinky-finger-tip':  'right_pinky_tip_site',
    },
}


class PandaCalibrator:
    """Synchronizes initial robot pose with human hand poses from Meta Quest.

    State machine:
        IDLE  --[space]--> ACTIVE  --[space]--> IDLE
    While IDLE the robot holds the canonical pose and hand tracking data is ignored.
    While ACTIVE the robot follows hand tracking input.

    On trajectory start the first WebXR frame is captured as the anchor.
    A per-joint offset is computed so that the WebXR anchor maps exactly onto the
    canonical FK site positions, and every subsequent frame is transformed by the
    same offset (i.e. purely relative motion from the initial pose).
    """

    def __init__(self) -> None:
        self._trajectory_active = False
        self._model = None
        self._data = None
        # canonical_positions[hand][joint] = np.array (x, y, z) from FK
        self._canonical_positions: dict[str, dict[str, np.ndarray]] = {
            'left': {}, 'right': {},
        }
        # offsets[hand][joint] = canonical_pos - initial_webxr_rotated
        self._offsets: dict[str, dict[str, np.ndarray]] = {
            'left': {}, 'right': {},
        }
        self._initial_captured: dict[str, bool] = {
            'left': False, 'right': False,
        }

    @property
    def trajectory_active(self) -> bool:
        return self._trajectory_active

    def init_simulator(self, model, data):
        self._model = model
        self._data = data
        self._set_canonical_pose()
        mujoco.mj_forward(model, data)
        self._compute_canonical_positions()
        print("[Calibrator] Simulator initialized with canonical pose.")
        print("[Calibrator] Press SPACE to start trajectory.")

    def _set_canonical_pose(self):
        if self._data is not None:
            self._data.qpos[:len(CANONICAL_QPOS)] = CANONICAL_QPOS
            self._data.qvel[:] = 0

    def _compute_canonical_positions(self):
        """Read FK site positions for every tracked joint in the canonical pose."""
        for hand_name, joint_map in JOINT_TO_ROBOT_SITE.items():
            for joint_name, site_name in joint_map.items():
                site_id = self._model.site(site_name).id
                pos = self._data.site_xpos[site_id].copy()
                self._canonical_positions[hand_name][joint_name] = pos
                print(f"  Canonical {hand_name} {joint_name}: {pos}")

    def hold_canonical_pose(self):
        """Call every sim step while idle to prevent drift from dynamics."""
        if not self._trajectory_active and self._data is not None:
            self._data.qpos[:len(CANONICAL_QPOS)] = CANONICAL_QPOS
            self._data.qvel[:] = 0

    # ---- per-joint calibration -------------------------------------------------

    def is_initial_captured(self, hand_name: str) -> bool:
        return self._initial_captured.get(hand_name, False)

    def capture_initial_positions(self, hand_name: str, joint_positions: dict[str, np.ndarray]):
        """Store the first WebXR reading and compute per-joint offsets.

        Parameters
        ----------
        hand_name : 'left' or 'right'
        joint_positions : {webxr_joint_name: rotated_xyz} for every tracked joint
        """
        for joint_name, webxr_pos in joint_positions.items():
            canonical = self._canonical_positions.get(hand_name, {}).get(joint_name)
            if canonical is not None:
                self._offsets[hand_name][joint_name] = canonical - webxr_pos
        self._initial_captured[hand_name] = True
        print(f"[Calibrator] Captured initial {hand_name} hand positions — offsets computed.")

    def transform_position(self, hand_name: str, joint_name: str, webxr_rotated_pos: np.ndarray) -> np.ndarray:
        """Apply the calibration offset: pos_robot = pos_webxr + offset."""
        offset = self._offsets.get(hand_name, {}).get(joint_name)
        if offset is not None:
            return webxr_rotated_pos + offset
        return webxr_rotated_pos

    # ---- state transitions ------------------------------------------------------

    def start_trajectory(self):
        self._trajectory_active = True
        self._offsets = {'left': {}, 'right': {}}
        self._initial_captured = {'left': False, 'right': False}
        print("[Calibrator] Trajectory STARTED — hand tracking data is now active.")

    def end_trajectory(self):
        self._trajectory_active = False
        self._set_canonical_pose()
        print("[Calibrator] Trajectory ENDED — robot returned to canonical pose.")
        print("[Calibrator] Press SPACE to start a new trajectory.")

    def handle_key(self, key):
        if key == SPACE_KEY:
            if self._trajectory_active:
                self.end_trajectory()
            else:
                self.start_trajectory()
