"""Load MuJoCo scene with two robot arms and show the viewer."""
from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "tesollo_hand 2" / "robot_scene_combined.xml"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print(f"Loading model from {MODEL_PATH}...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    print("Launching MuJoCo viewer. Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()
