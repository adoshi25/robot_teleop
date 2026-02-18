# Run commands

**One-time setup**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Each session**

1. Activate venv (in each terminal):
```bash
cd /Users/mohitdoshi/Desktop/WebXR_RobotTeleop
source venv/bin/activate
```

2. Terminal 1 – start the server:
```bash
python chimera/tools/start_hand_tracker.py
```

3. With Quest connected over USB – forward the port:
```bash
adb reverse tcp:9002 tcp:9002
```

4. On the Quest browser open **http://localhost:9002**, tap Enter VR, show your hands.

5. Terminal 2 – start the visualizer (activate venv in this terminal too):
```bash
python chimera/tools/visualize_hand_tracking.py
```
Or for tesollo scene:
```bash
python run_visualize_tesollo.py
```
On macOS if the viewer needs mjpython:
```bash
mjpython run_visualize_tesollo.py
```
or
```bash
venv/bin/mjpython run_visualize_tesollo.py
```
