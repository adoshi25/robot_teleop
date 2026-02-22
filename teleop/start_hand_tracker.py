from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import json
import logging
import os
import socket
import threading
import time
from datetime import datetime
import faulthandler

os.environ['USE_NGROK'] = '0'
os.environ['HAND_TRACKER_DEBUG'] = '1'
app = Flask(__name__)
CORS(app)

# Store latest hand data
latest_hand_data = {}
last_hand_data_at = None

# Store latest MuJoCo offscreen frame (JPEG bytes)
_latest_frame = None
_frame_lock = threading.Lock()

logger = logging.getLogger("hand_tracker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
)

_stats_lock = threading.Lock()
_stats = {
    "hand_data_count": 0,
    "last_hand_data_at": None,
    "last_request_from": None,
    "last_user_agent": None,
    "last_request_path": None,
    "last_content_length": 0,
    "inflight": 0,
    "last_request_duration_ms": 0.0,
    "max_request_duration_ms": 0.0,
    "last_error": None,
}
_last_log_time = 0.0
_debug_enabled = os.environ.get("HAND_TRACKER_DEBUG", "1") == "1"
_debug_tracebacks = os.environ.get("HAND_TRACKER_DEBUG_TRACE", "0") == "1"


def _maybe_log_debug(message: str) -> None:
    if _debug_enabled:
        logger.info(message)


def _start_watchdog() -> None:
    if not _debug_enabled:
        return

    def _watchdog_loop() -> None:
        while True:
            time.sleep(10)
            with _stats_lock:
                count = _stats["hand_data_count"]
                last_at = _stats["last_hand_data_at"]
                last_duration = _stats["last_request_duration_ms"]
                max_duration = _stats["max_request_duration_ms"]
                last_error = _stats["last_error"]
            if last_at:
                age = (datetime.now() - last_at).total_seconds()
                _maybe_log_debug(
                    f"watchdog: last_hand_data_age={age:.1f}s "
                    f"count={count} last_req_ms={last_duration:.1f} "
                    f"max_req_ms={max_duration:.1f} last_error={last_error}"
                )
            else:
                _maybe_log_debug("watchdog: no hand data received yet")

    thread = threading.Thread(target=_watchdog_loop, name="watchdog", daemon=True)
    thread.start()


@app.before_request
def _track_inflight() -> None:
    with _stats_lock:
        _stats["inflight"] += 1
        _stats["last_request_path"] = request.path


@app.after_request
def _track_outflight(response):
    with _stats_lock:
        _stats["inflight"] = max(0, _stats["inflight"] - 1)
    return response

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebXR Hand Tracking</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; overflow: hidden; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 5px;
            max-width: 300px;
            z-index: 100;
        }
        #startButton {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            padding: 20px 40px;
            font-size: 20px;
            cursor: pointer;
            z-index: 100;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 10px;
        }
        #startButton:hover {
            background: #45a049;
        }
        #container { width: 100%; height: 100vh; }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>WebXR Hand Tracking</h3>
        <p id="status">Click "Enter VR" to start</p>
        <pre id="data">Waiting for hand data...</pre>
    </div>
    <button id="startButton">Enter VR</button>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        let session = null;
        let refSpace = null;
        let renderer, scene, camera;
        let handMeshes = { left: {}, right: {} };
        let mjCanvas, mjCtx, mjTexture, mjPanel;
        let mjFetchInFlight = false;
        let lastMjFetch = 0;
        const mjFetchIntervalMs = 100;
        const status = document.getElementById('status');
        const dataDisplay = document.getElementById('data');
        const startButton = document.getElementById('startButton');
        const container = document.getElementById('container');
        const sendIntervalMs = 33;
        let inFlight = false;
        let lastSendAt = 0;
        let lastOkAt = 0;
        let lastErrAt = 0;
        let errorCount = 0;
        let droppedCount = 0;

        // Joint names for WebXR hand tracking
        const jointNames = [
            'wrist',
            'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
            'index-finger-metacarpal', 'index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate', 'index-finger-phalanx-distal', 'index-finger-tip',
            'middle-finger-metacarpal', 'middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal', 'middle-finger-tip',
            'ring-finger-metacarpal', 'ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal', 'ring-finger-tip',
            'pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal', 'pinky-finger-tip'
        ];

        // Connections between joints for drawing bones
        const connections = [
            ['wrist', 'thumb-metacarpal'],
            ['thumb-metacarpal', 'thumb-phalanx-proximal'],
            ['thumb-phalanx-proximal', 'thumb-phalanx-distal'],
            ['thumb-phalanx-distal', 'thumb-tip'],
            
            ['wrist', 'index-finger-metacarpal'],
            ['index-finger-metacarpal', 'index-finger-phalanx-proximal'],
            ['index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate'],
            ['index-finger-phalanx-intermediate', 'index-finger-phalanx-distal'],
            ['index-finger-phalanx-distal', 'index-finger-tip'],
            
            ['wrist', 'middle-finger-metacarpal'],
            ['middle-finger-metacarpal', 'middle-finger-phalanx-proximal'],
            ['middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate'],
            ['middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal'],
            ['middle-finger-phalanx-distal', 'middle-finger-tip'],
            
            ['wrist', 'ring-finger-metacarpal'],
            ['ring-finger-metacarpal', 'ring-finger-phalanx-proximal'],
            ['ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate'],
            ['ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal'],
            ['ring-finger-phalanx-distal', 'ring-finger-tip'],
            
            ['wrist', 'pinky-finger-metacarpal'],
            ['pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal'],
            ['pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate'],
            ['pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal'],
            ['pinky-finger-phalanx-distal', 'pinky-finger-tip'],
            
            // Palm connections
            ['index-finger-metacarpal', 'middle-finger-metacarpal'],
            ['middle-finger-metacarpal', 'ring-finger-metacarpal'],
            ['ring-finger-metacarpal', 'pinky-finger-metacarpal'],
        ];

        function initThreeJS() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 100);
            camera.matrixAutoUpdate = false; // Important for WebXR!
            
            // Add lighting
            const light = new THREE.DirectionalLight(0xffffff, 1);
            light.position.set(1, 1, 1);
            scene.add(light);
            
            const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
            light2.position.set(-1, -1, -1);
            scene.add(light2);
            
            scene.add(new THREE.AmbientLight(0x606060));
            
            // Add a reference grid
            const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
            gridHelper.position.y = -0.5;
            scene.add(gridHelper);
            
            // MuJoCo viewer panel
            mjCanvas = document.createElement('canvas');
            mjCanvas.width = 640;
            mjCanvas.height = 480;
            mjCtx = mjCanvas.getContext('2d');
            mjCtx.fillStyle = '#111';
            mjCtx.fillRect(0, 0, 640, 480);
            mjCtx.fillStyle = '#666';
            mjCtx.font = '24px Arial';
            mjCtx.textAlign = 'center';
            mjCtx.fillText('Waiting for MuJoCo...', 320, 240);

            mjTexture = new THREE.CanvasTexture(mjCanvas);
            mjTexture.minFilter = THREE.LinearFilter;

            const borderGeom = new THREE.PlaneGeometry(1.0, 0.77);
            const borderMat = new THREE.MeshBasicMaterial({ color: 0x222222 });
            const borderMesh = new THREE.Mesh(borderGeom, borderMat);
            borderMesh.position.set(0, 1.4, -1.5);
            scene.add(borderMesh);

            const panelGeom = new THREE.PlaneGeometry(0.96, 0.72);
            const panelMat = new THREE.MeshBasicMaterial({ map: mjTexture });
            mjPanel = new THREE.Mesh(panelGeom, panelMat);
            mjPanel.position.set(0, 1.4, -1.49);
            scene.add(mjPanel);

            // Create hand visualizations
            for (const hand of ['left', 'right']) {
                const color = hand === 'left' ? 0x00ff88 : 0x00aaff;
                const emissiveColor = hand === 'left' ? 0x003322 : 0x002233;
                
                // Create joint spheres
                for (const jointName of jointNames) {
                    const isTip = jointName.includes('tip');
                    const isWrist = jointName === 'wrist';
                    const radius = isWrist ? 0.015 : (isTip ? 0.008 : 0.01);
                    
                    const geometry = new THREE.SphereGeometry(radius, 16, 16);
                    const material = new THREE.MeshPhongMaterial({ 
                        color: color,
                        emissive: emissiveColor,
                        shininess: 100
                    });
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.visible = false;
                    scene.add(sphere);
                    handMeshes[hand][jointName] = sphere;
                }
                
                // Create bone cylinders instead of lines for better visibility
                handMeshes[hand].bones = [];
                for (const [start, end] of connections) {
                    // Use a cylinder for each bone
                    const geometry = new THREE.CylinderGeometry(0.004, 0.004, 1, 8);
                    geometry.translate(0, 0.5, 0);
                    geometry.rotateX(Math.PI / 2);
                    
                    const material = new THREE.MeshPhongMaterial({ 
                        color: color,
                        emissive: emissiveColor,
                        shininess: 50
                    });
                    const cylinder = new THREE.Mesh(geometry, material);
                    cylinder.visible = false;
                    scene.add(cylinder);
                    handMeshes[hand].bones.push({ mesh: cylinder, start, end });
                }
            }
        }

        async function sendHandData(handData) {
            const now = Date.now();
            if (inFlight || now - lastSendAt < sendIntervalMs) {
                droppedCount += 1;
                return;
            }
            lastSendAt = now;
            inFlight = true;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000);
            try {
                const response = await fetch('/hand_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(handData),
                    signal: controller.signal
                });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                lastOkAt = Date.now();
            } catch (e) {
                lastErrAt = Date.now();
                errorCount += 1;
                console.error('Error sending hand data:', e);
            } finally {
                clearTimeout(timeoutId);
                inFlight = false;
            }
        }

        function extractHandData(frame, hand, refSpace) {
            const joints = {};
            
            for (const jointName of jointNames) {
                const joint = hand.get(jointName);
                if (joint) {
                    const pose = frame.getJointPose(joint, refSpace);
                    if (pose) {
                        joints[jointName] = {
                            position: {
                                x: pose.transform.position.x,
                                y: pose.transform.position.y,
                                z: pose.transform.position.z
                            },
                            orientation: {
                                x: pose.transform.orientation.x,
                                y: pose.transform.orientation.y,
                                z: pose.transform.orientation.z,
                                w: pose.transform.orientation.w
                            },
                            radius: pose.radius
                        };
                    }
                }
            }
            
            return joints;
        }

        function updateHandVisuals(handData) {
            // Hide all joints and bones first
            for (const hand of ['left', 'right']) {
                for (const jointName of jointNames) {
                    if (handMeshes[hand][jointName]) {
                        handMeshes[hand][jointName].visible = false;
                    }
                }
                if (handMeshes[hand].bones) {
                    for (const bone of handMeshes[hand].bones) {
                        bone.mesh.visible = false;
                    }
                }
            }
            
            // Update visible hands
            for (const [handName, joints] of Object.entries(handData.hands)) {
                if (!handMeshes[handName]) continue;
                
                // Update joint positions
                for (const [jointName, data] of Object.entries(joints)) {
                    const sphere = handMeshes[handName][jointName];
                    if (sphere && data && data.position) {
                        sphere.position.set(data.position.x, data.position.y, data.position.z);
                        sphere.visible = true;
                    }
                }
                
                // Update bone cylinders
                if (handMeshes[handName].bones) {
                    for (const bone of handMeshes[handName].bones) {
                        const startJoint = joints[bone.start];
                        const endJoint = joints[bone.end];
                        
                        if (startJoint && endJoint && startJoint.position && endJoint.position) {
                            const startPos = new THREE.Vector3(
                                startJoint.position.x,
                                startJoint.position.y,
                                startJoint.position.z
                            );
                            const endPos = new THREE.Vector3(
                                endJoint.position.x,
                                endJoint.position.y,
                                endJoint.position.z
                            );
                            
                            // Position at start
                            bone.mesh.position.copy(startPos);
                            
                            // Calculate direction and length
                            const direction = new THREE.Vector3().subVectors(endPos, startPos);
                            const length = direction.length();
                            
                            if (length > 0.001) {
                                // Scale to correct length
                                bone.mesh.scale.set(1, 1, length);
                                
                                // Orient towards end point
                                bone.mesh.lookAt(endPos);
                                
                                bone.mesh.visible = true;
                            }
                        }
                    }
                }
            }
        }

        function fetchMjFrame() {
            const now = Date.now();
            if (mjFetchInFlight || now - lastMjFetch < mjFetchIntervalMs) return;
            lastMjFetch = now;
            mjFetchInFlight = true;

            fetch('/mujoco_frame')
                .then(resp => {
                    if (!resp.ok || resp.status === 204) throw new Error('no frame');
                    return resp.blob();
                })
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    const img = new Image();
                    img.onload = () => {
                        mjCtx.drawImage(img, 0, 0, mjCanvas.width, mjCanvas.height);
                        mjTexture.needsUpdate = true;
                        URL.revokeObjectURL(url);
                        mjFetchInFlight = false;
                    };
                    img.onerror = () => {
                        URL.revokeObjectURL(url);
                        mjFetchInFlight = false;
                    };
                    img.src = url;
                })
                .catch(() => { mjFetchInFlight = false; });
        }

        function onXRFrame(time, frame) {
            session.requestAnimationFrame(onXRFrame);
            fetchMjFrame();
            
            const pose = frame.getViewerPose(refSpace);
            if (!pose) return;
            
            const inputSources = session.inputSources;
            const handData = { timestamp: Date.now(), hands: {} };
            
            for (const source of inputSources) {
                if (source.hand) {
                    const handedness = source.handedness;
                    handData.hands[handedness] = extractHandData(frame, source.hand, refSpace);
                }
            }
            
            if (Object.keys(handData.hands).length > 0) {
                sendHandData(handData);
                updateHandVisuals(handData);
                
                let display = `Active: ${Object.keys(handData.hands).join(', ')}\n`;
                for (const [hand, joints] of Object.entries(handData.hands)) {
                    if (joints.wrist) {
                        const w = joints.wrist;
                        display += `${hand}: (${w.position.x.toFixed(2)}, ${w.position.y.toFixed(2)}, ${w.position.z.toFixed(2)})\n`;
                    }
                }
                const now = Date.now();
                const okAge = lastOkAt ? ((now - lastOkAt) / 1000).toFixed(1) : 'n/a';
                const errAge = lastErrAt ? ((now - lastErrAt) / 1000).toFixed(1) : 'n/a';
                display += `send: inFlight=${inFlight} dropped=${droppedCount} errors=${errorCount}\n`;
                display += `last ok: ${okAge}s ago | last err: ${errAge}s ago\n`;
                dataDisplay.textContent = display;
            }
            
            const glLayer = session.renderState.baseLayer;
            const gl = renderer.getContext();
            
            gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            
            for (const view of pose.views) {
                const viewport = glLayer.getViewport(view);
                gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
                
                // Properly set camera matrices for WebXR
                camera.matrix.fromArray(view.transform.matrix);
                camera.projectionMatrix.fromArray(view.projectionMatrix);
                camera.matrixWorldNeedsUpdate = true;
                camera.updateMatrixWorld(true);
                
                renderer.render(scene, camera);
            }
        }

        async function initXR() {
            if (!navigator.xr) {
                status.textContent = 'WebXR not supported in this browser';
                return;
            }

            const supported = await navigator.xr.isSessionSupported('immersive-vr');
            if (!supported) {
                status.textContent = 'Immersive VR not supported';
                return;
            }

            try {
                // Create renderer and add to DOM
                renderer = new THREE.WebGLRenderer({ 
                    antialias: true,
                    alpha: true 
                });
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.xr.enabled = true;
                renderer.autoClear = false;
                container.appendChild(renderer.domElement);
                
                initThreeJS();
                
                // Request session with hand tracking
                session = await navigator.xr.requestSession('immersive-vr', {
                    requiredFeatures: ['local-floor'],
                    optionalFeatures: ['hand-tracking']
                });
                
                // Make context XR compatible
                const gl = renderer.getContext();
                await gl.makeXRCompatible();
                
                // Create and set XR layer
                const glLayer = new XRWebGLLayer(session, gl);
                await session.updateRenderState({ baseLayer: glLayer });
                
                startButton.style.display = 'none';
                status.textContent = 'VR Session Active - Show your hands!';
                
                // Get reference space
                refSpace = await session.requestReferenceSpace('local-floor');
                
                // Start render loop
                session.requestAnimationFrame(onXRFrame);
                
                session.addEventListener('end', () => {
                    session = null;
                    status.textContent = 'Session ended - click button to restart';
                    startButton.style.display = 'block';
                });
                
            } catch (e) {
                status.textContent = 'Error: ' + e.message;
                console.error('XR Error:', e);
            }
        }

        // Handle window resize
        window.addEventListener('resize', () => {
            if (renderer) {
                renderer.setSize(window.innerWidth, window.innerHeight);
            }
        });

        startButton.addEventListener('click', initXR);
        
        // Check for WebXR support on load
        if (navigator.xr) {
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                if (supported) {
                    status.textContent = 'WebXR supported! Click "Enter VR" to start.';
                } else {
                    status.textContent = 'Immersive VR not supported on this device.';
                    startButton.disabled = true;
                }
            });
        } else {
            status.textContent = 'WebXR not available in this browser.';
            startButton.disabled = true;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/hand_data', methods=['POST'])
def receive_hand_data():
    global latest_hand_data
    start_time = time.monotonic()
    try:
        data = request.get_json(silent=True)
        if data is None:
            content_length = request.content_length or 0
            _maybe_log_debug(
                f"hand_data: invalid JSON payload content_length={content_length}"
            )
            return jsonify({"status": "error", "reason": "invalid_json"}), 400

        remote_addr = request.headers.get("X-Forwarded-For", request.remote_addr)
        user_agent = request.user_agent.string if request.user_agent else "unknown"
        content_length = request.content_length or 0

        with _stats_lock:
            if _stats["last_request_from"] != remote_addr:
                _stats["last_request_from"] = remote_addr
                _maybe_log_debug(f"hand_data: new remote_addr={remote_addr}")
            if _stats["last_user_agent"] != user_agent:
                _stats["last_user_agent"] = user_agent
                _maybe_log_debug(f"hand_data: user_agent={user_agent}")
            _stats["last_content_length"] = content_length

        latest_hand_data = data
        now = datetime.now()

        with _stats_lock:
            _stats["hand_data_count"] += 1
            _stats["last_hand_data_at"] = now
            last_count = _stats["hand_data_count"]

        # Rate-limited debug log (at most once per second)
        global _last_log_time
        if _debug_enabled and time.monotonic() - _last_log_time >= 1.0:
            _last_log_time = time.monotonic()
            hands = data.get("hands", {})
            hand_list = ", ".join(hands.keys()) if hands else "none"
            joint_counts = {
                hand_name: len(joints) for hand_name, joints in hands.items()
            }
            _maybe_log_debug(
                f"hand_data: count={last_count} hands={hand_list} joints={joint_counts} "
                f"content_length={content_length}"
            )
            for hand_name, joints in hands.items():
                if "wrist" in joints:
                    pos = joints["wrist"]["position"]
                    _maybe_log_debug(
                        f"hand_data: {hand_name} wrist=({pos['x']:.3f}, "
                        f"{pos['y']:.3f}, {pos['z']:.3f})"
                    )
            if not hands:
                _maybe_log_debug("hand_data: payload has no hands")

        return jsonify({"status": "success"})
    except Exception as exc:
        with _stats_lock:
            _stats["last_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("hand_data: exception while processing request")
        return jsonify({"status": "error", "reason": "server_exception"}), 500
    finally:
        duration_ms = (time.monotonic() - start_time) * 1000.0
        with _stats_lock:
            _stats["last_request_duration_ms"] = duration_ms
            _stats["max_request_duration_ms"] = max(
                _stats["max_request_duration_ms"], duration_ms
            )
        if duration_ms > 50:
            _maybe_log_debug(f"hand_data: slow request {duration_ms:.1f}ms")

@app.route('/get_hand_data', methods=['GET'])
def get_hand_data():
    return jsonify(latest_hand_data)


@app.route('/mujoco_frame', methods=['POST'])
def receive_mujoco_frame():
    global _latest_frame
    with _frame_lock:
        _latest_frame = request.get_data()
    return '', 204


@app.route('/mujoco_frame', methods=['GET'])
def serve_mujoco_frame():
    with _frame_lock:
        frame = _latest_frame
    if frame is None:
        return '', 204
    return frame, 200, {'Content-Type': 'image/jpeg', 'Cache-Control': 'no-cache'}


@app.route('/stats', methods=['GET'])
def get_stats():
    with _stats_lock:
        stats_snapshot = dict(_stats)
    if stats_snapshot["last_hand_data_at"]:
        stats_snapshot["last_hand_data_at"] = stats_snapshot[
            "last_hand_data_at"
        ].isoformat()
    return jsonify(stats_snapshot)


def _get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _maybe_start_ngrok(port: int) -> str | None:
    if os.environ.get("USE_NGROK", "0") != "1":
        return None
    try:
        from pyngrok import ngrok
    except Exception as exc:
        print(f"pyngrok not available: {exc}")
        return None

    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    if authtoken:
        try:
            ngrok.set_auth_token(authtoken)
        except Exception as exc:
            print(f"Failed to set ngrok authtoken: {exc}")
            return None

    try:
        tunnel = ngrok.connect(port, "http")
    except Exception as exc:
        print(f"Failed to start ngrok tunnel: {exc}")
        return None

    return tunnel.public_url


if __name__ == '__main__':
    print("Starting server...", flush=True)
    faulthandler.enable()
    if _debug_tracebacks:
        faulthandler.dump_traceback_later(120, repeat=True)
        _maybe_log_debug("faulthandler: periodic traceback dump enabled (120s)")
    _start_watchdog()
    from waitress import serve
    host = os.environ.get("HAND_TRACKER_HOST", "0.0.0.0")
    port = int(os.environ.get("HAND_TRACKER_PORT", "9002"))
    print("=" * 50)
    print("WebXR Hand Tracking Server")
    print("=" * 50)
    print(f"Open http://localhost:{port} in a WebXR-compatible browser")
    print("Requires: VR headset with hand tracking (Meta Quest, etc.)")
    print("=" * 50)
    print(f"Listening on http://{host}:{port} (Ctrl+C to stop)", flush=True)
    lan_ip = _get_lan_ip()
    print(f"LAN address: http://{lan_ip}:{port}")
    public_url = _maybe_start_ngrok(port)
    if public_url:
        print(f"Public tunnel: {public_url}")
    serve(app, host=host, port=port, connection_limit=5000, threads=16)
