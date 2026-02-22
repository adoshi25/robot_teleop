from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Store latest hand data
latest_hand_data = {}

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
        const status = document.getElementById('status');
        const dataDisplay = document.getElementById('data');
        const startButton = document.getElementById('startButton');
        const container = document.getElementById('container');

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
            try {
                await fetch('/hand_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(handData)
                });
            } catch (e) {
                console.error('Error sending hand data:', e);
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

        function onXRFrame(time, frame) {
            session.requestAnimationFrame(onXRFrame);
            
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
    data = request.json
    latest_hand_data = data
    
    # Log received data
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Hand data received:")
    for hand_name, joints in data.get('hands', {}).items():
        print(f"  {hand_name} hand - {len(joints)} joints tracked")
        if 'wrist' in joints:
            pos = joints['wrist']['position']
            print(f"    Wrist position: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
    
    return jsonify({'status': 'success'})

@app.route('/get_hand_data', methods=['GET'])
def get_hand_data():
    return jsonify(latest_hand_data)

if __name__ == '__main__':
    print("Starting server...", flush=True)
    from waitress import serve
    print("=" * 50)
    print("WebXR Hand Tracking Server")
    print("=" * 50)
    print("Open http://localhost:9002 in a WebXR-compatible browser")
    print("Requires: VR headset with hand tracking (Meta Quest, etc.)")
    print("=" * 50)
    print("Listening on http://0.0.0.0:9002 (Ctrl+C to stop)", flush=True)
    app.run(host='0.0.0.0', port=9002)
    serve(app, host='0.0.0.0', port=9002, connection_limit=5000, threads=16)
