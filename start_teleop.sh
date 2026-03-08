#!/usr/bin/env bash
# Start teleop services: hand tracker, MuJoCo visualizer, and localtunnel.
# Run from robot_teleop folder.

set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
PIDFILE="$ROOT/.teleop_pids"

# Clean up any stale PID file
rm -f "$PIDFILE"

# Choose Python for run_visualize_tesollo: mjpython on macOS, else python3
if [[ "$(uname)" == "Darwin" ]]; then
  if command -v mjpython &>/dev/null; then
    PYTHON_CMD="mjpython"
  elif [[ -f "$ROOT/venv/bin/mjpython" ]]; then
    PYTHON_CMD="$ROOT/venv/bin/mjpython"
  else
    echo "On macOS, mjpython is required. Install with: pip install mujoco"
    exit 1
  fi
else
  PYTHON_CMD="python3"
fi

echo "Starting hand tracker..."
python3 "$ROOT/teleop/start_hand_tracker.py" &
echo $! >> "$PIDFILE"

echo "Starting MuJoCo visualizer (using $PYTHON_CMD)..."
"$PYTHON_CMD" "$ROOT/run_visualize_tesollo.py" "$@" &
echo $! >> "$PIDFILE"

echo "Starting localtunnel (lt --port 9002 --subdomain teleop)..."
lt --port 9002 --subdomain teleop &
echo $! >> "$PIDFILE"

echo "All services started. PIDs saved to $PIDFILE"
echo "Stop with: ./stop_teleop.sh"
