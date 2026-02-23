#!/usr/bin/env bash
# Stop all teleop services started by start_teleop.sh.

set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
PIDFILE="$ROOT/.teleop_pids"

if [[ ! -f "$PIDFILE" ]]; then
  echo "No PID file found. Nothing to stop."
  exit 0
fi

echo "Stopping teleop services..."
while read -r pid; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
done < "$PIDFILE"

rm -f "$PIDFILE"
echo "All teleop services stopped."
