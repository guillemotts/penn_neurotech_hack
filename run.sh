#!/usr/bin/env bash
# run.sh — Launch the BCI dance game + Synapse decoder bridge
#
# Usage:
#   ./run.sh <device-ip>          # e.g. ./run.sh 192.168.0.29
#   ./run.sh                      # game only (keyboard/manual UDP)
#
# The first argument is the IP address of the SciFi device on the network.
# If omitted, only the game launches (useful for testing without hardware).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Activate virtualenv ──────────────────────────────────────────────────────
if [ -f "$ROOT/venv/Scripts/activate" ]; then
  # Windows (Git Bash / MSYS2)
  . "$ROOT/venv/Scripts/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  # macOS / Linux
  . "$ROOT/venv/bin/activate"
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  . "$ROOT/.venv/bin/activate"
fi

# ── Resolve Python ───────────────────────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3/python not found on PATH" >&2
  exit 127
fi

# ── Install dependencies if needed ───────────────────────────────────────────
if ! "$PYTHON_BIN" -c "import pygame" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  "$PYTHON_BIN" -m pip install -e . --quiet
  # Synapse client (pre-release)
  "$PYTHON_BIN" -m pip install --pre science-synapse --quiet
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DEVICE_IP="${1:-}"

# ── Start Synapse bridge (if device IP provided) ─────────────────────────────
BRIDGE_PID=""
if [ -n "$DEVICE_IP" ]; then
  echo "Starting Synapse bridge → $DEVICE_IP (tap: decoded_controls) ..."
  "$PYTHON_BIN" "$ROOT/scripts/synapse_to_udp.py" "$DEVICE_IP" decoded_controls &
  BRIDGE_PID=$!
  echo "  Bridge PID: $BRIDGE_PID"
  sleep 1   # give bridge a moment to connect before game opens
else
  echo "No device IP given — launching game in keyboard-only mode."
  echo "  To use the decoder: ./run.sh <device-ip>"
fi

# ── Cleanup bridge on exit ───────────────────────────────────────────────────
cleanup() {
  if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "Stopping bridge (PID $BRIDGE_PID)..."
    kill "$BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Launch game ──────────────────────────────────────────────────────────────
echo "Launching dance game..."
exec "$PYTHON_BIN" "$ROOT/scripts/dance_sim_game.py"
