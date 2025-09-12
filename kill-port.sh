#!/usr/bin/env bash
set -euo pipefail

# Kill any process listening on the given ports (macOS/Linux with lsof available)
# Usage:
#   ./kill-port.sh 5432 5434 8000 5555 6379 9000 9001

graceful_kill() {
  local pid=$1
  local port=$2
  # Try SIGTERM first
  if kill -15 "$pid" >/dev/null 2>&1; then
    echo "Sent SIGTERM to PID $pid on port $port"
    sleep 0.5
  fi
  # If still alive, SIGKILL
  if ps -p "$pid" >/dev/null 2>&1; then
    kill -9 "$pid" >/dev/null 2>&1 || true
    echo "Sent SIGKILL to PID $pid on port $port"
  fi
}

ports=("$@")
if [[ ${#ports[@]} -eq 0 ]]; then
  # Defaults commonly used in this project
  ports=(5432 5434 8000 5555 6379 9000 9001)
fi

for port in "${ports[@]}"; do
  # Both macOS and Linux support lsof -ti tcp:PORT
  if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof not found. Please install lsof to use this script."
    exit 1
  fi

  # shellcheck disable=SC2046
  pids=$(lsof -ti tcp:"$port" || true)
  if [[ -n "${pids}" ]]; then
    echo "Found process(es) on port $port: ${pids}"
    for pid in ${pids}; do
      graceful_kill "$pid" "$port"
    done
  else
    echo "No process found using port $port."
  fi
done

echo "Done."
