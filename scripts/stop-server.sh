#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .run/server.pid ]]; then
  pid=$(cat .run/server.pid)
  if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= | grep -q watermark_studio.server; then
    kill -TERM "$pid"
    for _ in {1..150}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "Service stopped"
        exit 0
      fi
      sleep 0.2
    done
    echo "Service has not stopped yet; inspect .run/server.log" >&2
    exit 1
  fi
fi
echo "No running service"
