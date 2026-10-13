#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .run/server.pid ]]; then
  pid=$(cat .run/server.pid)
  if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= | grep -q watermark_studio.server; then
    kill -TERM "$pid"
    echo "Stop requested"
  fi
fi
