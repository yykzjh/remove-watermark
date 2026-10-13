#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .run
if [[ -f .run/server.pid ]]; then
  pid=$(cat .run/server.pid)
  if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= | grep -q watermark_studio.server; then
    echo "Service already running: $pid"
    exit 0
  fi
fi
uv_bin=$(command -v uv || true)
uv_bin=${uv_bin:-"$HOME/.local/bin/uv"}
nohup "$uv_bin" run --frozen --no-sync python -m watermark_studio.server \
  --config "${1:-config/deployment.toml}" >.run/server.log 2>&1 </dev/null &
echo $! > .run/server.pid
echo "Service starting: $(cat .run/server.pid)"
