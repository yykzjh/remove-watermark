#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .run
exec 9>.run/start.lock
flock -n 9 || { echo "Another startup is in progress"; exit 0; }
if [[ -f .run/server.pid ]]; then
  pid=$(cat .run/server.pid)
  if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= | grep -q watermark_studio.server; then
    echo "Service already running: $pid"
    exit 0
  fi
fi
uv_bin=$(command -v uv || true)
uv_bin=${uv_bin:-"$HOME/.local/bin/uv"}
python_bin=$("$uv_bin" run --no-sync python -c 'import sys; print(sys.executable)')
nohup "$python_bin" -m watermark_studio.server \
  --config "${1:-config/deployment.toml}" >.run/server.log 2>&1 </dev/null 9>&- &
echo $! > .run/server.pid
echo "Service starting: $(cat .run/server.pid)"
