#!/usr/bin/env bash
# Pass every server option through unchanged. No image data is sent by this script.
set -euo pipefail
cd "$(dirname "$0")/.."

# Preserve the old invocation: bash scripts/start-server.sh config/deployment.toml
if [[ $# -gt 0 && "$1" != -* ]]; then
  config_path=$1
  shift
  set -- --config "$config_path" "$@"
fi

uv_bin=$(command -v uv || true)
uv_bin=${uv_bin:-"$HOME/.local/bin/uv"}
python_bin=$("$uv_bin" run --no-sync python -c 'import sys; print(sys.executable)')
for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" || "$arg" == "--print-config" ]]; then
    exec "$python_bin" -m watermark_studio.server "$@"
  fi
done

mkdir -p .run
exec 9>.run/start.lock
flock -n 9 || { echo "Another startup is in progress" >&2; exit 1; }
if [[ -f .run/server.pid ]]; then
  pid=$(cat .run/server.pid)
  if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= | grep -q watermark_studio.server; then
    if [[ $# -gt 0 ]]; then
      echo "Service $pid is running. New parameters were NOT applied." >&2
      echo "Finish active jobs, then run scripts/stop-server.sh before starting with new options." >&2
      exit 1
    fi
    echo "Service already running: $pid"
    exit 0
  fi
fi

# Validate before creating the daemon; save the exact effective settings for auditing.
config_tmp=$(mktemp .run/deployment.XXXXXX)
trap 'rm -f -- "$config_tmp"' EXIT
"$python_bin" -m watermark_studio.server "$@" --print-config >"$config_tmp"
mv -- "$config_tmp" .run/deployment.json
nohup "$python_bin" -m watermark_studio.server "$@" >.run/server.log 2>&1 </dev/null 9>&- &
pid=$!
echo "$pid" >.run/server.pid
sleep 0.5
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Service exited during startup; inspect .run/server.log" >&2
  exit 1
fi
echo "Service starting: $pid"
echo "Effective configuration: .run/deployment.json"
echo "Log: .run/server.log. Check /health until stage=ready."
