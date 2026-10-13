import argparse
import json
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8765")
args = parser.parse_args()
with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=5) as response:
    state = json.load(response)
print(json.dumps({k: state.get(k) for k in ("stage", "worker_count", "capacity", "batch_size", "batch_window", "workers", "error")},
                 ensure_ascii=False))
