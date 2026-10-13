import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=5) as response:
    state = json.load(response)
print(json.dumps({k: state[k] for k in ("stage", "capacity", "workers", "error")},
                 ensure_ascii=False))
