"""Audit checkpoint identity locally; prints only small metadata, never weight data."""
import argparse
import json
import time

from watermark_studio.model_identity import compare_model_weights

parser = argparse.ArgumentParser()
parser.add_argument("--editor", required=True)
parser.add_argument("--verifier", required=True)
args = parser.parse_args()
started = time.monotonic()
result = compare_model_weights(args.editor, args.verifier)
print(json.dumps(result | {"seconds": round(time.monotonic()-started, 3)}), flush=True)
if result["relation"] == "unknown":
    raise SystemExit(1)
