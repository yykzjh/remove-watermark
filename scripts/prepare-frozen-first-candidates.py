"""Prepare first-generation snapshots, entirely on the development machine."""
import argparse
import json
from pathlib import Path

from watermark_studio.frozen_candidates import first_snapshot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", required=True)
parser.add_argument("--state-dir", type=Path, default=Path.home()/".local/state/watermark-studio")
parser.add_argument("--target-root", type=Path, required=True)
parser.add_argument("--tag", required=True)
args = parser.parse_args()
print(json.dumps(first_snapshot(args.state_dir, args.job, args.target_root, args.tag),
                 ensure_ascii=False))
