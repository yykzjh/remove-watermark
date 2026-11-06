"""Small, bounded status proofs keep local fallback out of unfinished server work."""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from .dataset_transfer import write_json


def allowed_fallback(row, proof, minimum_attempts=0):
    if type(minimum_attempts) is not int or minimum_attempts < 0:
        raise ValueError("Minimum server attempts must be a nonnegative integer")
    if proof["sha256"] != row["sha256"]:
        raise ValueError("Server final result differs from the pinned fallback candidate")
    return proof["status"] != "passed" and proof["attempts"] >= minimum_attempts


def server_dispositions(node, root, unit, keys):
    if node not in {"cdu_dev1", "cdu_dev2"} or len(keys) > 2:
        raise ValueError("Fallback status requests are limited to two records on servers 1 and 2")
    if len(unit) != 64 or any(c not in "0123456789abcdef" for c in unit):
        raise ValueError("Invalid committed unit identity")
    if any(len(k) != 64 or any(c not in "0123456789abcdef" for c in k) for k in keys):
        raise ValueError("Invalid photo identity")
    code = ("from pathlib import Path; import json,hashlib; p=Path(" + repr(root) + ")/'controller'; "
            "u=" + repr(unit) + "; assert (p/(u+'-committed.json')).is_file(); "
            "rows=json.loads((p/(u+'-results.json')).read_text())['rows']; keys=" + repr(keys) + "; "
            "d={hashlib.sha256(r['file'].encode()).hexdigest():r for r in rows}; "
            "print(json.dumps({k:{'attempts':d[k].get('quality_attempts',0),"
            "'status':d[k]['quality_status'],'sha256':d[k]['sha256']} for k in keys}))")
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                             "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", node,
                             "python3 -c " + shlex.quote(code)],
                            check=False, capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise ConnectionError("Could not verify committed fallback status on " + node)
    proof = json.loads(result.stdout)
    if set(proof) != set(keys):
        raise ValueError("Fallback status response omitted requested photos")
    return proof


def packet_dispositions(meta, remote_states, folder):
    identity = {k: meta[k] for k in ["node", "unit", "manifest_sha256"]}
    path = Path(folder) / "server-final-dispositions.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Fallback status cache belongs to another committed unit")
        return saved["photos"]
    keys = [hashlib.sha256(r["file"].encode()).hexdigest() for r in meta["files"]]
    proof = {}
    for offset in range(0, len(keys), 2):
        proof.update(server_dispositions(meta["node"], remote_states[meta["node"]],
                                         meta["unit"], keys[offset:offset+2]))
    write_json(path, {"identity": identity, "photos": proof})
    return proof
