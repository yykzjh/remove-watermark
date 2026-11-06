"""Observe small remote receipts, download via Baidu, assemble, and run individual audits."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shlex
import subprocess
import time
from pathlib import Path

from .codex_photo_audit import run_audit
from .dataset_transfer import write_json
from .production_assembly import assemble
from .production_cloud import retrieve


def remote_snapshot(node, root):
    # Only small state and final pinned SHA256 receipts cross the public SSH gateway.
    code = ("import pathlib,json; p=pathlib.Path(" + repr(root) + "); "
            "names=['workflow/status.json','workflow/delivery.json','controller/status.json',"
            "'full-dataset-transfer/status.json','full-dataset-transfer/report.json','pilot/status.json']; "
            "data={n:json.loads((p/n).read_text()) for n in names if (p/n).is_file()}; "
            "keys={'phase','state','complete','completed','files','total','bytes','total_bytes',"
            "'failures','verified_files','directories_including_root','controller_attempt',"
            "'error','message','gpus','directory','job','node','images','passed','review'}; "
            "print(json.dumps({n:(v if n.endswith('delivery.json') else "
            "{k:(len(x) if isinstance(x,list) and k!='gpus' else x[:400] "
            "if isinstance(x,str) else x) for k,x in v.items() if k in keys}) "
            "for n,v in data.items()},ensure_ascii=False))")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
           node, "python3 -c " + shlex.quote(code)]
    r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=40)
    if r.returncode:
        raise ConnectionError("Could not read compact remote state: " + node)
    return json.loads(r.stdout)


def run(c):
    state = Path(c["state"])
    state.mkdir(parents=True, exist_ok=True)
    state.chmod(0o700)
    deliveries = {}
    while len(deliveries) != 2:
        snapshots = {}
        with cf.ThreadPoolExecutor(2) as pool:
            futures = {pool.submit(remote_snapshot, node, root): node
                       for node, root in c["remote_states"].items()}
            for future in cf.as_completed(futures):
                node = futures[future]
                try:
                    snapshot = future.result()
                except (ConnectionError, subprocess.TimeoutExpired) as error:
                    snapshots[node] = {"state": "temporarily_unreachable", "error": type(error).__name__}
                    continue
                snapshots[node] = snapshot
                receipt = snapshot.get("workflow/delivery.json")
                if receipt and receipt["state"] == "published":
                    deliveries[node] = receipt
        write_json(state / "remote-status.json", snapshots)
        write_json(state / "status.json", {"phase": "waiting_for_remote_delivery",
                                          "published_nodes": sorted(deliveries)})
        if len(deliveries) != 2:
            time.sleep(30)
    ordered = [deliveries[node] for node in ["cdu_dev1", "cdu_dev2"]]
    write_json(state / "trusted-deliveries.json", ordered)
    write_json(state / "status.json", {"phase": "downloading_via_baidu"})
    indexes = retrieve(ordered, Path(c["baidu_config"]), state / "cloud",
                       Path(c["artifacts"]))
    write_json(state / "status.json", {"phase": "assembling_exact_tree"})
    report = assemble(indexes, Path(c["artifacts"]), Path(c["destination"]), state / "assembly")
    write_json(state / "status.json", {"phase": "individual_codex_review", "assembly": report})
    audit = run_audit(state / "assembly", Path(c["destination"]), Path(c["original_root"]),
                      state / "codex", Path(c["codex_executable"]), Path(c["baidu_config"]))
    # A reviewed dataset can still need repairs. Never call pending repairs "complete".
    write_json(state / "status.json", {"phase": "codex_audit_complete",
                                      "assembly": report, "audit": audit})
    return audit


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    try:
        result = run(c)
    except Exception as error:
        write_json(Path(c["state"]) / "status.json", {"phase": "error",
                   "error": type(error).__name__, "message": str(error)[:500]})
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
