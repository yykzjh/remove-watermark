"""Read-only GPU isolation checks for explicitly offline, single-GPU pilots."""
from __future__ import annotations

import errno
import os
import subprocess
import urllib.error
import urllib.request


def offline_reservation(gpu):
    if type(gpu) is not int or gpu not in (0,1,2,3):
        raise ValueError("Offline pilot is restricted to authorized GPU0-3")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2):
            raise RuntimeError("Offline training requires the inference API to be stopped")
    except urllib.error.URLError as error:
        if not isinstance(error.reason,OSError) or error.reason.errno!=errno.ECONNREFUSED:
            raise RuntimeError("Cannot prove the inference API is stopped") from error
    devices=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True)
    mapping={int(line.split(",")[0].strip()):line.split(",")[1].strip() for line in devices.splitlines() if line.strip()}
    if gpu not in mapping:
        raise ValueError("Requested physical GPU not found")
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True)
    for line in apps.splitlines():
        if not line.strip():continue
        uuid,pid=[s.strip() for s in line.split(",")]
        if uuid==mapping[gpu] and int(pid)!=os.getpid():
            raise RuntimeError("Offline training GPU is occupied by another process")
    return {"active":False,"gpu":gpu,"api":"stopped","other_compute_processes":0}
