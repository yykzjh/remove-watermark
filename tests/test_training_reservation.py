import errno
import os
import urllib.error
from contextlib import nullcontext

import pytest

from watermark_studio.training_reservation import offline_reservation


def stopped(*args,**kwargs):
    raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED,"refused"))


def test_offline_pilot_requires_stopped_api_and_free_authorized_gpu(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",stopped)
    # Actual newlines deliberately retained in the simulated CLI output.
    monkeypatch.setattr("subprocess.check_output",lambda args,**kwargs:
        "0, UUID0\n1, UUID1\n4, UUID4\n" if "--query-gpu=index,uuid" in args else f"UUID1, {os.getpid()}\nUUID4, 777\n")
    assert offline_reservation(1)["other_compute_processes"]==0
    monkeypatch.setattr("subprocess.check_output",lambda args,**kwargs:
        "1, UUID1\n" if "--query-gpu=index,uuid" in args else "UUID1, 999999\n")
    with pytest.raises(RuntimeError,match="occupied"):
        offline_reservation(1)
    with pytest.raises(ValueError,match="authorized"):
        offline_reservation(4)


def test_unreachable_or_live_api_is_not_proof_of_offline_isolation(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:nullcontext(None))
    with pytest.raises(RuntimeError,match="requires"):
        offline_reservation(0)
    def timeout(*args,**kwargs):
        raise urllib.error.URLError(TimeoutError("network timeout"))
    monkeypatch.setattr("urllib.request.urlopen",timeout)
    with pytest.raises(RuntimeError,match="Cannot prove"):
        offline_reservation(0)
