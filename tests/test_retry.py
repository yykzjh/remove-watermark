import json

import pytest

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.retry import process_attempts, quality_rank, retry_allowed
from watermark_studio.runtime import Runtime


def verdict(status="review", content=True, removed=False):
    return {"status": status, "content_preserved": content, "watermark_removed": removed,
            "natural": True, "uncertain": False}


def test_retry_keeps_better_candidate_and_always_uses_original(tmp_path):
    source, target = tmp_path/"original.png", tmp_path/"out.png"
    source.write_bytes(b"original")
    task = {"source": str(source), "destination": str(target), "relative": "a.png"}
    cfg = DeploymentConfig(state_dir=str(tmp_path/"state"), backend="mock")
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True, quality_max_attempts=3)
    seen = []
    def process(tasks, settings, *_):
        seen.append((tasks[0]["source"], settings.scope, settings.seed))
        attempt = tasks[0].get("_attempt", 1)
        from pathlib import Path
        Path(tasks[0]["destination"]).write_bytes(f"attempt{attempt}".encode())
        return [{"file": "a.png", "ok": True, "quality": verdict(content=attempt != 2)}]
    first = process_attempts([task], opts, cfg, None, {}, "job", process)[0]
    assert target.read_bytes() == b"attempt1"
    assert first["quality_attempts"] == 1
    second = process_attempts([first["_retry_task"]], opts, cfg, None, {}, "job", process)[0]
    assert target.read_bytes() == b"attempt1"  # Worse content must not replace it.
    assert second["best_attempt"] == 1
    third = process_attempts([second["_retry_task"]], opts, cfg, None, {}, "job", process)[0]
    assert third["retry_exhausted"] and "_retry_task" not in third
    assert len(third["attempt_history"]) == 3
    assert source.read_bytes() == b"original"
    assert all(item[0] == str(source) for item in seen)
    assert seen[1][1] == "regions" and seen[1][2] != seen[0][2]
    assert not list((tmp_path/"state").rglob("best.png"))


def test_pass_stops_retry_and_readonly_output_is_never_rewritten(tmp_path):
    cfg = DeploymentConfig(state_dir=str(tmp_path))
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True)
    target = tmp_path/"out.png"
    target.write_bytes(b"existing")
    task = {"source": "source", "destination": str(target), "relative": "out.png",
            "reuse_existing": True}
    def process(tasks, *_):
        assert tasks[0]["reuse_existing"]
        return [{"file": "out.png", "ok": True, "quality": verdict()}]
    result = process_attempts([task], opts, cfg, None, {}, "job", process)[0]
    assert result["retry_blocked"] == "existing_output_read_only"
    assert "_retry_task" not in result and target.read_bytes() == b"existing"
    assert not retry_allowed({"quality": verdict("passed")}, opts, 1)
    assert quality_rank({"ok": True, "quality": verdict("passed")}) > quality_rank(result)


def test_unlimited_retry_requires_explicit_opt_in_and_verification():
    with pytest.raises(ValueError):
        ProcessingConfig(quality_auto_retry=True)
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True, quality_max_attempts=0)
    assert retry_allowed({"quality": verdict()}, opts, 100)
    assert not retry_allowed({"quality": {"status": "skipped"}}, opts, 1)


def test_runtime_requeues_once_and_counts_only_terminal_result(tmp_path):
    runtime = Runtime(DeploymentConfig(backend="mock", state_dir=str(tmp_path)))
    runtime.job = {"id": "job", "state": "running", "total": 1, "completed": 0,
                   "succeeded": 0, "failed": 0, "errors": [], "no_watermark": 0,
                   "invalid": 0, "options": {"quality_check": True}}
    runtime._persist()
    runtime._record({"file": "a", "ok": True, "quality": verdict(),
                     "_retry_task": {"relative": "a", "_attempt": 2}})
    assert runtime.job["completed"] == 0 and len(runtime.retry_pending) == 1
    assert not (tmp_path/"job/results.jsonl").exists()
    runtime.job["state"] = "cancelling"
    runtime._record({"file": "a", "ok": True, "quality": verdict(),
                     "quality_attempts": 2, "_retry_task": {"relative": "a"}})
    assert runtime.job["completed"] == 1 and len(runtime.retry_pending) == 1
    assert runtime.job["retry_attempts"] == 1
    runtime._finish()
    assert not runtime.job["acceptance_complete"]
    assert len((tmp_path/"job/results.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path/"job/summary.json").read_text())["completed"] == 1

def test_dispatch_balances_tail_and_prioritizes_first_pass(tmp_path):
    from collections import deque

    class Inbox:
        def __init__(self):
            self.requests = []
        def put(self, item):
            self.requests.append(item)

    runtime = Runtime(DeploymentConfig(backend="mock", state_dir=str(tmp_path),
                                      batch_size=1, batch_window=4, adaptive_dispatch=True))
    runtime.job = {"id": "job", "state": "running", "options": {"concurrency": 12}}
    runtime.workers = [
        {"state": "idle", "task": None, "queue": Inbox()},
        {"state": "busy", "task": [{"relative": "busy1"}], "queue": Inbox()},
        {"state": "busy", "task": [{"relative": "busy2"}], "queue": Inbox()},
    ]
    runtime.pending = deque({"relative": str(i)} for i in range(4))
    runtime.retry_pending = deque([{"relative": "retry"}])
    runtime._dispatch()
    request = runtime.workers[0]["queue"].requests[0]
    assert [t["relative"] for t in request["tasks"]] == ["0", "1"]
    assert len(runtime.pending) == 2 and len(runtime.retry_pending) == 1


def test_cancellation_clears_pending_retries_without_dispatch(tmp_path):
    from collections import deque

    runtime = Runtime(DeploymentConfig(backend="mock", state_dir=str(tmp_path)))
    runtime.job = {"id": "job", "state": "cancelling", "options": {"quality_check": True},
                   "quality_counts": {}, "total": 3, "invalid": 0}
    runtime._persist()
    runtime.pending = deque([{"relative": "first"}])
    runtime.retry_pending = deque([{"relative": "retry1"}, {"relative": "retry2"}])
    runtime._dispatch()
    assert runtime.job["state"] == "cancelled"
    assert runtime.job["unprocessed"] == 3
    assert not runtime.pending and not runtime.retry_pending
    assert not runtime.job["acceptance_complete"]


def test_successful_retry_installs_pass_and_stops(tmp_path):
    from pathlib import Path

    cfg = DeploymentConfig(state_dir=str(tmp_path/"state"))
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True)
    target = tmp_path/"out.png"
    task = {"source": "original", "destination": str(target), "relative": "out.png"}
    def process(tasks, *_):
        attempt = tasks[0].get("_attempt", 1)
        Path(tasks[0]["destination"]).write_bytes(str(attempt).encode())
        q = verdict("passed", removed=True) if attempt == 2 else verdict()
        return [{"file": "out.png", "ok": True, "quality": q}]
    first = process_attempts([task], opts, cfg, None, {}, "job", process)[0]
    final = process_attempts([first["_retry_task"]], opts, cfg, None, {}, "job", process)[0]
    assert target.read_bytes() == b"2" and final["best_attempt"] == 2
    assert final["quality"]["status"] == "passed" and "_retry_task" not in final
    assert not final["retry_exhausted"]

def test_resume_verifies_existing_and_only_regenerates_failed_outputs(tmp_path):
    from pathlib import Path

    cfg = DeploymentConfig(state_dir=str(tmp_path/"state"))
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True, overwrite=True)
    target = tmp_path/"out.png"
    target.write_bytes(b"previous")
    task = {"source": "original", "destination": str(target), "relative": "out.png",
            "reuse_existing": True}
    calls = []
    def process(tasks, *_):
        t = tasks[0]
        calls.append(t["reuse_existing"])
        if t["reuse_existing"]:
            assert Path(t["destination"]).read_bytes() == b"previous"
            return [{"file": "out.png", "ok": True, "quality": verdict()}]
        Path(t["destination"]).write_bytes(b"repaired")
        return [{"file": "out.png", "ok": True, "quality": verdict("passed", removed=True)}]
    first = process_attempts([task], opts, cfg, None, {}, "job", process)[0]
    final = process_attempts([first["_retry_task"]], opts, cfg, None, {}, "job", process)[0]
    assert calls == [True, False]
    assert final["quality"]["status"] == "passed" and target.read_bytes() == b"repaired"
