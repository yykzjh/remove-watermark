import json
from pathlib import Path

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
    opts = ProcessingConfig(scope="full_image",quality_check=True, quality_auto_retry=True, quality_max_attempts=3)
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
    assert seen[1][1] == "full_image" and seen[1][2] != seen[0][2]
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

def test_residual_hints_survive_candidate_selection_and_merge(tmp_path):
    from pathlib import Path
    cfg = DeploymentConfig(state_dir=str(tmp_path/"state"), backend="mock")
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True)
    task = {"source": "original", "destination": str(tmp_path/"out.png"), "relative": "out.png",
            "_extra_regions": [{"kind": "logo", "box": [.7, .7, .9, .9]}]}
    def process(tasks, *_):
        Path(tasks[0]["destination"]).write_bytes(b"candidate")
        q = verdict()
        q["residual_regions"] = [{"kind": "logo", "box": [.8, .8, 1, 1]}]
        return [{"file": "out.png", "ok": True, "quality": q}]
    result = process_attempts([task], opts, cfg, None, {}, "job", process)[0]
    hints = result["_retry_task"]["_extra_regions"]
    assert len(hints) == 1 and hints[0]["box"] == (.7, .7, 1., 1.)

def test_best_candidate_penalizes_unresolved_local_change_and_flat_fill():
    unsafe = {"ok": True, "quality": verdict(removed=True)}
    unsafe["quality"].update(uncertain=True, guard_codes=["outside_repair_change"])
    safer = {"ok": True, "quality": verdict(removed=False)}
    assert quality_rank(safer) > quality_rank(unsafe)
    unsafe["quality"]["guard_codes"] = ["new_flat_fill"]
    assert quality_rank(safer) > quality_rank(unsafe)


def test_retry_prefers_residual_free_candidate_without_passing_it(tmp_path):
    cfg = DeploymentConfig(state_dir=str(tmp_path/"state"), backend="mock")
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True, quality_max_attempts=2)
    target = tmp_path/"output.png"
    task = {"source": "original", "destination": str(target), "relative": "output.png"}
    def process(tasks, *_):
        attempt = int(tasks[0].get("_attempt", 1))
        Path(tasks[0]["destination"]).write_bytes(str(attempt).encode())
        q = verdict()
        q["residual_regions"] = [{"kind": "logo", "box": [.8, .95, 1, 1]}] if attempt == 1 else []
        return [{"file": "output.png", "ok": True, "quality": q}]
    first = process_attempts([task], opts, cfg, None, {}, "job", process)[0]
    final = process_attempts([first["_retry_task"]], opts, cfg, None, {}, "job", process)[0]
    assert target.read_bytes() == b"2" and final["best_attempt"] == 2
    assert final["quality"]["status"] == "review" and final["retry_exhausted"]
    unsafe = {"ok": True, "quality": verdict(content=False)}
    unsafe["quality"]["residual_regions"] = []
    assert quality_rank(first) > quality_rank(unsafe)


def test_concrete_quality_evidence_is_recorded_and_used_on_original_retry(tmp_path):
    cfg=DeploymentConfig(state_dir=str(tmp_path/"state"),backend="mock")
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_max_attempts=2)
    target=tmp_path/"out.png"
    task={"source":"original","destination":str(target),"relative":"out.png"}
    prompts=[]
    def process(tasks,settings,*_):
        prompts.append(settings.extra_prompt)
        assert tasks[0]["source"]=="original"
        Path(tasks[0]["destination"]).write_bytes(b"candidate")
        q=verdict() if len(prompts)==1 else verdict("passed",removed=True)
        q["model_explanation"]={"status":"ok","findings":[{"source":"vlm", "issue":"rectangular_shadow",
            "box":[.1,.5,.4,.9],"observation":"矩形区域偏暗","certainty":"visible"}]}
        return [{"file":"out.png","ok":True,"quality":q}]
    first=process_attempts([task],opts,cfg,None,{},"job",process)[0]
    assert first["_retry_task"]["_retry_feedback"][0]["issue"]=="rectangular_shadow"
    final=process_attempts([first["_retry_task"]],opts,cfg,None,{},"job",process)[0]
    assert "明暗和色温台阶" in prompts[1]
    assert "0.1000,0.5000,0.4000,0.9000" not in prompts[1]  # Full-photo coordinates are projected per edit.
    assert final["attempt_history"][1]["repair_feedback_used"][0]["issue"]=="rectangular_shadow"
    assert final["attempt_history"][1]["repair_feedback_attempt"]==1
    saved=json.loads(next((tmp_path/"state").rglob("attempts.json")).read_text())
    assert saved["attempts"][0]["quality"]["model_explanation"]["findings"][0]["observation"]=="矩形区域偏暗"


def test_retry_evidence_is_isolated_per_photograph(tmp_path):
    cfg=DeploymentConfig(state_dir=str(tmp_path/"state"),backend="mock")
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_max_attempts=2)
    tasks=[]
    for name,issue in (("dark.png","rectangular_shadow"),("text.png","residual_text")):
        tasks.append({"source":"original-"+name,"relative":name,"destination":str(tmp_path/name),
                      "_retry":True,"_attempt":2,"_retry_feedback":[{"issue":issue,"box":None}]})
    seen={}
    def process(group,settings,*_):
        assert len(group)==1
        t=group[0];seen[t["relative"]]=settings.extra_prompt
        Path(t["destination"]).write_bytes(b"fixed")
        return [{"file":t["relative"],"ok":True,"quality":verdict("passed",removed=True)}]
    process_attempts(tasks,opts,cfg,None,{},"job",process)
    assert "明暗和色温台阶" in seen["dark.png"]
    assert "明暗和色温台阶" not in seen["text.png"]


def test_dispute_budget_is_once_and_supported_physical_text_is_not_blindly_repainted(tmp_path):
    cfg=DeploymentConfig(state_dir=str(tmp_path/"state"),backend="mock")
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_cross_review=True)
    task={"source":"original","destination":str(tmp_path/"out.png"),"relative":"out.png"}
    def process(tasks,*_):
        t=tasks[0];Path(t["destination"]).write_bytes(b"candidate")
        q=verdict();q["cross_review"]={"attempted":True,"status":"unresolved"}
        q["retry_recheck_only"]=True
        return [{"file":"out.png","ok":True,"quality":q}]
    first=process_attempts([task],opts,cfg,None,{},"job",process)[0]
    retry=first["_retry_task"]
    assert retry["_cross_review_used"] and retry["_protect_disputed_text"] and retry["_recheck"]


def test_inconclusive_physical_text_recheck_ends_in_review_even_in_unlimited_mode(tmp_path):
    cfg=DeploymentConfig(state_dir=str(tmp_path/"state"),backend="mock")
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_cross_review=True,
                          quality_max_attempts=0)
    task={"source":"original","destination":str(tmp_path/"out.png"),"relative":"out.png"}
    calls=[]
    def process(tasks,*_):
        t=tasks[0];calls.append(t.get("reuse_existing",False))
        if not calls[-1]:
            Path(t["destination"]).write_bytes(b"preserved physical text")
        q=verdict();q["cross_review"]={"attempted":True,"status":"unresolved"}
        q["retry_recheck_only"]=True
        return [{"file":"out.png","ok":True,"quality":q}]
    first=process_attempts([task],opts,cfg,None,{},"job",process)[0]
    final=process_attempts([first["_retry_task"]],opts,cfg,None,{},"job",process)[0]
    assert calls==[False,True] and "_retry_task" not in final
    assert final["quality"]["status"]=="review"
    assert final["retry_blocked"]=="unresolved_physical_text_dispute"
    assert final["retry_exhausted"] is False
    assert (tmp_path/"out.png").read_bytes()==b"preserved physical text"


def test_runtime_counts_current_dispute_event_not_reused_best_candidate_twice(tmp_path):
    runtime=Runtime(DeploymentConfig(backend="mock",state_dir=str(tmp_path)))
    runtime.job={"id":"cross-events","state":"running","total":1,"completed":0,"succeeded":0,
                 "failed":0,"errors":[],"no_watermark":0,"invalid":0,"options":{"quality_check":True}}
    runtime._persist()
    q=verdict();q["cross_review"]={"attempted":True,"status":"error"}
    runtime._record({"file":"brand.png","ok":True,"quality":q,
        "cross_review_attempted_now":True,"cross_review_error_now":True,
        "_retry_task":{"relative":"brand.png"}})
    runtime._record({"file":"brand.png","ok":True,"quality":q,
        "cross_review_attempted_now":False,"cross_review_error_now":False})
    assert runtime.job["cross_review_attempts"]==1 and runtime.job["cross_review_errors"]==1


def test_unsupported_residual_refusal_is_rechecked_without_repainting(tmp_path):
    cfg = DeploymentConfig(state_dir=str(tmp_path / "state"), backend="mock")
    opts = ProcessingConfig(quality_check=True, quality_auto_retry=True, quality_max_attempts=0)
    source, target = tmp_path / "original.png", tmp_path / "output.png"
    source.write_bytes(b"source")
    seen = []
    def process(tasks, settings, *_):
        task = tasks[0]
        seen.append(task.get("reuse_existing", False))
        path = Path(task["destination"])
        if task.get("_recheck"):
            assert task["reuse_existing"] and path.read_bytes() == b"safe-first"
        else:
            path.write_bytes(b"safe-first")
        q = verdict()
        q.update(residual_regions=[], model_explanation={
            "status": "ok", "assessment": "no_clear_defect", "findings": []})
        return [{"file": "a.png", "ok": True, "quality": q}]
    task = {"source": str(source), "destination": str(target), "relative": "a.png"}
    first = process_attempts([task], opts, cfg, None, {}, "job", process)[0]
    assert first["quality"]["status"] == "review"
    assert first["_retry_task"]["_recheck"]
    final = process_attempts([first["_retry_task"]], opts, cfg, None, {}, "job", process)[0]
    assert seen == [False, True]
    assert target.read_bytes() == b"safe-first" and source.read_bytes() == b"source"
    assert final["quality"]["status"] == "review"
    assert final["retry_blocked"] == "unresolved_residual_dispute"
    assert "_retry_task" not in final


@pytest.mark.parametrize("change", [
    {"natural": False}, {"content_preserved": False}, {"uncertain": True},
    {"guard_codes": ["repair_boundary_step"]}, {"residual_regions": [{"box": [0,.5,.5,1]}]},
    {"model_explanation": {"status": "error", "assessment": "no_clear_defect"}},
    {"model_explanation": {"status": "ok", "assessment": "defect_visible", "findings": []}},
    {"model_explanation": {"status": "ok", "assessment": "no_clear_defect",
                           "findings": [{"issue": "rectangular_shadow"}]}},
])
def test_real_or_uncertain_defects_never_enter_unsupported_dispute_shortcut(change):
    from watermark_studio.retry import unsupported_residual_dispute
    q = verdict() | {"residual_regions": [], "model_explanation": {
        "status": "ok", "assessment": "no_clear_defect", "findings": []}}
    assert unsupported_residual_dispute(q)
    q.update(change)
    assert not unsupported_residual_dispute(q)

@pytest.mark.parametrize("custom",[False,True])
def test_retry_preserves_whole_direct_or_explicit_custom_path(tmp_path,custom):
    from watermark_studio.config import Region
    source=tmp_path/"source.png";source.write_bytes(b"original")
    task={"source":str(source),"destination":str(tmp_path/"out.png"),"relative":"a.png",
          "_retry":True,"_attempt":2}
    options=ProcessingConfig(scope="regions" if custom else "full_image",preset="custom" if custom else "auto",
        regions=[Region(box=(.1,.1,.2,.2))] if custom else [],quality_check=True,quality_auto_retry=True,
        quality_max_attempts=2)
    def process(tasks,settings,*args):
        assert settings.scope==options.scope and settings.first_pass_mode=="whole"
        assert settings.regions==options.regions and not settings.preserve_background
        assert tasks[0]["source"]==str(source)
        Path(tasks[0]["destination"]).write_bytes(b"edited")
        return [{"file":"a.png","ok":True,"quality":verdict("passed",removed=True)}]
    result=process_attempts([task],options,DeploymentConfig(backend="mock",state_dir=str(tmp_path/"state")),None,{},"job",process)[0]
    assert result["quality"]["status"]=="passed" and source.read_bytes()==b"original"


def test_whole_image_retry_default_does_not_reduce_the_default_input_budget():
    options=ProcessingConfig(scope="full_image",quality_check=True,quality_auto_retry=True)
    assert options.resolution==options.quality_retry_resolution==1024
    assert options.whole_prompt_profile=="fidelity"


def test_protected_whole_retry_never_switches_to_region_crops(tmp_path):
    target=tmp_path/"out.png"
    options=ProcessingConfig(scope="full_image",preserve_background=True,quality_check=True,
                             quality_auto_retry=True,quality_max_attempts=2)
    task={"source":"unchanged-original","destination":str(target),"relative":"a.png","_retry":True,"_attempt":2}
    def process(tasks,settings,*args):
        assert settings.scope=="full_image" and settings.first_pass_mode=="whole"
        assert settings.preserve_background is True
        Path(tasks[0]["destination"]).write_bytes(b"result")
        return [{"file":"a.png","ok":True,"quality":verdict("passed",removed=True)}]
    result=process_attempts([task],options,DeploymentConfig(backend="mock",state_dir=str(tmp_path/"state")),
                            None,{},"job",process)[0]
    assert result["quality"]["status"]=="passed" and target.read_bytes()==b"result"
