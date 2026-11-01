import json
from pathlib import Path

import pytest

from watermark_studio.evaluation import exclusion_sources


def write_sources(root, name, paths):
    folder=root/name
    folder.mkdir()
    (folder/"sources.json").write_text(json.dumps(paths))
    return folder


def test_explicit_history_never_drops_original_development_samples(tmp_path):
    base=write_sources(tmp_path,"original64",["/images/old64.jpg"])
    write_sources(tmp_path,"first-pass-holdout-neutral-v1",["/images/old128.jpg"])
    write_sources(tmp_path,"first-pass-prior1024",["/images/other.jpg"])
    seen,automatic=exclusion_sources(tmp_path,base,["first-pass-prior1024"])
    assert seen=={"/images/old64.jpg","/images/old128.jpg","/images/other.jpg"}
    assert automatic==["original64","first-pass-holdout-neutral-v1","first-pass-prior1024"]


def test_first_evaluation_without_older_128_is_still_supported(tmp_path):
    base=write_sources(tmp_path,"original64",["/images/old64.jpg"])
    assert exclusion_sources(tmp_path,base,[])[0]=={"/images/old64.jpg"}
    with pytest.raises(ValueError,match="Invalid excluded"):
        exclusion_sources(tmp_path,base,["../outside"])


def test_whole_comparison_uses_independent_source_scan_instead_of_empty_compositing_mask(tmp_path):
    import runpy
    import sys
    from unittest import mock

    script=Path(__file__).resolve().parents[1]/"scripts/compare-first-pass.py"
    work=tmp_path/"work";work.mkdir()
    fake_home=tmp_path/"home"
    records=[]
    for i,profile in enumerate(["baseline","candidate"]):
        job=f"{i+1:032x}"
        state=fake_home/".local/state/watermark-studio"/job;state.mkdir(parents=True)
        row={"file":"a.png","processing_path":"whole_direct","repair_regions":[],
             "quality":{"status":"review","model_status":"passed",
                        "source_regions":[{"kind":"logo","box":[.1+i*.1,.1,.3,.3]}]}}
        (state/"results.jsonl").write_text(json.dumps(row)+"\n")
        (work/profile).mkdir();(work/profile/"a.png").write_bytes(b"same frozen output")
        records.append({"profile":profile,"job":job,"images":1,"source_selection_sha256":"frozen",
                        "source_inputs_unchanged":True,"code_revision":"same","seconds":1})
    (work/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    output=work/"comparison.json"
    argv=[str(script),"--work",str(work),"--baseline","baseline","--candidate","candidate","--output",str(output)]
    with mock.patch.object(Path,"home",return_value=fake_home),mock.patch.object(sys,"argv",argv):
        runpy.run_path(str(script),run_name="__main__")
    assert json.loads(output.read_text())["same_source_masks"]==0


def test_all_registered_exposures_are_automatically_excluded(tmp_path):
    base = write_sources(tmp_path, "original64", ["/images/old64.jpg"])
    write_sources(tmp_path, "first-pass-new1024", ["/images/heldout_then_tuned.jpg"])
    write_sources(tmp_path, "source-analysis-only", ["/images/examined.jpg"])
    (tmp_path/"current-empty-preparation").mkdir()
    seen, automatic = exclusion_sources(tmp_path, base, [])
    assert seen == {"/images/old64.jpg", "/images/heldout_then_tuned.jpg", "/images/examined.jpg"}
    assert set(automatic) == {"original64", "first-pass-new1024", "source-analysis-only"}


def test_broken_automatic_provenance_cannot_silently_enter_a_new_holdout(tmp_path):
    base = write_sources(tmp_path, "original64", ["/images/old64.jpg"])
    write_sources(tmp_path, "inspected", [{"not": "a source path"}])
    with pytest.raises(ValueError, match="source path strings"):
        exclusion_sources(tmp_path, base, [])


def test_legacy_visual_review_path_records_are_excluded_too(tmp_path):
    base = write_sources(tmp_path, "original64", ["/images/old64.jpg"])
    write_sources(tmp_path, "sample-review", [{"path": "/images/viewed.jpg", "group": "native", "size": 2}])
    assert exclusion_sources(tmp_path, base, [])[0] == {"/images/old64.jpg", "/images/viewed.jpg"}


def test_benchmark_refuses_a_stale_service_before_submitting_any_images(tmp_path, monkeypatch):
    import hashlib
    import runpy
    import sys

    work = tmp_path/"prepared"
    (work/"input").mkdir(parents=True)
    photo = work/"input"/"a.png"
    photo.write_bytes(b"immutable test image source")
    (work/"source-selection.json").write_text(json.dumps({
        "images": 1, "unique_sha256": 1, "prior_sha256_overlap": 0,
        "source_sha256": {"a.png": hashlib.sha256(photo.read_bytes()).hexdigest()}}))
    calls = []
    def request(url, endpoint, *args):
        calls.append(endpoint)
        assert endpoint == "/health"
        return {"stage": "ready", "code_revision": "old", "job": None}
    monkeypatch.setattr("watermark_studio.cli.request", request)
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "new\n")
    script = Path(__file__).resolve().parents[1]/"scripts/benchmark-protected-e2e.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--work", str(work), "--run-tag", "fresh"])
    with pytest.raises(RuntimeError, match="no job submitted"):
        runpy.run_path(str(script), run_name="__main__")
    assert calls and all(endpoint == "/health" for endpoint in calls)
