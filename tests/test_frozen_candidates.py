import json
from pathlib import Path

import pytest

from watermark_studio.frozen_candidates import digest, first_snapshot


def fixture(tmp_path):
    source, output, state = (tmp_path / p for p in ["input", "output", "state"])
    for p in (source, output, state / "job"):
        p.mkdir(parents=True)
    folder = state / "job"
    for name in ["town/village/a.jpg", "town/village/b.jpg"]:
        for root in (source, output):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(("source:" if root == source else "final:").encode() +
                                     name.encode())
    clean = {"status": "passed", "source_regions": []}
    rejected = {"status": "review", "guard_codes": ["repair_boundary_step"]}
    masks = [{"kind": "panel", "box": [0, .7, .4, 1]}]
    initial = {"file": "town/village/b.jpg", "ok": True, "quality": rejected,
               "quality_attempts": 1, "repair_regions": masks,
               "protected_core_boxes": [{"mask": [0, .7, .4, 1], "core": [.01, .72, .38, .99]}],
               "background_preserved": True, "source_scene": "outdoor",
               "harmonization": {"correction_withheld": False}}
    first_file = folder / "retries/retained.jpg"
    first_file.parent.mkdir()
    first_file.write_bytes(b"original rejected candidate")
    final = [
        {"file": "town/village/a.jpg", "ok": True, "quality": clean,
         "quality_attempts": 1, "source_scene": None},
        initial | {"quality": clean, "quality_attempts": 2, "repair_regions": [],
                   "protected_core_boxes": [], "harmonization": {"correction_withheld": True},
                   "attempt_history": [{"quality": rejected}, {"quality": clean}],
                   "first_rejected_candidate": str(first_file),
                   "first_rejected_sha256": digest(first_file)},
    ]
    (folder / "summary.json").write_text(json.dumps(
        {"state": "completed", "total": 2, "input_dir": str(source), "output_dir": str(output)}))
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in final))
    (folder / "retry-events.jsonl").write_text(json.dumps(initial) + "\n")
    return state, source, output, first_file, final


def test_first_snapshot_never_substitutes_a_later_success_for_first_generation(tmp_path):
    state, source, output, retained, _ = fixture(tmp_path)
    original = {p: p.read_bytes() for root in (source, output) for p in root.rglob("*.jpg")}
    result = first_snapshot(state, "job", tmp_path / "snapshots", "first")
    target = Path(result["output_dir"])
    assert (target / "town/village/a.jpg").read_bytes() == (output / "town/village/a.jpg").read_bytes()
    assert (target / "town/village/b.jpg").read_bytes() == retained.read_bytes()
    rows = [json.loads(s) for s in (target.parent / "results.jsonl").read_text().splitlines()]
    assert rows[1]["quality"]["status"] == "review"
    assert rows[1]["repair_regions"] and rows[1]["protected_core_boxes"]
    assert rows[1]["harmonization"] == {"correction_withheld": False}
    assert all(p.read_bytes() == content for p, content in original.items())


def test_changed_retained_candidate_cannot_enter_a_snapshot(tmp_path):
    state, _, _, retained, _ = fixture(tmp_path)
    retained.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA changed"):
        first_snapshot(state, "job", tmp_path / "snapshots", "first")
    assert not (tmp_path / "snapshots/first/summary.json").exists()


def test_missing_first_event_cannot_borrow_final_mask_metadata(tmp_path):
    state, _, _, _, _ = fixture(tmp_path)
    (state / "job/retry-events.jsonl").write_text("")
    with pytest.raises(ValueError, match="Missing full first"):
        first_snapshot(state, "job", tmp_path / "snapshots", "first")


@pytest.mark.parametrize("name", ["../escape.jpg", "/escape.jpg"])
def test_snapshot_rejects_filename_escape(tmp_path, name):
    state, _, _, _, rows = fixture(tmp_path)
    rows[0]["file"] = name
    (state / "job/results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        first_snapshot(state, "job", tmp_path / "snapshots", "first")


def test_snapshot_refuses_to_overwrite_a_prior_comparison(tmp_path):
    state, _, _, _, _ = fixture(tmp_path)
    first_snapshot(state, "job", tmp_path / "snapshots", "first")
    with pytest.raises(FileExistsError):
        first_snapshot(state, "job", tmp_path / "snapshots", "first")


def test_verifier_factory_retains_first_vetoes_and_handles_unknown_source_scene(tmp_path):
    import importlib.util
    state, _, _, _, _ = fixture(tmp_path)
    first_snapshot(state, "job", tmp_path / "snapshots", "first")
    script=Path(__file__).resolve().parents[1]/"scripts/benchmark-verifier-known-source.py"
    spec=importlib.util.spec_from_file_location("frozen_benchmark",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=module.frozen_job_cases(tmp_path / "snapshots","first")
    assert cases[0]["scene"]=="unknown"
    assert cases[1]["frozen_vetoes"]==["repair_boundary_step"]
    assert cases[1]["source_cores"] and cases[1]["preserved"]


def identity():
    return {"relation": "different_weights", "editor_sha256": "a"*64, "verifier_sha256": "b"*64}


def validated_fixture(tmp_path):
    state, source, output, retained, rows = fixture(tmp_path)
    path=state/"job/summary.json";s=json.loads(path.read_text())
    s["quality_weight_identity"]=identity();path.write_text(json.dumps(s))
    first_snapshot(state,"job",tmp_path/"snapshots","first")
    return tmp_path/"snapshots/first",source,output,retained,rows


def test_cached_verdict_requires_unchanged_weights_pixels_and_metadata(tmp_path):
    from watermark_studio.frozen_candidates import validate_first_snapshot
    root,*_=validated_fixture(tmp_path)
    assert validate_first_snapshot(root,identity())["snapshot_schema"]==2
    other=identity()|{"verifier_sha256":"c"*64}
    with pytest.raises(ValueError,match="canonical primary"):
        validate_first_snapshot(root,other)


@pytest.mark.parametrize("changed",["source","output","verdict"])
def test_cached_adjudication_cannot_silently_accept_modified_inputs(tmp_path,changed):
    from watermark_studio.frozen_candidates import validate_first_snapshot
    root,source,*_=validated_fixture(tmp_path)
    if changed in {"source","output"}:
        base=source if changed=="source" else root/"outputs"
        (base/"town/village/a.jpg").write_bytes(b"changed pixels")
    else:
        p=root/"results.jsonl";rows=[json.loads(l) for l in p.read_text().splitlines()]
        rows[1]["quality"]={"status":"passed"}
        p.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError,match="changed"):
        validate_first_snapshot(root,identity())
