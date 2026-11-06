import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from watermark_studio.dataset_transfer import digest
from watermark_studio.production_archives import build_archives
from watermark_studio.production_assembly import assemble, checked_entries
from watermark_studio.production_cloud import BaiduDelivery, cloud_path


def shards(tmp_path):
    original = {"schema": 1, "directories": [".", "镇", "镇/村", "空目录", "空目录/深层"],
                "files": []}
    paths = ["镇/村/图像一.jpg", "镇/村/图像二.jpg", "说明.txt", "原始空图.jpg"]
    source = tmp_path / "source"
    source.mkdir()
    (source / "镇/村").mkdir(parents=True)
    for n in paths[:2]:
        Image.new("RGB", (48, 32), (90, 120, 50)).save(source / n)
    (source / paths[2]).write_bytes(b"original auxiliary bytes")
    (source / paths[3]).write_bytes(b"")
    for name in paths:
        p = source / name
        original["files"].append({"path": name, "size": p.stat().st_size, "md5": digest(p)})
    sha = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    ownership = {"manifest_sha256": sha, "directories": original["directories"],
                 "auxiliary": original["files"][2:],
                 "units": {"one": {"node": "cdu_dev1", "files": [paths[0]]},
                           "two": {"node": "cdu_dev2", "files": [paths[1]]}}}
    archive_dir = tmp_path / "archives"
    indexes = []
    for i, name in enumerate(paths[:2], 1):
        state = tmp_path / ("state" + str(i))
        state.mkdir()
        unit = ["one", "two"][i - 1]
        row = {"file": name, "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest(),
               "size": (source / name).stat().st_size, "original_size": [48, 32],
               "quality_status": "passed" if i == 1 else "review"}
        (state / (unit + "-results.json")).write_text(json.dumps({"rows": [row]}))
        (state / (unit + "-committed.json")).write_text("{}")
        index = build_archives(source, state, ownership, original, "cdu_dev" + str(i),
                               archive_dir / str(i))
        for a in index["artifacts"]:
            Path(a["local_path"]).rename(archive_dir / a["name"])
        public = {k: index[k] for k in ["node", "manifest_sha256", "artifacts"]}
        out = tmp_path / ("index" + str(i) + ".json")
        out.write_text(json.dumps(public))
        indexes.append(out)
    return indexes, archive_dir, source


def test_roundtrip_exact_recursive_tree_and_original_size(tmp_path):
    indexes, artifacts, source = shards(tmp_path)
    output = tmp_path / "assembled"
    report = assemble(indexes, artifacts, output, tmp_path / "assembly")
    assert report["files"] == 4 and report["directories_including_root"] == 5
    assert report["valid_images"] == 2 and report["needs_codex_repair"] == 1
    assert (output / "空目录/深层").is_dir()
    for p in source.rglob("*"):
        if p.is_file():
            assert (output / p.relative_to(source)).read_bytes() == p.read_bytes()
    assert assemble(indexes, artifacts, output, tmp_path / "assembly") == report


def test_tampered_artifact_rejected_before_assembly(tmp_path):
    indexes, artifacts, _ = shards(tmp_path)
    index = json.loads(indexes[0].read_text())
    p = artifacts / index["artifacts"][0]["name"]
    with p.open("ab") as f:
        f.write(b"unexpected")
    with pytest.raises(ValueError, match="SHA256"):
        assemble(indexes, artifacts, tmp_path / "assembled", tmp_path / "assembly")


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "folder/../../escape.jpg"])
def test_zip_paths_are_confined(tmp_path, name):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name, b"bad")
    with zipfile.ZipFile(p) as z, pytest.raises(ValueError, match="Unsafe"):
        list(checked_entries(z))


def test_exit_zero_without_download_never_counts_as_success(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "pcs_config.json").write_text("{}")
    (config / "baidu.cookie").write_text("test-cookie")
    client = BaiduDelivery(config, tmp_path / "state")
    monkeypatch.setattr(client, "command", lambda *args: 0)
    monkeypatch.setattr("watermark_studio.production_cloud.time.sleep", lambda n: None)
    with pytest.raises(RuntimeError, match="SHA256"):
        client.download("/own/results", {"name": "part.zip", "size": 42, "sha256": "bad"},
                        tmp_path / "downloads")


@pytest.mark.parametrize("path", ["relative", "/a/../b"])
def test_cloud_path_confined(path):
    with pytest.raises(ValueError):
        cloud_path(path)


def final_fixture(tmp_path):
    from watermark_studio.production_final import finalize
    indexes, artifacts, _ = shards(tmp_path)
    output, assembly, audit, repairs = [tmp_path / n for n in ["assembled", "assembly", "audit", "repairs"]]
    assemble(indexes, artifacts, output, assembly)
    audit.mkdir()
    repairs.mkdir()
    rows = [json.loads(line) for line in (assembly / "assembled-images.jsonl").read_text().splitlines()]
    plan = {"sample": [rows[0]["file"]], "initially_rejected": [rows[1]["file"]]}
    (audit / "audit-plan.json").write_text(json.dumps(plan))
    verdict = {"acceptable": True, "watermark_removed": True, "natural_fill": True,
               "content_preserved": True, "uncertain": False, "defects": [],
               "evidence": ["No digital overlay remains; reconstructed regions are plausible."]}
    (audit / "audit-results.jsonl").write_text(
        "".join(json.dumps({"file": r["file"], "verdict": verdict}) + "\n" for r in rows))
    (repairs / "repair-results.jsonl").write_text("")
    return finalize, assembly, audit, repairs, output


def test_final_inventory_keeps_initial_sample_grades_and_exact_original_tree(tmp_path):
    finalize, assembly, audit, repairs, output = final_fixture(tmp_path)
    report = finalize(assembly, audit, repairs, output, tmp_path / "final")
    assert report["files"] == 4 and report["directories_including_root"] == 5
    assert report["valid_images"] == 2 and report["initial_sample_passed"] == 1
    assert report["codex_repair_dispositions"] == 0
    assert report["source_empty_images"] == ["原始空图.jpg"]


def test_final_inventory_rejects_later_byte_changes(tmp_path):
    finalize, assembly, audit, repairs, output = final_fixture(tmp_path)
    (output / "镇/村/图像一.jpg").write_bytes(b"corrupted after initial assembly")
    with pytest.raises(ValueError, match="differs"):
        finalize(assembly, audit, repairs, output, tmp_path / "final")


def test_final_inventory_requires_every_individual_review(tmp_path):
    finalize, assembly, audit, repairs, output = final_fixture(tmp_path)
    p = audit / "audit-results.jsonl"
    p.write_text(p.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="every required"):
        finalize(assembly, audit, repairs, output, tmp_path / "final")


def test_scope_deferred_server_failures_are_preserved_and_never_labelled_verified(tmp_path):
    finalize, assembly, audit, repairs, output = final_fixture(tmp_path)
    path = audit / "audit-plan.json"
    plan = json.loads(path.read_text())
    deferred = plan["initially_rejected"]
    plan.update(initially_rejected=[], deferred_server_review=deferred, minimum_server_attempts=3)
    path.write_text(json.dumps(plan))
    reviews = [json.loads(line) for line in (audit / "audit-results.jsonl").read_text().splitlines()]
    (audit / "audit-results.jsonl").write_text("".join(json.dumps(r) + chr(10) for r in reviews
                                                    if r["file"] not in deferred))
    report = finalize(assembly, audit, repairs, output, tmp_path / "final")
    assert report["state"] == "completed_with_review"
    assert report["remaining_valid_images_needing_review"] == 1
    assert (tmp_path / "final/needs-review-relative-paths.txt").read_text().strip() == deferred[0]
    inventory = json.loads((tmp_path / "final/final-output-manifest.json").read_text())
    row = next(r for r in inventory["files"] if r["file"] == deferred[0])
    assert row["status"] == "server_needs_review_outside_local_scope"


@pytest.mark.parametrize("minimum,only_failures", [(3, False), (0, True)])
def test_passed_server_sample_can_be_inspected_but_not_locally_edited_under_strict_scope(
        tmp_path, monkeypatch, minimum, only_failures):
    from watermark_studio.codex_photo_repair import run_repairs
    from watermark_studio.production_archives import file_sha256
    _, assembly, audit, repairs, output = final_fixture(tmp_path)
    file = json.loads((audit / "audit-plan.json").read_text())["sample"][0]
    before = file_sha256(output / file)
    entry = {"file": file, "group": "sample-1000", "server_final_status": "passed", "server_attempts": 1}
    (audit / "codex-repair-queue.jsonl").write_text(json.dumps(entry) + chr(10))
    monkeypatch.setattr("watermark_studio.codex_photo_repair.repair_one",
                        lambda *a: pytest.fail("Strict scope prohibits local edits to server-passed samples"))
    report = run_repairs(assembly, audit, output, tmp_path / "originals", repairs, "unused",
                         concurrency=4, minimum_server_attempts=minimum, server_failures_only=only_failures)
    assert report["needs_review"] == 1
    assert file_sha256(output / file) == before


def test_completed_before_scope_change_is_reused_without_any_new_codex_call(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from watermark_studio.codex_photo_audit import PROMPT, run_audit
    from watermark_studio.production_archives import file_sha256
    indexes, artifacts, source = shards(tmp_path)
    output, assembly, audit = [tmp_path / n for n in ["output", "assembly", "audit"]]
    assemble(indexes, artifacts, output, assembly)
    name = "镇/村/图像二.jpg"
    folder = audit / "initially-rejected" / hashlib.sha256(name.encode()).hexdigest()
    folder.mkdir(parents=True)
    sha = file_sha256(output / name)
    verdict = {"acceptable": True, "watermark_removed": True, "natural_fill": True,
               "content_preserved": True, "uncertain": False, "defects": [],
               "evidence": ["Previously completed independent comparison accepted these exact bytes."]}
    cached = {"identity": {"original_md5": digest(source / name), "output_sha256": sha,
                          "model": "gpt-6.1-sol", "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()},
              "verdict": verdict}
    path = folder / "review.json"
    path.write_text(json.dumps(cached))
    before = path.read_bytes()
    monkeypatch.setattr("watermark_studio.codex_photo_audit.BaiduDelivery",
                        lambda *a: SimpleNamespace(command=lambda *a: pytest.fail("Original is already verified")))
    monkeypatch.setattr("watermark_studio.codex_photo_audit.subprocess.run",
                        lambda *a, **k: pytest.fail("Preserving completed work cannot create another Codex session"))
    report = run_audit(assembly, output, source, audit, "must-not-run", tmp_path / "baidu",
                       sample_size=0, concurrency=4, minimum_server_attempts=3,
                       completed_before_scope_change=[{"file": name, "baseline_sha256": sha, "final_sha256": sha}])
    assert report["individually_reviewed"] == 1
    assert report["deferred_server_reviews"] == 0
    assert path.read_bytes() == before
    plan = json.loads((audit / "audit-plan.json").read_text())
    assert plan["completed_before_scope_change"] == [name]


@pytest.mark.parametrize("reuse", [False, True])
def test_only_explicit_pre_scope_cache_can_be_merged_without_any_new_edit(tmp_path, monkeypatch, reuse):
    from watermark_studio.codex_photo_repair import run_repairs
    from watermark_studio.production_archives import file_sha256
    _, assembly, audit, repairs, output = final_fixture(tmp_path)
    name = json.loads((audit / "audit-plan.json").read_text())["initially_rejected"][0]
    baseline = file_sha256(output / name)
    case = repairs / hashlib.sha256(name.encode()).hexdigest()
    case.mkdir()
    selected = case / "selected.jpg"
    Image.new("RGB", (48, 32), "gray").save(selected)
    final_sha = file_sha256(selected)
    good = {"acceptable": True, "watermark_removed": True, "natural_fill": True,
            "content_preserved": True, "uncertain": False, "defects": [],
            "evidence": ["Prior independent comparison accepted this exact candidate."]}
    report = {"file": name, "status": "passed", "initial_sha256": baseline,
              "final_sha256": final_sha, "original_md5": digest(tmp_path / "source" / name),
              "selected_file": str(selected), "final_verdict": good, "history": []}
    (case / "report.json").write_text(json.dumps(report))
    bad = dict(good, acceptable=False, natural_fill=False)
    entry = {"file": name, "server_attempts": 1, "server_final_status": "review",
             "identity": {"output_sha256": baseline}, "verdict": bad,
             "reuse_completed_before_scope_change": reuse, "prior_completed_final_sha256": final_sha}
    (audit / "codex-repair-queue.jsonl").write_text(json.dumps(entry) + chr(10))
    monkeypatch.setattr("watermark_studio.codex_photo_repair.generate",
                        lambda *a: pytest.fail("Merging a prior cache cannot call the image model"))
    result = run_repairs(assembly, audit, output, tmp_path / "source", repairs, "must-not-run",
                         concurrency=4, minimum_server_attempts=3, server_failures_only=True)
    assert result["accepted"] == int(reuse)
    assert file_sha256(output / name) == (final_sha if reuse else baseline)
