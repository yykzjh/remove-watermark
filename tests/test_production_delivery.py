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
