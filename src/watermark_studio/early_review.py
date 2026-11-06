"""Overlap committed-directory Codex reviews with the remaining GPU workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
import zipfile
from pathlib import Path

from .codex_photo_audit import review_photo
from .codex_photo_repair import repair_one
from .codex_pool import photo_map
from .dataset_transfer import beneath, digest, signature, write_json
from .images import atomic_copy
from .production_archives import file_sha256
from .production_assembly import checked_entries, extract_one
from .production_cloud import BaiduDelivery, publish, retrieve


def create_packet(output, rows, sources, identity, node, unit, folder):
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "index.json"
    if cached.exists():
        index = json.loads(cached.read_text())
        if index["manifest_sha256"] != identity or index["node"] != node:
            raise ValueError("Cached early packet belongs to another snapshot")
        for a in index["artifacts"]:
            if file_sha256(Path(a["local_path"])) != a["sha256"]:
                raise ValueError("Cached early-review packet changed")
        return index
    bad = [r for r in rows if r["quality_status"] != "passed"]
    if not bad:
        return None
    keep = ["file", "sha256", "size", "original_size", "quality_status", "regions"]
    records = [{k: r[k] for k in keep if k in r} for r in bad]
    metadata = {"schema": 1, "node": node, "unit": unit, "manifest_sha256": identity,
                "files": records, "sources": [sources[r["file"]] for r in bad]}
    temporary = folder / "review.zip.partial"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        info = zipfile.ZipInfo("review.json", date_time=(1980, 1, 1, 0, 0, 0))
        z.writestr(info, json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode())
        for row in records:
            p = beneath(output, row["file"])
            before = signature(p)
            h = hashlib.sha256()
            member = zipfile.ZipInfo.from_file(p, arcname="images/" + row["file"], strict_timestamps=False)
            with p.open("rb") as source, z.open(member, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(block)
                    target.write(block)
            if before != signature(p) or h.hexdigest() != row["sha256"]:
                raise ValueError("Committed early-review image differs from its evidence")
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError("Early-review packet failed CRC")
    sha = file_sha256(temporary)
    final = folder / f"{node}-review-{unit}-{sha[:16]}.zip"
    temporary.replace(final)
    index = {"schema": 1, "node": node, "manifest_sha256": identity, "files": len(bad),
             "image_files": len(bad), "auxiliary_files": 0,
             "artifacts": [{"name": final.name, "local_path": str(final), "size": final.stat().st_size,
                            "sha256": sha, "kind": "review-packet"}]}
    write_json(cached, index)
    return index


def export(c, manifest_path):
    pc = c["production"]
    state = Path(c["workflow_state"]).parent / "early-export"
    state.mkdir(parents=True, exist_ok=True)
    state.chmod(0o700)
    manifest = json.loads(Path(manifest_path).read_text())
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    sources = {r["path"]: r for r in manifest["files"]}
    done_path = state / "exported-units.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    controller = Path(pc["state_dir"])
    while True:
        for receipt in sorted(controller.glob("*-committed.json")):
            unit = receipt.name.removesuffix("-committed.json")
            if unit in done:
                continue
            rows = json.loads((controller / (unit + "-results.json")).read_text())["rows"]
            folder = state / "packets" / unit
            index = create_packet(Path(pc["output_root"]), rows, sources, identity, pc["node"], unit, folder)
            if index:
                write_json(state / "status.json", {"phase": "uploading", "unit": unit,
                                                   "images": index["image_files"]})
                delivery = publish(folder / "index.json",
                                   c["cloud_root"].rsplit("/", 1)[0] + "/early-review",
                                   Path(c["baidu_config"]), folder / "cloud")
                delivery["unit"] = unit
                with (state / "deliveries.jsonl").open("ab") as journal:
                    journal.write(json.dumps(delivery).encode() + b"\n")
            done.add(unit)
            write_json(done_path, sorted(done))
            write_json(state / "status.json", {"phase": "following_committed_directories",
                                               "directories_seen": len(done)})
        current = Path(c["workflow_state"]) / "status.json"
        phase = json.loads(current.read_text()).get("phase") if current.exists() else ""
        if phase in {"packaging", "uploading", "published"}:
            write_json(state / "status.json", {"phase": "completed", "directories_seen": len(done)})
            return
        time.sleep(15)


def receipts(node, root, cursor):
    # Bounded metadata only: never return image bytes or a whole growing journal over SSH.
    code = ("import pathlib,json; p=pathlib.Path(" + repr(root + "/early-export/deliveries.jsonl") +
            "); f=p.open('rb') if p.exists() else None; "
            "f.seek(" + str(int(cursor)) + ") if f else None; "
            "lines=[f.readline() for _ in range(2)] if f else []; "
            "pos=f.tell() if f else " + str(int(cursor)) + "; "
            "print(json.dumps({'cursor':pos-sum(len(x) for x in lines if x and not x.endswith(bytes([10]))),"
            "'deliveries':[json.loads(x) for x in lines if x.endswith(bytes([10]))]}))")
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", node,
                        "python3 -c " + shlex.quote(code)],
                       check=False, capture_output=True, text=True, timeout=40)
    if p.returncode:
        raise ConnectionError("Early-review receipt source unavailable: " + node)
    return json.loads(p.stdout)


def unpack_packet(path, destination, receipt):
    marker = destination / "verified.json"
    if marker.exists():
        meta = json.loads((destination / "review.json").read_text())
        if (meta["node"] != receipt["node"] or meta["unit"] != receipt["unit"]
                or meta["manifest_sha256"] != receipt["manifest_sha256"]):
            raise ValueError("Cached early packet belongs to another committed snapshot")
        return meta
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        entries = {name: entry for entry, name in checked_entries(z)}
        if "review.json" not in entries:
            raise ValueError("Missing early-review image inventory")
        meta = json.loads(z.read(entries["review.json"]))
        if (meta["node"] != receipt["node"] or meta["unit"] != receipt["unit"]
                or meta["manifest_sha256"] != receipt["manifest_sha256"]):
            raise ValueError("Early-review packet belongs to another committed snapshot")
        expected = {"review.json"} | {"images/" + r["file"] for r in meta["files"]}
        if set(entries) != expected:
            raise ValueError("Early-review ZIP differs from its exact image inventory")
        if len({r["file"] for r in meta["files"]}) != len(meta["files"]):
            raise ValueError("Duplicate early-review image path")
        for row in meta["files"]:
            name = "images/" + row["file"]
            extract_one(z, entries[name], beneath(destination, name), row["sha256"],
                        row["size"], destination / "partial")
    write_json(destination / "review.json", meta)
    write_json(marker, {"packet_sha256": file_sha256(path), "images": len(meta["files"])})
    return meta


def original_for(row, c, client):
    name = row["path"]
    candidates = [beneath(Path(c["original_root"]), name),
                  beneath(Path(c["state"]) / "codex/verified-originals", name)]
    for p in candidates:
        if p.is_file() and p.stat().st_size == row["size"] and digest(p) == row["md5"]:
            return p
    target = candidates[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    client.command(["download", "--saveto", str(target.parent), "-p", "8", "-l", "1",
                    "--retry", "3", "/八标照片集/" + name],
                   "original-" + hashlib.sha256(name.encode()).hexdigest())
    if (not target.is_file() or target.stat().st_size != row["size"]
            or digest(target) != row["md5"]
            or Path(str(target) + ".BaiduPCS-Go-downloading").exists()):
        raise ValueError("Early review lacks a checksum-verified original")
    return target


def local(c):
    root = Path(c["state"])
    state = root / "early-codex"
    state.mkdir(parents=True, exist_ok=True)
    state.chmod(0o700)
    control = state / "finish-current-and-handoff"
    cursors_path = state / "cursors.json"
    cursors = json.loads(cursors_path.read_text()) if cursors_path.exists() else {}
    results_path = state / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    client = BaiduDelivery(Path(c["baidu_config"]), state / "original-downloads")
    overrides = [json.loads(Path(p).read_text()) for p in c.get("human_acceptance", [])]
    human = {(r["file"], r["candidate_sha256"]) for r in overrides
             if r["decision"] == "accepted_by_user"}
    while not control.exists():
        did_work = False
        for node, remote_root in c["remote_states"].items():
            try:
                batch = receipts(node, remote_root, cursors.get(node, 0))
            except (OSError, subprocess.TimeoutExpired):
                continue
            for delivery in batch["deliveries"]:
                if control.exists():
                    break
                did_work = True
                token = delivery["index"]["sha256"]
                folder = state / "packets" / token
                indexes = retrieve([delivery], Path(c["baidu_config"]), folder / "cloud",
                                   folder / "artifacts")
                index = json.loads(Path(indexes[0]).read_text())
                for artifact in index["artifacts"]:
                    packet = folder / "artifacts" / artifact["name"]
                    meta = unpack_packet(packet, folder / "unpacked", delivery)
                    sources = {r["path"]: r for r in meta["sources"]}
                    def prepare(rows=meta["files"], source_records=sources, packet_folder=folder):
                        for row in rows:
                            if control.exists():
                                break
                            name = row["file"]
                            if name in results and results[name]["baseline_sha256"] == row["sha256"]:
                                continue
                            if (name, row["sha256"]) in human:
                                yield name, (row, None, None, True)
                                continue
                            original = original_for(source_records[name], c, client)
                            candidate = beneath(packet_folder / "unpacked/images", name)
                            if file_sha256(candidate) != row["sha256"]:
                                raise ValueError("Early candidate changed after committed-byte verification")
                            yield name, (row, original, candidate, False)

                    def inspect(payload):
                        row, original, candidate, human_accepted = payload
                        name = row["file"]
                        status = "accepted_by_user" if human_accepted else "accepted_without_edit"
                        if not human_accepted:
                            key = hashlib.sha256(name.encode()).hexdigest()
                            reviewed = review_photo(c["codex_executable"], original, candidate,
                                                    root / "codex/initially-rejected" / key,
                                                    row.get("regions", []))
                            entry = {"file": name, "group": "initially-rejected", **reviewed}
                            if not reviewed["verdict"]["acceptable"]:
                                target = beneath(state / "repair-targets", name)
                                atomic_copy(candidate, target, True, preserve_metadata=True)
                                report = repair_one(c["codex_executable"], original, target, entry,
                                                    row.get("regions", []), root / "codex-repairs" / key)
                                status = report["status"]
                        return {"file": name, "baseline_sha256": row["sha256"],
                                "status": status, "codex_reviewed": not human_accepted}

                    concurrency = c.get("codex_concurrency", 1)
                    def progress(active, workers=concurrency):
                        write_json(state / "status.json", {"phase": "reviewing_and_repairing",
                                   "concurrency": workers, "active": list(active),
                                   "reviewed": len(results), "accepted": sum(
                                       r["status"] in {"passed", "accepted_without_edit", "accepted_by_user"}
                                       for r in results.values()),
                                   "needs_review": sum(r["status"] == "needs_review" for r in results.values())})
                    for name, result in photo_map(prepare(), inspect, concurrency,
                                                  stop=control.exists, progress=progress):
                        results[name] = result
                        write_json(results_path, results)
                    if control.exists():
                        break
            if not control.exists():
                cursors[node] = batch["cursor"]
                write_json(cursors_path, cursors)
        if not did_work:
            time.sleep(10)
    write_json(state / "status.json", {"phase": "handoff_ready", "reviewed": len(results)})


def wait_for_handoff(root):
    state = Path(root) / "early-codex"
    metadata = state / "process.json"
    if not metadata.exists():
        return
    from .production import process_start
    process = json.loads(metadata.read_text())
    (state / "finish-current-and-handoff").write_text("Final assembled files are ready.")
    while True:
        try:
            alive = (Path("/proc", str(process["pid"]), "cmdline").read_bytes()
                     and process_start(process["pid"]) == process["start"])
        except (FileNotFoundError, ProcessLookupError):
            alive = False
        if not alive:
            return
        time.sleep(2)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["export", "local"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    try:
        if a.mode == "export":
            if not a.manifest:
                p.error("export requires --manifest")
            export(c, a.manifest)
        else:
            local(c)
    except Exception as error:
        state = (Path(c["workflow_state"]).parent / "early-export" if a.mode == "export"
                 else Path(c["state"]) / "early-codex")
        write_json(state / "status.json", {"phase": "error", "error": type(error).__name__,
                                           "message": str(error)[:400]})
        raise


if __name__ == "__main__":
    main()
