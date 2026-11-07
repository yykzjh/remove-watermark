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

from .codex_fallback import allowed_fallback, packet_dispositions
from .codex_photo_audit import review_photo
from .codex_photo_repair import repair_one
from .codex_pool import photo_map
from .codex_sampling import interleave, make_sampling_plan, validate_sampling_plan
from .dataset_transfer import beneath, digest, signature, write_json
from .images import atomic_copy
from .production_archives import file_sha256
from .production_assembly import checked_entries, extract_one
from .production_cloud import BaiduDelivery, publish, retrieve
from .production_queue import IMAGE_SUFFIXES

GROUP_FOLDERS = {"initially-rejected": ("early-export", "early-review"),
                 "sample-1000": ("passed-export", "passed-review")}


def create_packet(output, rows, sources, identity, node, unit, folder,
                  group="initially-rejected", sampling_plan=None):
    if group not in GROUP_FOLDERS:
        raise ValueError("Unknown early review group")
    if group == "sample-1000":
        validate_sampling_plan(sampling_plan or {})
    plan_sha = (hashlib.sha256(json.dumps(sampling_plan, sort_keys=True).encode()).hexdigest()
                if sampling_plan is not None else None)
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "index.json"
    if cached.exists():
        index = json.loads(cached.read_text())
        if (index["manifest_sha256"] != identity or index["node"] != node
                or index.get("audit_group", "initially-rejected") != group
                or index.get("sampling_plan_sha256") != plan_sha):
            raise ValueError("Cached early packet belongs to another snapshot")
        for a in index["artifacts"]:
            if file_sha256(Path(a["local_path"])) != a["sha256"]:
                raise ValueError("Cached early-review packet changed")
        return index
    if group == "sample-1000":
        primary = set(sampling_plan["primary"])
        bad = [r for r in rows if r["quality_status"] == "passed" and r["file"] in primary]
    else:
        bad = [r for r in rows if r["quality_status"] != "passed"]
    if not bad:
        return None
    keep = ["file", "sha256", "size", "original_size", "quality_status", "regions"]
    records = [{k: r[k] for k in keep if k in r} for r in bad]
    metadata = {"schema": 1, "node": node, "unit": unit, "manifest_sha256": identity,
                "files": records, "sources": [sources[r["file"]] for r in bad],
                "audit_group": group}
    if sampling_plan is not None:
        metadata["sampling_plan"] = sampling_plan
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
             "image_files": len(bad), "auxiliary_files": 0, "audit_group": group,
             "sampling_plan_sha256": plan_sha,
             "artifacts": [{"name": final.name, "local_path": str(final), "size": final.stat().st_size,
                            "sha256": sha, "kind": "review-packet"}]}
    write_json(cached, index)
    return index


def export(c, manifest_path, group="initially-rejected", sample_size=1000, seed=20261106, population=None):
    pc = c["production"]
    export_folder, cloud_folder = GROUP_FOLDERS[group]
    state = Path(c["workflow_state"]).parent / export_folder
    state.mkdir(parents=True, exist_ok=True)
    state.chmod(0o700)
    manifest = json.loads(Path(manifest_path).read_text())
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    sources = {r["path"]: r for r in manifest["files"]}
    sampling_plan = None
    if group == "sample-1000":
        frame = [r["path"] for r in manifest["files"] if r["size"] > 0
                 and Path(r["path"]).suffix.lower() in IMAGE_SUFFIXES]
        if population is not None and len(frame) != population:
            raise ValueError("Verified input image count differs from sampling population")
        sampling_plan = make_sampling_plan(frame, sample_size, seed)
        frozen = state / "sampling-plan.json"
        if frozen.exists() and json.loads(frozen.read_text()) != sampling_plan:
            raise ValueError("Streaming export candidates were already frozen")
        write_json(frozen, sampling_plan)
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
            index = create_packet(Path(pc["output_root"]), rows, sources, identity, pc["node"], unit, folder,
                                  group, sampling_plan)
            if index:
                write_json(state / "status.json", {"phase": "uploading", "unit": unit,
                                                   "images": index["image_files"]})
                delivery = publish(folder / "index.json",
                                   c["cloud_root"].rsplit("/", 1)[0] + "/" + cloud_folder,
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


def receipts(node, root, cursor, group="initially-rejected"):
    # Bounded metadata only: never return image bytes or a whole growing journal over SSH.
    folder = GROUP_FOLDERS[group][0]
    code = ("import pathlib,json; p=pathlib.Path(" + repr(root + "/" + folder + "/deliveries.jsonl") +
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



def receipt_batch(node, root, cursor, group="initially-rejected", maximum=2):
    """Prefetch a bounded queue through ordinary two-receipt SSH commands."""
    if type(maximum) is not int or not 2 <= maximum <= 32:
        raise ValueError("Receipt prefetch must be from 2 to 32")
    found = []
    position = cursor
    for _ in range((maximum + 1) // 2):
        try:
            batch = (receipts(node, root, position) if group == "initially-rejected"
                     else receipts(node, root, position, group))
        except (OSError, subprocess.TimeoutExpired):
            if not found:
                raise
            break
        found.extend(batch["deliveries"])
        position = batch["cursor"]
        if len(batch["deliveries"]) < 2:
            break
    return {"deliveries": found, "cursor": position}

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


def prepared_deliveries(c, state, batches, results, client, control, human,
                        group="initially-rejected"):
    """One serialized producer keeps all photo slots occupied across directory packets."""
    for batch in batches.values():
        for delivery in batch["deliveries"]:
            if control.exists():
                return
            token = delivery["index"]["sha256"]
            folder = state / "packets" / token
            indexes = retrieve([delivery], Path(c["baidu_config"]), folder / "cloud",
                               folder / "artifacts")
            index = json.loads(Path(indexes[0]).read_text())
            for artifact in index["artifacts"]:
                packet = folder / "artifacts" / artifact["name"]
                meta = unpack_packet(packet, folder / "unpacked", delivery)
                if meta.get("audit_group", "initially-rejected") != group:
                    raise ValueError("Early packet review group changed")
                if group == "sample-1000":
                    plan = meta.get("sampling_plan") or {}
                    validate_sampling_plan(plan)
                    if (plan["sample_size"] != c.get("codex_sample_size", 1000)
                            or plan["seed"] != c.get("codex_sample_seed", 20261106)
                            or plan["population"] != c["codex_sample_population"]):
                        raise ValueError("Streaming sample differs from the authorized frame")
                    frozen = state / "stream-sampling-plan.json"
                    if frozen.exists() and json.loads(frozen.read_text()) != plan:
                        raise ValueError("Streaming primary candidates changed")
                    write_json(frozen, plan)
                sources = {r["path"]: r for r in meta["sources"]}
                proofs = packet_dispositions(meta, c["remote_states"], folder)
                for row in meta["files"]:
                    if control.exists():
                        return
                    name = row["file"]
                    proof = proofs[hashlib.sha256(name.encode()).hexdigest()]
                    if group == "sample-1000":
                        if (proof["sha256"] != row["sha256"] or proof["status"] != "passed"
                                or row.get("quality_status") != "passed" or name not in plan["primary"]):
                            raise ValueError("Streaming audit requires a pinned committed passed candidate")
                        eligible = True
                    else:
                        eligible = allowed_fallback(row, proof, c.get("codex_min_server_attempts", 0))
                    if name in results:
                        results[name].update(server_attempts=proof["attempts"],
                                             server_final_status=proof["status"],
                                             eligible_under_current_scope=eligible)
                        write_json(state / "results.json", results)
                    if name in results and results[name]["baseline_sha256"] == row["sha256"]:
                        continue
                    if not eligible:
                        continue
                    if group == "initially-rejected" and (name, row["sha256"]) in human:
                        yield name, (row, None, None, True, proof)
                        continue
                    original = original_for(sources[name], c, client)
                    candidate = beneath(folder / "unpacked/images", name)
                    if file_sha256(candidate) != row["sha256"]:
                        raise ValueError("Early candidate changed after committed-byte verification")
                    yield name, (dict(row, _audit_group=group), original, candidate, False, proof)


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
    passed_path = state / "passed-results.json"
    passed = json.loads(passed_path.read_text()) if passed_path.exists() else {}
    streaming = c.get("stream_passed_sample", False)
    if passed and not streaming:
        raise ValueError("Previously started streaming sample cannot be silently disabled")
    client = BaiduDelivery(Path(c["baidu_config"]), state / "original-downloads")
    overrides = [json.loads(Path(p).read_text()) for p in c.get("human_acceptance", [])]
    human = {(r["file"], r["candidate_sha256"]) for r in overrides
             if r["decision"] == "accepted_by_user"}
    concurrency = c.get("codex_concurrency", 1)
    minimum_attempts = c.get("codex_min_server_attempts", 0)
    def inspect(payload):
        row, original, candidate, human_accepted, proof = payload
        name = row["file"]
        group = row.get("_audit_group", "initially-rejected")
        status = "accepted_by_user" if human_accepted else "accepted_without_edit"
        if not human_accepted:
            key = hashlib.sha256(name.encode()).hexdigest()
            reviewed = review_photo(c["codex_executable"], original, candidate,
                                    root / "codex" / group / key,
                                    row.get("regions", []))
            entry = {"file": name, "group": group, **reviewed}
            if not reviewed["verdict"]["acceptable"] and group == "sample-1000":
                status = "needs_review"
            elif not reviewed["verdict"]["acceptable"]:
                target = beneath(state / "repair-targets", name)
                atomic_copy(candidate, target, True, preserve_metadata=True)
                report = repair_one(c["codex_executable"], original, target, entry,
                                    row.get("regions", []), root / "codex-repairs" / key)
                status = report["status"]
        result = {"file": name, "baseline_sha256": row["sha256"],
                "status": status, "codex_reviewed": not human_accepted,
                "server_attempts": proof["attempts"], "server_final_status": proof["status"],
                "eligible_under_current_scope": True, "audit_group": group}
        if group == "sample-1000":
            result.update(audit_only=True, review=reviewed)
        return result

    def progress(active, workers=concurrency, required_attempts=minimum_attempts):
        write_json(state / "status.json", {"phase": "reviewing_and_repairing",
                   "concurrency": workers, "minimum_server_attempts": required_attempts,
                   "scope": ("server_failures_repair_and_passed_sample_audit" if streaming
                              else "committed_server_failures_only"), "active": list(active),
                   "reviewed": len(results),
                   "completed_in_scope": sum(r.get("eligible_under_current_scope") is True for r in results.values()),
                   "completed_before_scope_change": sum(r.get("eligible_under_current_scope") is False for r in results.values()),
                   "accepted": sum(
                       r["status"] in {"passed", "accepted_without_edit", "accepted_by_user"}
                       for r in results.values()),
                   "needs_review": sum(r["status"] == "needs_review" for r in results.values()),
                    "passed_sample_reviewed": len(passed),
                    "passed_sample_accepted": sum(r["status"] == "accepted_without_edit" for r in passed.values()),
                    "passed_sample_needs_review": sum(r["status"] == "needs_review" for r in passed.values()),
                    "sample_target": c.get("codex_sample_size", 1000)})
    while not control.exists():
        batches, sample_batches = {}, {}
        for node, remote_root in c["remote_states"].items():
            try:
                batches[node] = receipt_batch(node, remote_root, cursors.get(node, 0),
                                              maximum=c.get("codex_receipt_prefetch", 8 if streaming else 2))
            except (OSError, subprocess.TimeoutExpired):
                pass
            if streaming:
                key = node + ":sample-1000"
                try:
                    sample_batches[key] = receipt_batch(
                        node, remote_root, cursors.get(key, 0), "sample-1000",
                        c.get("codex_receipt_prefetch", 8))
                except (OSError, subprocess.TimeoutExpired):
                    pass
        snapshots = {**batches, **sample_batches}
        did_work = any(batch["deliveries"] for batch in snapshots.values())
        rejected_work = prepared_deliveries(c, state, batches, results, client, control, human)
        sampled_work = (prepared_deliveries(c, state, sample_batches, passed, client, control, set(),
                                            "sample-1000") if streaming else ())
        for name, result in photo_map(interleave(rejected_work, sampled_work), inspect, concurrency,
                                      stop=control.exists, progress=progress):
            if result.get("audit_group") == "sample-1000":
                passed[name] = result
                write_json(passed_path, passed)
                flagged = [r for r in passed.values() if r["status"] == "needs_review"]
                write_json(state / "passed-needs-review-details.json", flagged)
                (state / "passed-needs-review-relative-paths.txt").write_text(
                    "".join(r["file"] + chr(10) for r in flagged))
            else:
                results[name] = result
                write_json(results_path, results)
        # Both groups share one bounded pool. Acknowledge only fully drained batches.
        if not control.exists():
            for key, batch in snapshots.items():
                cursors[key] = batch["cursor"]
            write_json(cursors_path, cursors)
        if not did_work and not control.exists():
            time.sleep(10)
    write_json(state / "status.json", {"phase": "handoff_ready", "reviewed": len(results),
                                     "passed_sample_reviewed": len(passed),
                                     "passed_sample_needs_review": sum(
                                         r["status"] == "needs_review" for r in passed.values())})


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
    p.add_argument("--group", choices=list(GROUP_FOLDERS), default="initially-rejected")
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--population", type=int)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    try:
        if a.mode == "export":
            if not a.manifest:
                p.error("export requires --manifest")
            export(c, a.manifest, a.group, a.sample_size, a.seed, a.population)
        else:
            local(c)
    except Exception as error:
        state = (Path(c["workflow_state"]).parent / GROUP_FOLDERS[a.group][0] if a.mode == "export"
                 else Path(c["state"]) / "early-codex")
        write_json(state / "status.json", {"phase": "error", "error": type(error).__name__,
                                           "message": str(error)[:400]})
        raise


if __name__ == "__main__":
    main()
