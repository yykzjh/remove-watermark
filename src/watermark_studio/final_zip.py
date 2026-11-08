"""Rebuild a byte-identical final ZIP from distributed outputs and verified overrides."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath

from .dataset_transfer import write_json


class _ForwardWriter:
    def __init__(self, stream):
        self.stream = stream
        self.position = 0
        self.sha = hashlib.sha256()
        self.md5 = hashlib.md5()

    def write(self, data):
        n = self.stream.write(data)
        if n != len(data):
            raise OSError("Short archive write")
        self.position += n
        self.sha.update(data)
        self.md5.update(data)
        return n

    def tell(self):
        return self.position

    def seek(self, *_):
        raise io.UnsupportedOperation("Forward archive writer")

    def flush(self):
        self.stream.flush()


def _relative(name, root):
    if name == root:
        return "."
    if not name.startswith(root + "/"):
        raise ValueError("Archive entry is outside its root")
    value = name[len(root) + 1:]
    if (not value or "\\" in value or ":" in value or
            any(p in {"", ".", ".."} for p in value.split("/")) or
            str(PurePosixPath(value)) != value):
        raise ValueError("Unsafe archive entry")
    return value


def _info(row, directory=False):
    name = row["Path"].rstrip("/") + ("/" if directory else "")
    dt = datetime.datetime.fromisoformat(row["Modified"])
    info = zipfile.ZipInfo(name, (max(1980, min(2107, dt.year)),
                                 dt.month, dt.day, dt.hour, dt.minute, dt.second))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = ((0o40755 if directory else 0o100644) << 16) | (0x10 if directory else 0)
    info.file_size = int(row["Size"])
    return info


def build_zip(bundle, roots, destination, state, bundle_sha256=None):
    bundle, destination, state = Path(bundle), Path(destination), Path(state)
    roots = list(map(Path, roots))
    state.mkdir(parents=True, exist_ok=True)
    if bundle_sha256:
        sha = hashlib.sha256()
        with bundle.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != bundle_sha256:
            raise ValueError("Delivery bundle SHA256 mismatch")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing archive")
    temporary = Path(str(destination) + ".part")
    if temporary.exists():
        raise FileExistsError("A previous partial archive requires inspection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as patch:
        members = patch.infolist()
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Duplicate bundle members")
        if patch.getinfo("delivery-plan.json").file_size > 64 * 1024 * 1024:
            raise ValueError("Oversized delivery plan")
        plan = json.loads(patch.read("delivery-plan.json"))
        root, rows, expected = plan["root"], plan["entries"], plan["files"]
        if not root or "/" in root or "\\" in root or ":" in root or root in {".", ".."}:
            raise ValueError("Unsafe archive root")
        relative = {r["Path"]: _relative(r["Path"], root) for r in rows}
        if len(relative) != len(rows):
            raise ValueError("Duplicate archive entry")
        directories = [r for r in rows if r.get("Attributes", "").startswith("D")]
        files = [r for r in rows if not r.get("Attributes", "").startswith("D")]
        if {relative[r["Path"]]: int(r["Size"]) for r in files} != {
                name: int(r["size"]) for name, r in expected.items()}:
            raise ValueError("Archive plan inventory mismatch")
        overrides = set(plan["overrides"])
        if not overrides <= set(expected):
            raise ValueError("Unknown override")
        if {m.filename for m in members} != {"delivery-plan.json"} | {"payload/" + n for n in overrides}:
            raise ValueError("Unexpected delivery bundle member")
        nonempty = sorted([r for r in files if int(r["Size"])], key=lambda r: int(r["Block"]))
        if [int(r["Block"]) for r in nonempty] != list(range(len(nonempty))):
            raise ValueError("Archive data block order changed")
        progress = {"phase": "building", "files": len(files), "directories_including_root": len(directories),
                    "completed": 0, "bytes_completed": 0, "destination": str(destination)}
        write_json(state / "status.json", progress)
        last = time.monotonic()

        def source(name):
            if name in overrides:
                return patch.open("payload/" + name)
            owner = expected[name].get("source_root")
            if type(owner) is not int or not 0 <= owner < len(roots):
                raise ValueError("Missing authoritative source owner: " + name)
            candidate = roots[owner] / name
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError("Missing or linked source: " + name)
            return candidate.open("rb")

        with temporary.open("wb", buffering=8 * 1024 * 1024) as output:
            writer = _ForwardWriter(output)
            with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for row in sorted(directories, key=lambda r: (r["Path"].count("/"), r["Path"])):
                    archive.writestr(_info(row, True), b"")
                ordered = [r for r in files if not int(r["Size"])] + nonempty
                for row in ordered:
                    name = relative[row["Path"]]
                    sha, count = hashlib.sha256(), 0
                    with source(name) as stream, archive.open(_info(row), "w", force_zip64=True) as target:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            target.write(block)
                            sha.update(block)
                            count += len(block)
                    if count != int(row["Size"]) or sha.hexdigest() != expected[name]["sha256"]:
                        raise ValueError("Source bytes differ from final manifest: " + name)
                    progress["completed"] += 1
                    progress["bytes_completed"] += count
                    if time.monotonic() - last >= 5:
                        write_json(state / "status.json", progress)
                        last = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        sha, md5 = writer.sha.hexdigest(), writer.md5.hexdigest()
        if plan.get("expected_archive_sha256") and sha != plan["expected_archive_sha256"]:
            raise ValueError("Rebuilt ZIP differs from the local final ZIP")
        with zipfile.ZipFile(temporary) as archive:
            actual = archive.infolist()
            if len(actual) != len(rows):
                raise ValueError("Final ZIP entry count mismatch")
            by_name = {r.filename.rstrip("/"): r for r in actual}
            for row in rows:
                entry = by_name[row["Path"]]
                if entry.file_size != int(row["Size"]):
                    raise ValueError("Final ZIP size mismatch")
                if not entry.is_dir() and entry.CRC != int(row.get("CRC") or "0", 16):
                    raise ValueError("Final ZIP CRC metadata mismatch")
        temporary.replace(destination)
        result = progress | {"phase": "built", "sha256": sha, "md5": md5,
                             "bytes": destination.stat().st_size,
                             "identical_to_local_zip": bool(plan.get("expected_archive_sha256"))}
        write_json(state / "report.json", result)
        write_json(state / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_zip(args.bundle, args.root, args.destination, args.state, args.bundle_sha256)
    except Exception as error:
        write_json(args.state / "status.json", {"phase": "error", "error": type(error).__name__,
                                              "message": str(error)[:400]})
        raise
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
