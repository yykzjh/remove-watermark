"""Canonical safetensors weight identity, independent of sharding and config formatting."""
from __future__ import annotations

import hashlib
import json
import os
import struct
from contextlib import ExitStack
from pathlib import Path

VERSION = 1
CHUNK = 8 * 1024 * 1024


def weight_fingerprint(directory):
    root = Path(directory).expanduser().resolve()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        names = sorted(set(mapping.values()))
    else:
        mapping = None
        names = ["model.safetensors"] if (root / "model.safetensors").is_file() else []
    if not names:
        raise ValueError("No supported safetensors checkpoint found")
    files = []
    for name in names:
        file = (root / name).resolve()
        if not file.is_relative_to(root) or file.suffix != ".safetensors":
            raise ValueError("Weight index must stay inside its model directory")
        info = file.stat()
        probe = hashlib.sha256()
        with file.open("rb") as stream:
            for offset in (0, max(0, info.st_size//2-2048), max(0, info.st_size-4096)):
                stream.seek(offset)
                probe.update(stream.read(4096))
        files.append({"name": name, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                      "ctime_ns": info.st_ctime_ns, "probe_sha256": probe.hexdigest()})
    index_digest = hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None
    cache = root / ".watermark-weight-fingerprint-v1.json"
    try:
        saved = json.loads(cache.read_text())
        if (saved.get("version") == VERSION and saved.get("files") == files
                and saved.get("index_sha256") == index_digest
                and len(saved.get("sha256", "")) == 64
                and isinstance(saved.get("tensor_count"), int)):
            return saved
    except (OSError, ValueError, TypeError):
        pass
    tensors = {}
    with ExitStack() as stack:
        streams = {}
        for record in files:
            stream = stack.enter_context((root / record["name"]).open("rb"))
            streams[record["name"]] = stream
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError("Truncated safetensors header")
            length = struct.unpack("<Q", length_bytes)[0]
            if not 0 < length <= min(16 * 1024 * 1024, record["size"] - 8):
                raise ValueError("Invalid safetensors header length")
            header = json.loads(stream.read(length))
            base = 8 + length
            for key, desc in header.items():
                if key == "__metadata__":
                    continue
                lo, hi = desc["data_offsets"]
                if (type(lo) is not int or type(hi) is not int
                        or not 0 <= lo <= hi <= record["size"] - base or key in tensors):
                    raise ValueError("Invalid or repeated safetensors tensor")
                if mapping is not None and mapping.get(key) != record["name"]:
                    raise ValueError("Weight index does not match tensor storage")
                tensors[key] = (record["name"], base + lo, hi - lo, desc["dtype"], desc["shape"])
        if mapping is not None and set(mapping) != set(tensors):
            raise ValueError("Weight index is incomplete")
        digest = hashlib.sha256()
        total = 0
        for key in sorted(tensors):
            name, start, size, dtype, shape = tensors[key]
            metadata = json.dumps([key, dtype, shape], separators=(",", ":")).encode()
            digest.update(len(metadata).to_bytes(8, "little"))
            digest.update(metadata)
            stream = streams[name]
            stream.seek(start)
            remaining = size
            while remaining:
                chunk = stream.read(min(CHUNK, remaining))
                if not chunk:
                    raise ValueError("Truncated tensor data")
                digest.update(chunk)
                remaining -= len(chunk)
            total += size
    result = {"version": VERSION, "files": files, "index_sha256": index_digest, "sha256": digest.hexdigest(),
              "tensor_count": len(tensors), "weight_bytes": total}
    temp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(result))
        temp.replace(cache)
    except OSError:
        temp.unlink(missing_ok=True)  # Read-only model stores still allow the audit.
    return result


def compare_model_weights(editor_directory, verifier_directory):
    try:
        editor = weight_fingerprint(editor_directory)
        verifier = weight_fingerprint(verifier_directory)
        return {"relation": "same_base_weights" if editor["sha256"] == verifier["sha256"]
                else "different_weights", "method": "canonical_tensor_sha256_v1",
                "editor_sha256": editor["sha256"], "verifier_sha256": verifier["sha256"],
                "editor_tensors": editor["tensor_count"], "verifier_tensors": verifier["tensor_count"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"relation": "unknown", "method": "canonical_tensor_sha256_v1",
                "error": str(error)[:240]}
