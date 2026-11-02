"""Bounded source-SHA partitioning; no decoded-image exposure or model execution."""
from __future__ import annotations

import hashlib
import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def source_sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def unique_sources(paths,forbidden,count):
    if type(count) is not int or not 1<=count<=4096:
        raise ValueError("Bounded positive source pool required")
    forbidden=set(forbidden);selected=[];seen=set();visited=0;iterator=iter(paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        while len(selected)<count:
            chunk=list(itertools.islice(iterator,64))
            if not chunk:break
            hashes=list(pool.map(source_sha,chunk));visited+=len(chunk)
            for path,sha in zip(chunk,hashes,strict=True):
                if sha in forbidden or sha in seen:continue
                selected.append({"path":str(Path(path).resolve()),"sha256":sha});seen.add(sha)
                if len(selected)==count:break
    if len(selected)!=count:raise ValueError("Insufficient unexposed unique source bytes")
    return selected,visited
