"""Download one pinned public adapter on the dev machine; verify size and SHA256."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--repo",default="ausboss/Qwen-Image-2.1-Consistency-LoRA")
    p.add_argument("--revision",default="58897a02f4f60f88376eab5676738dd45e5ed1a2")
    p.add_argument("--filename",default="qwen-image-2.1-consistency.safetensors")
    p.add_argument("--sha256",default="4f44ada1be2189cc23b3d010f9603543403f48454e2f76842a3d30109b20bd63")
    p.add_argument("--size",type=int,default=159436496)
    p.add_argument("--endpoint",choices=["https://huggingface.co","https://hf-mirror.com"],default="https://hf-mirror.com")
    a=p.parse_args()
    if (not re.fullmatch(r"[\w.-]+/[\w.-]+",a.repo) or not re.fullmatch(r"[0-9a-f]{40}",a.revision)
            or Path(a.filename).name!=a.filename or not a.filename.endswith(".safetensors")
            or not re.fullmatch(r"[0-9a-f]{64}",a.sha256) or not 0<a.size<=1024**3):
        p.error("Pinned repository revision, basename, SHA and bounded byte count required")
    root=Path.home()/".cache/watermark-models/adapters"/a.repo.replace("/","--")/a.revision
    root.mkdir(parents=True,exist_ok=True)
    target=root/a.filename
    def fingerprint(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    if target.exists():
        if target.stat().st_size!=a.size or fingerprint(target)!=a.sha256:
            raise ValueError("Existing adapter differs; do not overwrite")
    else:
        url=f"{a.endpoint}/{a.repo}/resolve/{a.revision}/{a.filename}"
        request=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
        temporary=None
        try:
            with urllib.request.urlopen(request,timeout=60) as response, tempfile.NamedTemporaryFile(
                    dir=root,prefix=a.filename+".",suffix=".download",delete=False) as stream:
                temporary=Path(stream.name)
                received=0
                while chunk:=response.read(min(1024**2,a.size+1-received)):
                    received+=len(chunk)
                    if received>a.size:
                        raise ValueError("Adapter response exceeded pinned byte count")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size!=a.size or fingerprint(temporary)!=a.sha256:
                raise ValueError("Adapter size or SHA256 mismatch")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    receipt={"repo":a.repo,"revision":a.revision,"filename":a.filename,"bytes":a.size,"sha256":a.sha256,
             "verified":True,"path":str(target),"scope":"download only; no training or production deployment"}
    (root/"verified-adapter.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
