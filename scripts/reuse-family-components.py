"""Reuse local weights only when reconstructed bytes match publisher SHA256 exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();target=a.target.resolve()
    if source==target or target.is_relative_to(source) or source.is_relative_to(target):
        p.error("Separate existing source and new target model snapshots required")
    data=json.loads(a.manifest.read_text())
    files={r["Path"]:r for r in data["Data"]["Files"] if r["Type"]=="blob"}
    index=target/"text_encoder/model.safetensors.index.json";entry=files["text_encoder/model.safetensors.index.json"]
    if digest(index)!=entry["Sha256"] or index.stat().st_size!=entry["Size"]:
        raise ValueError("Publisher-bound target shard index required")
    groups={}
    for key,name in json.loads(index.read_text())["weight_map"].items():
        if not re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors",name):
            raise ValueError("Bounded named encoder shards required")
        groups.setdefault(name,[]).append(key)
    from safetensors import safe_open
    from safetensors.torch import save_file

    rows=[]
    def export(component,name,tensors):
        path=target/component/name;meta=files[component+"/"+name]
        if (not re.fullmatch(r"[a-f0-9]{64}",meta["Sha256"]) or
                type(meta["Size"]) is not int or not 0<meta["Size"]<=24*2**30):
            raise ValueError("Exact bounded publisher weight digest required")
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if digest(path)!=meta["Sha256"]:
                raise ValueError("Existing target weight differs from publisher; do not overwrite it")
            rows.append({"file":component+"/"+name,"matched":True,"already_exists":True})
            return
        part=path.with_name(path.name+".reuse.part")
        if part.exists():
            raise ValueError("New reuse scratch file required")
        save_file(tensors,str(part),metadata={"format":"pt"})
        actual=digest(part)
        good=part.stat().st_size==meta["Size"] and actual==meta["Sha256"]
        if good:os.replace(part,path)
        else:part.unlink()  # Only this invocation's unsuccessful scratch file.
        row={"file":component+"/"+name,"matched":good,"publisher_sha256":meta["Sha256"],"actual_sha256":actual}
        rows.append(row);print(json.dumps(row),flush=True)
    with safe_open(source/"text_encoder/model.safetensors",framework="pt",device="cpu") as handle:
        keys=set(handle.keys());all_keys={key for group in groups.values() for key in group}
        if keys==all_keys:
            for name,group in sorted(groups.items()):
                export("text_encoder",name,{key:handle.get_tensor(key) for key in group})
        else:
            rows.append({"file":"text_encoder","matched":False,"reason":"source/target tensor keys differ",
                "source_only":len(keys-all_keys),"target_only":len(all_keys-keys)})
    import torch
    with safe_open(source/"vae/diffusion_pytorch_model.safetensors",framework="pt",device="cpu") as handle:
        export("vae","diffusion_pytorch_model.safetensors",
            {key:handle.get_tensor(key).to(torch.float32) for key in handle.keys()})  # noqa: SIM118 - safe_open is not a dict
    report={"scope":"Exact publisher byte reuse only; no source writes or inferred weight equivalence",
        "source":str(source),"target":str(target),"manifest_sha256":digest(a.manifest),"rows":rows}
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"matched":sum(r["matched"] for r in rows),"files":len(rows)}),flush=True)


if __name__=="__main__":
    main()
