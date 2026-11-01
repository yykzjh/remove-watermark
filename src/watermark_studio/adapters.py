"""Audited, one-time CPU fusion of plain Qwen21 LoRA matrices before quantization.

No runtime adapter branch, pickle deserialization, inferred alpha, or silently
ignored tensor keys. All pairs are checked before any model parameter changes.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from pathlib import Path


def adapter_plan(shapes, target_shapes, *, alpha=0, metadata_alpha=None, pair_alphas=None):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Explicit finite nonnegative LoRA alpha required")
    pairs, consumed, targets = [], set(), set()
    for name, shape in shapes.items():
        suffix=next((tail for tail in (".lora_A.weight",".lora_A.default.weight")
                     if name.endswith(tail)),None)
        if suffix is None:
            continue
        prefix = name.removesuffix(suffix)
        other = prefix+suffix.replace(".lora_A.",".lora_B.")
        target = prefix
        for wrapper in ("diffusion_model.", "transformer."):
            if target.startswith(wrapper):
                target = target.removeprefix(wrapper)
                break
        target += ".weight"

        assignments = []
        if target.endswith(".img_mlp.gate_up.weight") and target not in target_shapes:
            stem=target.removesuffix("gate_up.weight")
            gate,up=stem+"gate_layer.weight",stem+"proj.weight"
            if (gate not in target_shapes or up not in target_shapes
                    or len(target_shapes[gate])!=2 or tuple(target_shapes[gate])!=tuple(target_shapes[up])):
                raise ValueError("Missing matching split SwiGLU targets")
            half=target_shapes[gate][0]
            assignments=[{"target":gate,"rows":[0,half]},{"target":up,"rows":[half,2*half]}]
            expected=(2*half,target_shapes[gate][1])
        else:
            if target not in target_shapes:
                raise ValueError("Unknown LoRA target: "+target)
            expected=target_shapes[target]
        actual_targets=[item["target"] for item in assignments] if assignments else [target]
        if any(value in targets for value in actual_targets) or other not in shapes:
            raise ValueError("Duplicate or incomplete LoRA target: "+target)
        b=shapes[other]
        if (len(shape)!=2 or len(b)!=2 or len(expected)!=2 or shape[0]<1
                or b[1]!=shape[0] or (b[0],shape[1])!=tuple(expected)):
            raise ValueError("LoRA shape mismatch: "+target)
        pair_alpha = (alpha if alpha else (pair_alphas or {}).get(prefix+".alpha",metadata_alpha))
        try:
            effective = float(pair_alpha)
        except (ValueError,TypeError) as error:
            raise ValueError("Missing explicit alpha for "+target) from error
        if not math.isfinite(effective) or effective<=0:
            raise ValueError("LoRA alpha must be finite and positive")
        item={"a":name,"b":other,"target":target,"rank":shape[0],"alpha":effective}
        if assignments:
            item["split_targets"]=assignments
        pairs.append(item)
        consumed.update((name,other))
        if prefix+".alpha" in shapes:
            consumed.add(prefix+".alpha")
        targets.update(actual_targets)
    if not pairs or consumed!=set(shapes):
        raise ValueError("Empty adapter or unsupported tensor keys")
    return pairs


def fuse_cpu_adapter(transformer, path, *, scale=1., alpha=0., expected_sha256=""):
    if not math.isfinite(scale) or not 0<=scale<=2:
        raise ValueError("LoRA scale must be finite in0..2")
    import torch
    from safetensors import safe_open

    began = time.monotonic()
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix != ".safetensors":
        raise ValueError("Only plain safetensors LoRA adapters are supported")
    def sha():
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    fingerprint = sha()
    if expected_sha256 and fingerprint!=expected_sha256:
        raise ValueError("Adapter SHA256 differs from pinned artifact")
    parameters = dict(transformer.named_parameters())
    with safe_open(path,framework="pt",device="cpu") as handle:
        names = handle.keys()  # safe_open exposes keys(), not Mapping iteration.
        shapes = {name:handle.get_slice(name).get_shape() for name in names}
        metadata = handle.metadata() or {}
        pair_alphas = {}
        for name in shapes:
            if name.endswith(".alpha"):
                value=handle.get_tensor(name)
                if value.numel()!=1:
                    raise ValueError("LoRA alpha must be a scalar")
                pair_alphas[name]=value.item()
        plan = adapter_plan(shapes,{k:list(v.shape) for k,v in parameters.items()},
                            alpha=alpha,metadata_alpha=metadata.get("ss_network_alpha"),pair_alphas=pair_alphas)
        tensors = {}
        for item in plan:
            for target in item.get("split_targets",[{"target":item["target"]}]):
                weight = parameters[target["target"]]
                if weight.device.type!="cpu" or weight.dtype not in (torch.float32,torch.float16,torch.bfloat16):
                    raise ValueError("Fuse the unquantized CPU model before CUDA/offload/compilation")
            for name in (item["a"],item["b"]):
                tensor=handle.get_tensor(name)
                if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
                    raise ValueError("Nonfinite or non-floating adapter tensor")
                tensors[name]=tensor
        if sha()!=fingerprint:
            raise ValueError("Adapter changed during validation")
        # All independent targets have already passed validation. Compute one
        # temporary FP32 matrix at a time instead of duplicating the whole DiT.
        with torch.no_grad():
            for item in plan:
                a,b = tensors.pop(item["a"]),tensors.pop(item["b"])
                assignments=item.get("split_targets",[{"target":item["target"],"rows":[0,b.shape[0]]}])
                for assignment in assignments:
                    start,end=assignment["rows"]
                    delta=b[start:end].float().matmul(a.float())
                    delta.mul_(scale*item["alpha"]/item["rank"])
                    weight=parameters[assignment["target"]]
                    fused=weight.float().add(delta)
                    if not torch.isfinite(fused).all().item():
                        raise ValueError("Adapter fusion overflow; model must not be deployed")
                    cast=fused.to(weight.dtype)
                    if not torch.isfinite(cast).all().item():
                        raise ValueError("Fusion overflows the destination dtype; model must not be deployed")
                    weight.copy_(cast)
        if sha()!=fingerprint:
            raise ValueError("Adapter changed during fusion; model must not be deployed")
    return {"path":str(path),"sha256":fingerprint,"pairs":len(plan),
            "ranks":dict(Counter(item["rank"] for item in plan)),
            "split_gate_up_pairs":sum("split_targets" in item for item in plan),
            "target_weights":sum(len(item.get("split_targets",[None])) for item in plan),
            "scale":scale,"alpha_override":alpha,"accumulation_dtype":"float32",
            "runtime_adapter_forwards":0,"seconds":round(time.monotonic()-began,3)}
