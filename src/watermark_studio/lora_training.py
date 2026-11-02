"""Small frozen-backbone LoRA training helpers, exported for existing CPU fusion."""
from __future__ import annotations

import math
import re


def install_attention_lora(model,rank=8,alpha=8.,*,include_mlp=False):
    import torch
    from torch import nn
    from torch.nn import functional
    if type(rank) is not int or rank not in (4,8,16) or not math.isfinite(alpha) or not 0<alpha<=32:
        raise ValueError("Small explicit LoRA rank and positive alpha required")
    if any("lora_" in name for name,_ in model.named_parameters()):
        raise ValueError("Start from an unadapted, unquantized backbone")
    if type(include_mlp) is not bool:raise ValueError("Explicit boolean MLP scope required")
    pattern=re.compile(r"^transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out\.0)$")
    mlp_pattern=re.compile(r"^transformer_blocks\.\d+\.img_mlp\.(?:proj|out|gate_layer)$")
    targets=[(name,module) for name,module in model.named_modules() if pattern.fullmatch(name)]
    blocks=getattr(model,"transformer_blocks",[])
    if not blocks or len(targets)!=4*len(blocks):
        raise ValueError("Every Qwen21 attention block needs all four separate projections")
    if include_mlp:
        extra=[(name,module) for name,module in model.named_modules() if mlp_pattern.fullmatch(name)]
        if len(extra)!=3*len(blocks):raise ValueError("All three separate Qwen21 SwiGLU projections required")
        targets.extend(extra)
    if any(not isinstance(module,nn.Linear) or module.weight.dtype not in (
            torch.float32,torch.float16,torch.bfloat16) for _,module in targets):
        raise ValueError("Unquantized separate Linear projections required")
    for parameter in model.parameters():parameter.requires_grad_(False)
    class TrainLinear(nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base;self.rank=rank;self.alpha=alpha;self.scale=alpha/rank
            self._watermark_training_lora=True
            self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
            self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        @property
        def weight(self):return self.base.weight
        @property
        def bias(self):return self.base.bias
        @property
        def in_features(self):return self.base.in_features
        @property
        def out_features(self):return self.base.out_features
        def forward(self,value):
            result=self.base(value)
            with torch.autocast(device_type=value.device.type,enabled=False):
                delta=functional.linear(functional.linear(value.float(),self.lora_A),self.lora_B)
            return result+(delta*self.scale).to(result.dtype)
    for name,module in targets:
        parent,_,leaf=name.rpartition(".")
        setattr(model.get_submodule(parent),leaf,TrainLinear(module))
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"rank":rank,"alpha":alpha,"scope":"attention_mlp" if include_mlp else "attention","targets":[name for name,_ in targets],"trainable_parameters":count}


def adapter_tensors(model):
    result={}
    for name,module in model.named_modules():
        if getattr(module,"_watermark_training_lora",False):
            result["transformer."+name+".lora_A.weight"]=module.lora_A.detach().float().cpu().contiguous()
            result["transformer."+name+".lora_B.weight"]=module.lora_B.detach().float().cpu().contiguous()
    if not result:raise ValueError("No trained attention LoRA matrices found")
    return result


def flow_objective(prediction,noise,clean,mask,inside_weight=4.,outside_weight=1.):
    """Exact conditional flow matching, with no permission to alter original content."""
    import torch
    if (prediction.shape!=noise.shape or clean.shape!=noise.shape or mask.shape!=noise.shape[:2] or
            prediction.ndim!=3 or not 1<=inside_weight<=8 or not .5<=outside_weight<=inside_weight):
        raise ValueError("Aligned packed targets and bounded nonzero outside preservation weight required")
    if not torch.isfinite(mask).all() or not ((mask>=0)&(mask<=1)).all():
        raise ValueError("Known overlay support must be finite in0..1")
    target=noise.float()-clean.float()
    squared=(prediction.float()-target).square().mean(dim=-1)
    weights=outside_weight+(inside_weight-outside_weight)*mask.float()
    return (squared*weights).sum()/weights.sum()
