"""Small numerical hardware check, not an image-quality or serving-throughput benchmark."""
import argparse
import copy
import json
import os
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args();assert a.gpu in (0,1,2,3) and not a.output.exists()
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
import torch
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    PerTensor,
    quantize_,
)

torch.set_num_threads(1);torch.manual_seed(42)
assert torch.cuda.mem_get_info()[0]>40*2**30
base=torch.nn.Sequential(torch.nn.Linear(512,512,bias=False)).to(device="cuda",dtype=torch.bfloat16).eval().requires_grad_(False)
first=torch.randn(32,512,device="cuda",dtype=torch.bfloat16)
second=torch.randn_like(first)*32
rows=[]
with torch.inference_mode():
    reference=base(first).float()
    for name,granularity in [("tensor",PerTensor()),("row",PerRow())]:
        model=copy.deepcopy(base)
        try:
            quantize_(model,Float8DynamicActivationFloat8WeightConfig(granularity=granularity))
            alone=model(first).float()
            combined=model(torch.cat([first,second]))[:32].float()
            torch.cuda.synchronize()
            rows.append({"granularity":name,"status":"ok",
                "batch_dependence_mae":float((alone-combined).abs().mean()),
                "batch_dependence_max":float((alone-combined).abs().max()),
                "reference_mae_alone":float((alone-reference).abs().mean()),
                "reference_mae_combined":float((combined-reference).abs().mean())})
        except (RuntimeError,ValueError,TypeError,AssertionError) as error:
            rows.append({"granularity":name,"status":"unsupported","error":f"{type(error).__name__}: {error}"[:600]})
        del model
report={"gpu":a.gpu,"hardware":torch.cuda.get_device_name(0),"torch":torch.__version__,
        "scope":"Synthetic one-linear numeric smoke only; no image-quality claim","rows":rows}
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
