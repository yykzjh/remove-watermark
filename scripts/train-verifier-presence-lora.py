"""Offline BF16 LoRA for one output-only predicate; base QA weights stay frozen."""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path


def validate_manifest(path):
    from watermark_studio.training_features import digest
    data=json.loads(path.read_text())
    if (data.get("version")!=1 or data.get("state")!="completed"
            or data.get("task")!="output_only_digital_overlay_presence"):
        raise ValueError("Frozen presence-task data required")
    forbidden_path=Path(data["forbidden_manifest"])
    if digest(forbidden_path)!=data["forbidden_manifest_sha256"]:
        raise ValueError("Frozen source exclusions changed")
    forbidden=set(json.loads(forbidden_path.read_text()))
    rows=data["rows"]
    if not 100<=len(rows)<=6000 or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Unique bounded training rows required")
    groups={}
    files={}
    clean_splits={}
    from watermark_studio.quality_presence import canonical_answer
    for row in rows:
        canonical_answer(row["overlay_present"],row["uncertain"],row["area"])
        if row["source_original_sha256"] in forbidden:
            raise ValueError("Training contains a sealed/control parent")
        if row["split"] not in {"train","validation"}:
            raise ValueError("Explicit original-parent split required")
        groups.setdefault(row["source_original_sha256"],set()).add(row["split"])
        if not row["overlay_present"] and not row["uncertain"]:
            clean_splits.setdefault(row["after_sha256"],set()).add(row["split"])
        for field in ("before","after"):
            prior=files.setdefault(row[field],row[field+"_sha256"])
            if prior!=row[field+"_sha256"]:
                raise ValueError("Contradictory immutable image hash")
    if any(len(v)!=1 for v in clean_splits.values()):
        raise ValueError("Identical clean images cannot cross source splits")
    if any(len(v)!=1 for v in groups.values()):
        raise ValueError("One original photo cannot cross train/validation")
    for filename,sha in files.items():
        if digest(filename)!=sha:
            raise ValueError("Frozen image changed: "+filename)
    return data,digest(path)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=[0,1,2,3],required=True)
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--gradient-accumulation",type=int,default=2)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--alpha",type=int,default=8)
    p.add_argument("--vision-rank",type=int,choices=[0,4,8],default=0)
    p.add_argument("--vision-learning-rate",type=float,default=1e-5)
    p.add_argument("--initial-adapter",type=Path)
    p.add_argument("--initial-adapter-sha256",default="")
    p.add_argument("--freeze-language-lora",action="store_true")
    p.add_argument("--learning-rate",type=float,default=5e-5)
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--validate-every",type=int,default=50)
    p.add_argument("--validation-limit",type=int,default=32)
    p.add_argument("--gradient-checkpointing",action="store_true")
    p.add_argument("--seed",type=int,default=2026110381)
    a=p.parse_args()
    assert not a.target.exists() and 1<=a.steps<=1000 and 1<=a.gradient_accumulation<=8
    assert a.rank in {4,8,16,32} and 1<=a.alpha<=64 and 0<a.learning_rate<=2e-4
    assert bool(a.initial_adapter)==bool(a.initial_adapter_sha256)
    assert not a.freeze_language_lora or (a.initial_adapter and a.vision_rank)
    assert 0<a.vision_learning_rate<=2e-4
    assert a.save_every>0 and a.validate_every>0 and 8<=a.validation_limit<=256
    from watermark_studio.training_reservation import offline_reservation
    reservation=offline_reservation(a.gpu)
    data,manifest_sha=validate_manifest(a.manifest)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.nn import functional as F
    from transformers.utils import logging

    from watermark_studio.config import DeploymentConfig, Region
    from watermark_studio.images import load_photo
    from watermark_studio.quality import _structured_vqa, allowed_review_tokens
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.quality_presence import (
        FIELDS,
        PROMPT,
        branch_weights,
        canonical_answer,
        output_chat,
    )
    from watermark_studio.training_features import digest

    logging.disable_progress_bar();torch.set_num_threads(1);torch.manual_seed(a.seed)
    random.seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    base_path=Path.home()/".cache/watermark-models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    from watermark_studio.model_identity import compare_model_weights
    editor=base_path.parents[2]/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master/text_encoder"
    identity=compare_model_weights(editor,base_path)
    if identity["relation"]!="different_weights":
        raise ValueError("Presence verifier must be independent of the image editor")
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
    base=load_quality_encoder(base_path,cfg,"cuda:0")
    processor=load_quality_processor(base_path)
    eos=base.generation_config.eos_token_id or processor.tokenizer.eos_token_id
    eos_ids=[eos] if isinstance(eos,int) else list(eos)
    tree=allowed_review_tokens(processor.tokenizer,eos_ids,FIELDS)
    train=[row for row in data["rows"] if row["split"]=="train"]
    validation=[row for row in data["rows"] if row["split"]=="validation"]
    if not train or not validation:
        raise ValueError("Both independent source splits required")
    # Round-robin classes in the smoke validation; full evaluation is separate.
    categories=collections.defaultdict(list)
    for row in validation:
        categories[(row["parent_kind"],row["overlay_present"],row["uncertain"])].append(row)
    selected=[]
    while len(selected)<min(a.validation_limit,len(validation)):
        for group in categories.values():
            if group and len(selected)<a.validation_limit:
                selected.append(group.pop(0))
    by_id={row["id"]:row for row in data["rows"]}

    @functools.lru_cache(maxsize=16)
    def chat_for(row_id):
        row=by_id[row_id]
        return output_chat(load_photo(row["before"])[0],load_photo(row["after"])[0],
                           [Region.model_validate(r) for r in row["regions"]],detail_layout=row.get("detail_layout"))

    @functools.lru_cache(maxsize=16)
    def completion(row_id):
        row=by_id[row_id];chat=chat_for(row_id)
        inputs=processor.apply_chat_template([chat],tokenize=True,add_generation_prompt=True,
            processor_kwargs={"padding":True},return_dict=True,return_tensors="pt")
        inputs.pop("token_type_ids",None);inputs.pop("position_ids",None)
        # The vision tower itself casts to BF16; store the same actual inputs compactly.
        if "pixel_values" in inputs:
            inputs["pixel_values"]=inputs["pixel_values"].to(torch.bfloat16)
        answer=canonical_answer(row["overlay_present"],row["uncertain"],row["area"])
        ids=processor.tokenizer.encode(answer,add_special_tokens=False)+[eos_ids[0]]
        weights=branch_weights(tree,ids)
        return dict(inputs),torch.tensor(ids,dtype=torch.long)[None],torch.tensor(weights)[None]

    def batch_for(row_id):
        inputs,target,weights=completion(row_id)
        result={k:v.to("cuda:0") for k,v in inputs.items()}
        target=target.to("cuda:0");weights=weights.to("cuda:0")
        result["input_ids"]=torch.cat([result["input_ids"],target],dim=1)
        result["attention_mask"]=torch.cat([result["attention_mask"],torch.ones_like(target)],dim=1)
        if "mm_token_type_ids" in result:
            result["mm_token_type_ids"]=torch.cat([result["mm_token_type_ids"],torch.zeros_like(target)],dim=1)
        return result,target,weights

    # Prove disabling the adapter preserves the original QA forward numerically.
    sample,_,_=batch_for(train[0]["id"])
    with torch.inference_mode():
        baseline=base(**sample,use_cache=False,logits_to_keep=1).logits.detach().clone()
    base.requires_grad_(False)
    targets=["q_proj","v_proj"]+(["qkv"] if a.vision_rank else [])
    model=get_peft_model(base,LoraConfig(r=a.rank,lora_alpha=a.alpha,lora_dropout=0.,
        target_modules=targets,bias="none",task_type=TaskType.CAUSAL_LM,
        rank_pattern={"qkv":a.vision_rank} if a.vision_rank else {},
        alpha_pattern={"qkv":a.vision_rank} if a.vision_rank else {}))
    if a.initial_adapter:
        from peft.utils.save_and_load import set_peft_model_state_dict
        from safetensors.torch import load_file
        initial=json.loads((a.initial_adapter/"presence-task.json").read_text())
        if (digest(a.initial_adapter/"adapter_model.safetensors")!=a.initial_adapter_sha256 or
                initial["weights_sha256"]!=a.initial_adapter_sha256 or
                initial["prompt_sha256"]!=hashlib.sha256(PROMPT.encode()).hexdigest() or
                initial["training_manifest_sha256"]!=manifest_sha or
                initial.get("base_identity",{}).get("verifier_sha256")!=identity["verifier_sha256"]):
            raise ValueError("Initial adapter/base/protocol/source-split identity mismatch")
        saved=load_file(a.initial_adapter/"adapter_model.safetensors",device="cpu")
        if any("lora_" not in key for key in saved):
            raise ValueError("Initial adapter cannot replace any frozen base weights")
        loaded=set_peft_model_state_dict(model,saved,adapter_name="default")
        if loaded.unexpected_keys or any("lora_" in key and not (a.vision_rank and "visual" in key)
                                         for key in loaded.missing_keys):
            raise ValueError("Initial language adapter did not fully initialize its parameters")
        del saved
    if a.freeze_language_lora:
        for name,param in model.named_parameters():
            if "lora_" in name and "visual" not in name:
                param.requires_grad_(False)
    with model.disable_adapter(),torch.inference_mode():
        replay=model(**sample,use_cache=False,logits_to_keep=1).logits.detach()
    if not torch.equal(baseline,replay):
        raise ValueError("Disabled LoRA changed the preserved base QA forward")
    del baseline,replay,sample
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_" not in n or "lm_head" in n or
                        ("visual" in n and (not a.vision_rank or ".qkv." not in n)) for n in names):
        raise ValueError("Only explicitly selected attention LoRA parameters may train")
    if a.freeze_language_lora and any("visual" not in n for n in names):
        raise ValueError("Frozen language adapter gained trainable parameters")
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
    model.config.use_cache=False
    parameters=[p for p in model.parameters() if p.requires_grad]
    groups=[]
    for vision,name,lr in [(False,"language",a.learning_rate),(True,"vision",a.vision_learning_rate)]:
        selected_parameters=[p for n,p in model.named_parameters() if p.requires_grad and ("visual" in n)==vision]
        if selected_parameters:
            groups.append({"params":selected_parameters,"lr":lr,"base_lr":lr,"name":name})
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=.01)
    a.target.mkdir(parents=True)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    run_config={"revision":revision,"arguments":{key:str(value) if isinstance(value,Path) else value
                                               for key,value in vars(a).items()},
        "manifest_sha256":manifest_sha,"base_model":str(base_path),"base_identity":identity,
        "prompt_sha256":hashlib.sha256(PROMPT.encode()).hexdigest(),
        "trainable_parameters":sum(p.numel() for p in parameters),"trainable_modules":names,
        "base_disabled_bitwise_equal":True,"train_rows":len(train),"validation_rows":len(validation),
        "base_qa_natural_content_detector_training":False}
    (a.target/"config.json").write_text(json.dumps(run_config,indent=2))
    history=[];began=time.monotonic();order=[];rng=random.Random(a.seed)
    def evaluate(step):
        model.eval();results=[]
        for row in selected:
            started=time.monotonic()
            result,_=_structured_vqa([chat_for(row["id"])],FIELDS,model,processor,"cuda:0",96)
            predicted=result[0]
            results.append({"id":row["id"],"overlay_present":row["overlay_present"],
                "uncertain":row["uncertain"],"prediction":predicted,"seconds":time.monotonic()-started})
        report={"step":step,"rows":len(results),
            "unsafe_clear":sum((r["overlay_present"] or r["uncertain"]) and
                              not r["prediction"]["overlay_present"] and not r["prediction"]["uncertain"] for r in results),
            "clean_refused":sum(not r["overlay_present"] and not r["uncertain"] and
                               (r["prediction"]["overlay_present"] or r["prediction"]["uncertain"]) for r in results),
            "results":results}
        (a.target/f"validation-{step:04d}.json").write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)
        model.train()
    evaluate(0)
    optimizer.zero_grad(set_to_none=True)
    with (a.target/"training.jsonl").open("x") as log:
        for step in range(1,a.steps+1):
            factor=min(1.,step/max(1,a.warmup))
            if step>a.warmup:
                factor=.1+.9*.5*(1+math.cos(math.pi*(step-a.warmup)/max(1,a.steps-a.warmup)))
            for group in optimizer.param_groups:
                group["lr"]=group["base_lr"]*factor
            losses=[]
            model.train()
            for _ in range(a.gradient_accumulation):
                if not order:
                    order=list(range(len(train)));rng.shuffle(order)
                row=train[order.pop()]
                inputs,target,weights=batch_for(row["id"])
                # Only completion logits are materialized, avoiding a prefix x
                # 150k-vocabulary tensor without changing teacher-forced attention.
                logits=model(**inputs,use_cache=False,logits_to_keep=target.shape[1]+1).logits[:,:-1]
                if logits.shape[1]!=target.shape[1]:
                    raise ValueError("Completion/logit shift drift")
                per_token=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                                          target.reshape(-1),reduction="none").reshape_as(weights)
                loss=(per_token*weights).sum()/weights.sum()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite presence LoRA loss")
                (loss/a.gradient_accumulation).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            row={"step":step,"loss":sum(losses)/len(losses),"lr":optimizer.param_groups[0]["lr"],
                 "group_lr":{group["name"]:group["lr"] for group in optimizer.param_groups},
                 "seconds":round(time.monotonic()-began,3),
                 "peak_cuda_bytes":torch.cuda.max_memory_allocated()}
            history.append(row);log.write(json.dumps(row)+"\n");log.flush()
            if step%10==0:
                print(json.dumps(row),flush=True)
            if step%a.save_every==0 or step==a.steps:
                folder=a.target/f"adapter-step{step:04d}"
                model.save_pretrained(folder,safe_serialization=True)
                (folder/"presence-task.json").write_text(json.dumps({
                     "version":1,"task":"output_only_digital_overlay_presence","step":step,
                    "vision_rank":a.vision_rank,"freeze_language_lora":a.freeze_language_lora,
                    "initial_adapter_sha256":a.initial_adapter_sha256,
                    "base_model":str(base_path),"base_identity":identity,"prompt_sha256":run_config["prompt_sha256"],
                    "training_manifest_sha256":manifest_sha,"source_revision":revision,
                    "weights_sha256":digest(folder/"adapter_model.safetensors"),
                    "base_qa_guards_preserved":True},indent=2))
            if step%a.validate_every==0 or step==a.steps:
                evaluate(step)
    assert digest(a.manifest)==manifest_sha
    assert revision==subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert reservation==offline_reservation(a.gpu)
    report={"state":"completed","steps":a.steps,"seconds":time.monotonic()-began,
        "peak_cuda_bytes":torch.cuda.max_memory_allocated(),"manifest_sha256":manifest_sha,
        "revision":revision,"base_disabled_bitwise_equal":True,"trained_parameters":run_config["trainable_parameters"],
        "production_adopted":False,"scope":"Dedicated watermark-presence predicate only; independent quality validation required"}
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
