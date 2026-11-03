"""Experimental native-resolution OCR candidates; no digital/physical or quality verdict."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

OFFICIAL_SMALL_SHA256="89a96a8adc4e9cd0c994098edc76022e496d35844392562b4694c8fbc583f2da"


def normalized_candidates(boxes,scores,size):
    width,height=size
    if len(boxes)!=len(scores) or len(boxes)>3000:
        raise ValueError("Invalid bounded OCR candidate count")
    records=[]
    for box,score in zip(boxes,scores):
        points=np.asarray(box,dtype=float)
        if points.shape==(4,):
            points=points.reshape(2,2)
        if points.shape not in {(4,2),(2,2)} or not np.isfinite(points).all() or not np.isfinite(score):
            raise ValueError("Invalid native OCR geometry")
        low,high=points.min(axis=0),points.max(axis=0)
        if (np.any(low<-.001) or high[0]>width+.001 or high[1]>height+.001
                or np.any(high<=low) or not 0<=score<=1):
            raise ValueError("OCR candidate escaped the photograph")
        records.append({"box":[float(low[0]/width),float(low[1]/height),
                               float(high[0]/width),float(high[1]/height)],"score":float(score)})
    return records


class NativeTextLocator:
    def __init__(self,model_path,device="cuda:0",*,resolution=4096,box_threshold=.3,
                 pixel_threshold=.1,clahe_clip=0,weights_sha256=OFFICIAL_SMALL_SHA256):
        import torch
        from transformers import AutoImageProcessor, PPOCRV6SmallDetForObjectDetection

        path=Path(model_path)
        if (hashlib.sha256((path/"model.safetensors").read_bytes()).hexdigest()!=weights_sha256 or
                json.loads((path/"config.json").read_text()).get("model_type")!="pp_ocrv6_small_det"):
            raise ValueError("Pinned native OCR architecture/weight mismatch")
        if resolution not in {2048,3072,4096} or not .05<=pixel_threshold<=.5 or not .1<=box_threshold<=.7:
            raise ValueError("Invalid bounded OCR candidate configuration")
        if not 0<=clahe_clip<=8:
            raise ValueError("Invalid diagnostic contrast")
        self.device=device;self.resolution=resolution
        self.box_threshold=box_threshold;self.pixel_threshold=pixel_threshold;self.clahe_clip=clahe_clip
        self.weights_sha256=weights_sha256
        self.model,info=PPOCRV6SmallDetForObjectDetection.from_pretrained(
            path,local_files_only=True,use_safetensors=True,dtype=torch.float32,output_loading_info=True)
        if any(not key.endswith("num_batches_tracked") for key in info["missing_keys"]) or info["unexpected_keys"]:
            raise ValueError("Native OCR checkpoint did not fully initialize its detector")
        self.model=self.model.to(device).eval().requires_grad_(False)
        self.processor=AutoImageProcessor.from_pretrained(path,local_files_only=True)
        # Configure supported runtime resize fields, never modify cached model files.
        self.processor.limit_side_len=resolution
        self.processor.limit_type="max"
        self.processor.max_side_limit=resolution

    def inspect(self,photo):
        from types import SimpleNamespace

        import torch
        from PIL import Image

        if self.model.training:
            raise ValueError("OCR candidates require evaluation mode")
        if str(self.device).startswith("cuda"):torch.cuda.synchronize()
        began=time.monotonic();diagnostic=photo.convert("RGB")
        if self.clahe_clip:
            import cv2
            luminance=cv2.createCLAHE(clipLimit=self.clahe_clip,tileGridSize=(16,16)).apply(
                np.asarray(photo.convert("L")))
            diagnostic=Image.fromarray(luminance).convert("RGB")
        inputs=self.processor(images=diagnostic,return_tensors="pt",device=self.device,
            limit_side_len=self.resolution,limit_type="max",max_side_limit=self.resolution).to(self.device)
        actual=tuple(inputs["pixel_values"].shape[-2:])
        expected=min(self.resolution,max(photo.size))
        if abs(max(actual)-expected)>32:
            raise ValueError("Native OCR resize parameters were not applied")
        if str(self.device).startswith("cuda"):torch.cuda.synchronize()
        prepared=time.monotonic()
        with torch.inference_mode():
            result=self.model(pixel_values=inputs["pixel_values"])
        if not torch.isfinite(result.last_hidden_state).all():
            raise ValueError("Nonfinite native OCR probability map")
        if str(self.device).startswith("cuda"):torch.cuda.synchronize()
        inferred=time.monotonic()
        candidates=self.processor.post_process_object_detection(
            SimpleNamespace(last_hidden_state=result.last_hidden_state.float()),
            target_sizes=inputs["target_sizes"],threshold=self.pixel_threshold,
            box_threshold=self.box_threshold,max_candidates=3000,unclip_ratio=1.4)[0]
        boxes=candidates["boxes"].detach().cpu().tolist()
        scores=candidates["scores"].detach().cpu().tolist()
        records=normalized_candidates(boxes,scores,photo.size)
        return {"method":"ppocrv6_small_native_candidates_v1","candidates":records,
                "resolution":self.resolution,"actual_input_hw":list(actual),
                "candidate_limit_reached":len(records)>=3000,
                "preprocess_seconds":prepared-began,"inference_seconds":inferred-prepared,
                "seconds":time.monotonic()-began,"weights_sha256":self.weights_sha256,
                "scope":"Text-like candidate positions only, never a digital-watermark decision"}
