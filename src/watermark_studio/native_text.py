"""Experimental native-resolution OCR candidates; no digital/physical or quality verdict."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

OFFICIAL_SMALL_SHA256="89a96a8adc4e9cd0c994098edc76022e496d35844392562b4694c8fbc583f2da"
OFFICIAL_SERVER_DET_SHA256="06e7a44aa5c4146531e88703140e0f7329910b92f84f8bf2756bfd37cec5cb0a"
OFFICIAL_SERVER_REC_SHA256="da94c5dd42c88b00c44081942b157bee8eda9ccd0481dbd0af3824f6c241aa91"
DETECTOR_ARCHITECTURES={"pp_ocrv6_small_det","pp_ocrv5_server_det"}


def normalize_recognitions(rows,count):
    if not isinstance(rows,list) or len(rows)!=count:
        raise ValueError("Native OCR recognition count mismatch")
    records=[]
    for row in rows:
        text=row.get("text")
        if not isinstance(text,str) or len(text)>512:
            raise ValueError("Invalid bounded OCR transcription")
        score=row.get("score")
        if isinstance(score,bool) or not isinstance(score,(int,float,np.integer,np.floating)):
            raise TypeError("Invalid OCR recognition confidence")
        score=float(score)
        if not text.strip() and np.isnan(score):
            # The official CTC decoder averages an empty character selection.
            # Preserve an unscored blank; it can never certify text absence.
            records.append({"text":text,"confidence":None,"status":"blank_ctc"})
        elif not np.isfinite(score) or not 0<=score<=1:
            raise ValueError("Nonfinite or out-of-range OCR recognition confidence")
        else:
            records.append({"text":text,"confidence":score,
                            "status":"recognized" if text.strip() else "blank_ctc"})
    return records



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
                 pixel_threshold=.1,clahe_clip=0,contrast_space="gray",weights_sha256=OFFICIAL_SMALL_SHA256):
        import torch
        from transformers import (
            AutoImageProcessor,
            PPOCRV5ServerDetForObjectDetection,
            PPOCRV6SmallDetForObjectDetection,
        )

        path=Path(model_path)
        architecture=json.loads((path/"config.json").read_text()).get("model_type")
        if (hashlib.sha256((path/"model.safetensors").read_bytes()).hexdigest()!=weights_sha256 or
                architecture not in DETECTOR_ARCHITECTURES):
            raise ValueError("Pinned native OCR architecture/weight mismatch")
        factories={"pp_ocrv6_small_det":PPOCRV6SmallDetForObjectDetection,
                   "pp_ocrv5_server_det":PPOCRV5ServerDetForObjectDetection}
        self.architecture=architecture
        if resolution not in {2048,3072,4096} or not .05<=pixel_threshold<=.5 or not .1<=box_threshold<=.7:
            raise ValueError("Invalid bounded OCR candidate configuration")
        if not 0<=clahe_clip<=8 or contrast_space not in {"gray","lab"}:
            raise ValueError("Invalid diagnostic contrast")
        self.device=device;self.resolution=resolution
        self.box_threshold=box_threshold;self.pixel_threshold=pixel_threshold;self.clahe_clip=clahe_clip
        self.contrast_space=contrast_space
        self.weights_sha256=weights_sha256
        self.model,info=factories[architecture].from_pretrained(
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
            clahe=cv2.createCLAHE(clipLimit=self.clahe_clip,tileGridSize=(16,16))
            if self.contrast_space=="lab":
                lab=cv2.cvtColor(np.asarray(diagnostic),cv2.COLOR_RGB2LAB)
                lab[...,0]=clahe.apply(lab[...,0])
                diagnostic=Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
            else:
                diagnostic=Image.fromarray(clahe.apply(np.asarray(photo.convert("L")))).convert("RGB")
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
        import cv2
        bitmap=(result.last_hidden_state[0,0].detach().cpu().numpy()>self.pixel_threshold).astype(np.uint8)
        raw_contours=len(cv2.findContours(bitmap,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0])
        candidates=self.processor.post_process_object_detection(
            SimpleNamespace(last_hidden_state=result.last_hidden_state.float()),
            target_sizes=inputs["target_sizes"],threshold=self.pixel_threshold,
            box_threshold=self.box_threshold,max_candidates=3000,unclip_ratio=1.4)[0]
        boxes=candidates["boxes"].detach().cpu().tolist()
        scores=candidates["scores"].detach().cpu().tolist()
        records=normalized_candidates(boxes,scores,photo.size)
        return {"method":("ppocrv6_small_native_candidates_v1" if self.architecture=="pp_ocrv6_small_det" else "ppocrv5_server_native_candidates_v1"),
                "architecture":self.architecture,"candidates":records,
                "resolution":self.resolution,"actual_input_hw":list(actual),
                "candidate_limit_reached":raw_contours>3000 or len(records)>=3000,
                "raw_candidate_contours":raw_contours,
                "preprocess_seconds":prepared-began,"inference_seconds":inferred-prepared,
                "seconds":time.monotonic()-began,"weights_sha256":self.weights_sha256,
                "scope":"Text-like candidate positions only, never a digital-watermark decision"}


class NativeTextReader:
    """SHA-pinned GPU textline recognition; empty OCR never supplies QA approval."""

    def __init__(self,model_path,device="cuda:0",*,weights_sha256=OFFICIAL_SERVER_REC_SHA256):
        import torch
        from transformers import AutoImageProcessor, AutoModelForTextRecognition

        path=Path(model_path)
        if (hashlib.sha256((path/"model.safetensors").read_bytes()).hexdigest()!=weights_sha256 or
            json.loads((path/"config.json").read_text()).get("model_type")!="pp_ocrv5_server_rec"):
            raise ValueError("Pinned native text reader architecture/weight mismatch")
        self.model,info=AutoModelForTextRecognition.from_pretrained(path,local_files_only=True,
            use_safetensors=True,dtype=torch.float32,output_loading_info=True)
        if any(not key.endswith("num_batches_tracked") for key in info["missing_keys"]) or info["unexpected_keys"]:
            raise ValueError("Native text reader checkpoint was not fully initialized")
        self.device=device;self.weights_sha256=weights_sha256
        self.model=self.model.to(device).eval().requires_grad_(False)
        self.processor=AutoImageProcessor.from_pretrained(path,local_files_only=True)

    def read(self,photos):
        import torch
        from PIL import Image

        if not 1<=len(photos)<=32 or any(not isinstance(p,Image.Image) or min(p.size)<2 for p in photos):
            raise ValueError("Bounded real textline crops required")
        if self.model.training:
            raise ValueError("Text recognition requires evaluation mode")
        if str(self.device).startswith("cuda"):torch.cuda.synchronize()
        began=time.monotonic()
        inputs=self.processor(images=[p.convert("RGB") for p in photos],return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output=self.model(pixel_values=inputs["pixel_values"])
        if not torch.isfinite(output.last_hidden_state).all():
            raise ValueError("Nonfinite native text recognition map")
        recognized=normalize_recognitions(self.processor.post_process_text_recognition(output),len(photos))
        if str(self.device).startswith("cuda"):torch.cuda.synchronize()
        return {"method":"ppocrv5_server_textline_recognition_v1","results":recognized,
                "seconds":time.monotonic()-began,"weights_sha256":self.weights_sha256,
                "not_a_watermark_absence_proof":True,
                "scope":"Untrusted text and character confidence only; no digital/physical or image-quality decision"}
