"""CPU-only read-only OCR witnesses. Readable unchanged text is not physical-scene proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
import time
import unicodedata
from pathlib import Path

import numpy as np


def transcript_key(text):
    if not isinstance(text,str):raise TypeError("OCR transcription must be a string")
    normalized=unicodedata.normalize("NFKC",text).strip().casefold()
    if not normalized:raise ValueError("Empty OCR transcription")
    return hashlib.sha256(normalized.encode()).hexdigest()


def app_marker(text):
    value=unicodedata.normalize("NFKC",text).casefold()
    # A denial-only filter: these words never prove that other lettering is physical.
    return bool(re.search("水印|相机|防伪|拍摄|camera|timestamp|watermark|20[0-9]{2}[./年-][0-9]",value))


def scan(engine,photo):
    rows,_=engine(np.asarray(photo.convert("RGB"))[...,::-1].copy(),use_det=True,use_cls=False,use_rec=True)
    records=[]
    if rows is None:return records
    if len(rows)>512:raise ValueError("Unbounded OCR witness count")
    w,h=photo.size
    for quad,text,confidence in rows:
        points=np.asarray(quad,dtype=float);score=float(confidence)
        if (points.shape!=(4,2) or not np.isfinite(points).all() or not math.isfinite(score)
                or not 0<=score<=1 or np.any(points<0) or points[:,0].max()>w or points[:,1].max()>h):
            raise ValueError("Invalid OCR witness")
        low,high=points.min(axis=0),points.max(axis=0)
        if np.any(high<=low):raise ValueError("Degenerate OCR witness")
        records.append({"box":[float(low[0]/w),float(low[1]/h),float(high[0]/w),float(high[1]/h)],
            "quad":[[float(x/w),float(y/h)] for x,y in points],"confidence":score,
            "transcript_sha256":transcript_key(text),"characters":len(text.strip()),
            "app_marker":app_marker(text)})
    return records


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--max-edge",type=int,choices=[2048,3072,4096],default=4096)
    a=p.parse_args()
    assert not a.target.exists()
    from rapidocr_onnxruntime import RapidOCR

    from watermark_studio.cross_review import preserved_patch
    from watermark_studio.images import load_photo
    from watermark_studio.text_preservation import box_iou

    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["manifest_cases"](a.manifest);hashes=helper["file_hashes"](cases)
    engine=RapidOCR(det_use_cuda=False,rec_use_cuda=False,cls_use_cuda=False,
        intra_op_num_threads=1,inter_op_num_threads=1,det_limit_type="max",
        det_limit_side_len=a.max_edge,max_side_len=a.max_edge,det_box_thresh=.5,
        use_cls=False,use_rec=True)
    a.target.mkdir(parents=True);cache={};records=[];started=time.monotonic()
    def recognized(path,photo):
        with Path(path).open("rb") as file:sha=hashlib.file_digest(file,"sha256").hexdigest()
        if sha not in cache:cache[sha]=scan(engine,photo)
        return cache[sha]
    with (a.target/"results.jsonl").open("x") as log:
        for case in cases:
            before=load_photo(case["source"])[0];after=load_photo(case["output"])[0]
            if before.size!=after.size:raise ValueError("Native witnesses require original output size")
            source=recognized(case["source"],before);output=recognized(case["output"],after)
            witnesses=[];lost=[]
            for word in source:
                if word["confidence"]<.9 or word["characters"]<2 or word["app_marker"]:continue
                matches=[w for w in output if w["confidence"]>=.9 and
                    w["transcript_sha256"]==word["transcript_sha256"] and box_iou(w["box"],word["box"])>=.7]
                if matches:
                    pixel=preserved_patch(before,after,word["box"])
                    witnesses.append({"source":word,"output":max(matches,key=lambda w:box_iou(w["box"],word["box"])),
                                      "pixels":pixel})
                else:lost.append(word)
            record={"file":case["file"],"source_words":len(source),"output_words":len(output),
                    "same_readable_words":witnesses,"lost_source_word_candidates":lost,
                    "scope":"OCR and pixel witnesses only; no physical/digital classification or quality approval"}
            records.append(record);log.write(json.dumps(record)+chr(10));log.flush()
            if len(records)%16==0:
                print(json.dumps({"done":len(records),"unique_photos":len(cache),
                    "unchanged_readable_words":sum(w["pixels"]["unchanged"] for r in records for w in r["same_readable_words"])}),flush=True)
    assert helper["file_hashes"](cases)==hashes
    report={"state":"completed_read_only_witnesses","images":len(records),"seconds":time.monotonic()-started,
            "unique_photos":len(cache),"max_edge":a.max_edge,"minimum_recognition_confidence":.9,
            "source_bytes_unchanged":True,"production_adopted":False,
            "scope":"Readable unchanged text is not proof of physical printing; empty OCR is not absence proof"}
    (a.target/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
