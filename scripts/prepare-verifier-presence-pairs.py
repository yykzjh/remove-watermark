"""Freeze source-disjoint weak physical-print parents and exact synthetic overlay labels."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from watermark_studio.config import Region
from watermark_studio.quality import output_detail_views
from watermark_studio.quality_presence import parent_splits
from watermark_studio.training_features import digest, validate_pairs
from watermark_studio.training_pairs import render_profile_pair


def complete_records(path):
    # A collector may append while preparation reads. Only complete, flushed
    # JSONL records can enter a frozen training snapshot.
    return [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1] if line]


def area_name(box):
    x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
    column=0 if x<1/3 else 2 if x>2/3 else 1
    row=0 if y<1/3 else 2 if y>2/3 else 1
    return (("top_left","top","top_right"),("left","center","right"),
            ("bottom_left","bottom","bottom_right"))[row][column]


def injected(clean,style,rng,font):
    w,h=clean.size
    if style<2:
        marked,mask,metadata=render_profile_pair(clean,style,rng,font,profile="camera-v2")
    else:
        layer=Image.new("RGBA",clean.size)
        draw=ImageDraw.Draw(layer)
        if style<5:
            size=max(14,round(min(w,h)*rng.uniform(.025,.06)))
            face=ImageFont.truetype(str(font),size)
            content=(rng.choice(["现场记录 ","拍摄时间 ","施工现场 "])+str(rng.randint(2020,2026))+
                     "."+str(rng.randint(1,12)).zfill(2)+"."+str(rng.randint(1,28)).zfill(2))
            if style==4:
                content="".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(rng.randint(8,16)))
            bounds=draw.textbbox((0,0),content,font=face)
            if bounds[2]>w-12:
                face=ImageFont.truetype(str(font),max(10,round(size*(w-12)/bounds[2])))
                bounds=draw.textbbox((0,0),content,font=face)
            x=rng.randint(2,max(2,w-bounds[2]-3));y=rng.randint(2,max(2,h-(bounds[3]-bounds[1])-4))
            alpha=rng.choice([24,32,48,64,96]) if style in {2,4} else rng.choice([120,180,255])
            color=rng.choice([(255,255,255),(0,0,0),(255,230,30),(240,30,30)])
            draw.text((x,y-bounds[1]),content,font=face,fill=(*color,alpha))
            metadata={"kind":"scattered_faint_text","alpha":alpha,"font_size":face.size}
        else:
            bw,bh=round(w*rng.uniform(.22,.6)),round(h*rng.uniform(.10,.32))
            x,y=rng.randint(0,w-bw-1),rng.randint(0,h-bh-1)
            alpha=rng.choice([32,48,64,96,160])
            color=rng.choice([(255,255,255),(0,0,0),(30,115,230),(240,230,30)])
            draw.rounded_rectangle((x,y,x+bw,y+bh),radius=rng.randint(0,12),fill=(*color,alpha))
            if style==6:
                layer=layer.filter(ImageFilter.GaussianBlur(1.2))
            metadata={"kind":"unremoved_panel_backing","alpha":alpha}
        marked=Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB")
        mask=layer.getchannel("A")
    yy,xx=np.where(np.asarray(mask)>0)
    if not len(xx):
        raise ValueError("Synthetic overlay must have measurable support")
    box=[float(xx.min()/w),float(yy.min()/h),float((xx.max()+1)/w),float((yy.max()+1)/h)]
    return marked,box,metadata


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--physical-banks",type=Path,nargs="+",required=True)
    p.add_argument("--textfree-manifest",type=Path,required=True)
    p.add_argument("--forbidden",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--physical-parents",type=int,default=32)
    p.add_argument("--textfree-parents",type=int,default=64)
    p.add_argument("--seed",type=int,default=2026110377)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    a=p.parse_args()
    assert not a.target.exists() and 24<=a.physical_parents<=96 and 24<=a.textfree_parents<=120
    forbidden=set(json.loads(a.forbidden.read_text()))
    assert all(len(sha)==64 for sha in forbidden)
    physical={}
    screening=[]
    for bank in a.physical_banks:
        records=complete_records(bank/"screening.jsonl")
        screening.append({"path":str(bank),"completed_records":len(records),"collector_completed":(bank/"manifest.json").exists()})
        for record in records:
            if not record.get("accepted") or record["source_sha256"] in forbidden or record["crop_sha256"] in forbidden:
                continue
            assert digest(record["source"])==record["source_sha256"]
            assert digest(record["crop_path"])==record["crop_sha256"]
            physical.setdefault(record["source_sha256"],{
                "kind":"physical","source_original":record["source"],"source_original_sha256":record["source_sha256"],
                "clean":record["crop_path"],"clean_sha256":record["crop_sha256"],"screening":record})
    parent,_=validate_pairs(a.textfree_manifest)
    textfree={}
    for row in parent["pairs"]:
        sha=row["source_original_sha256"]
        if sha not in forbidden and sha not in physical and row["target_sha256"] not in forbidden:
            assert digest(row["target"])==row["target_sha256"]
            textfree.setdefault(sha,{"kind":"text_free","source_original":row["source_original"],
                "source_original_sha256":sha,"clean":row["target"],"clean_sha256":row["target_sha256"]})
    assert len(physical)>=a.physical_parents and len(textfree)>=a.textfree_parents, (len(physical),len(textfree))
    order=lambda row:hashlib.sha256((str(a.seed)+row["source_original_sha256"]).encode()).hexdigest()
    parents=sorted(physical.values(),key=order)[:a.physical_parents]+sorted(textfree.values(),key=order)[:a.textfree_parents]
    splits=parent_splits(parents,a.seed)
    a.target.mkdir(parents=True);(a.target/"clean").mkdir();(a.target/"overlays").mkdir()
    forbidden_snapshot=a.target/"forbidden-sha256.json"
    forbidden_snapshot.write_text(json.dumps(sorted(forbidden)))
    rows=[]
    for index,record in enumerate(parents):
        sha=record["source_original_sha256"]
        with Image.open(record["clean"]) as raw:
            clean=raw.convert("RGB");clean.thumbnail((1280,1280),Image.Resampling.LANCZOS)
        cp=a.target/"clean"/(sha[:24]+".png");clean.save(cp);clean_sha=digest(cp)
        for style in range(7):
            rng=random.Random(a.seed+int(sha[:12],16)+style)
            marked,box,render=injected(clean,style,rng,a.font)
            name=sha[:24]+"-"+str(style)
            bp=a.target/"overlays"/(name+".png");marked.save(bp);before_sha=digest(bp)
            # Shared geometry for the paired clean/marked results prevents
            # position or detail-count cues from disclosing the target label.
            regions=[] if style==4 else [{"kind":"panel","box":box}]
            views,view_boxes=output_detail_views(marked,marked,4,[Region.model_validate(r) for r in regions])
            layout=[{"box":b,"max_edge":max(v.size)} for b,v in zip(view_boxes,views)]
            common={"detail_layout":layout,"source_original_sha256":sha,"parent_kind":record["kind"],"split":splits[sha],
                "before":str(bp.resolve()),"before_sha256":before_sha,"regions":regions,"render":render}
            rows.append(common|{"id":name+"-clean","after":str(cp.resolve()),"after_sha256":clean_sha,
                                "overlay_present":False,"uncertain":False,"area":"none"})
            rows.append(common|{"id":name+"-marked","after":str(bp.resolve()),"after_sha256":before_sha,
                                "overlay_present":True,"uncertain":False,"area":area_name(box)})
        # Deliberately insufficient photographic detail must teach abstention,
        # not a confident pass. This branch is never a native-photo truth label.
        uncertain=clean.resize((32,32),Image.Resampling.BILINEAR).resize(clean.size,Image.Resampling.BILINEAR)
        up=a.target/"overlays"/(sha[:24]+"-uncertain.png");uncertain.save(up)
        rows.append({"id":sha[:24]+"-uncertain","source_original_sha256":sha,"parent_kind":record["kind"],
            "split":splits[sha],"before":str(up.resolve()),"before_sha256":digest(up),"after":str(up.resolve()),
            "after_sha256":digest(up),"regions":[],"overlay_present":False,"uncertain":True,"area":"none"})
        print(json.dumps({"parents":index+1,"total":len(parents),"rows":len(rows)}),flush=True)
    assert len({r["id"] for r in rows})==len(rows)
    assert not set(splits)&forbidden
    report={"version":1,"state":"completed","revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "task":"output_only_digital_overlay_presence","parents":parents,"rows":rows,"seed":a.seed,
            "forbidden_manifest_sha256":digest(forbidden_snapshot),"forbidden_manifest":str(forbidden_snapshot.resolve()),"forbidden_overlap":0,"font_sha256":digest(a.font),
            "screening_snapshots":screening,"parent_counts":dict(collections.Counter(splits.values())),
            "row_counts":dict(collections.Counter(r["split"] for r in rows)),
            "label_provenance":"Known synthetic additions paired on source-disjoint weak physical-print and source-screened text-free parents. No generated-photo QA verdict is a training label.",
            "training_control_inputs_used":False}
    path=a.target/"manifest.json";path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"manifest":str(path),"parents":len(parents),"rows":len(rows),
                      "splits":report["row_counts"],"forbidden_overlap":0}),flush=True)


if __name__=="__main__":
    main()
