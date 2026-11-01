import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest
from PIL import Image


def fixture(tmp_path):
    home=tmp_path/"home";state=home/".local/state/watermark-studio";source=tmp_path/"input"
    source.mkdir();jobs=[]
    for i in range(2):
        name=f"village/{i}.png";path=source/name;path.parent.mkdir(exist_ok=True)
        Image.new("RGB",(12,8),(20+i,30,40)).save(path)
        job=f"{i+1:032x}";jobs.append(job);folder=state/job;folder.mkdir(parents=True)
        output=tmp_path/f"output{i}";target=output/name;target.parent.mkdir(parents=True)
        Image.new("RGB",(12,8),(40,50,60)).save(target)
        summary={"job":job,"state":"completed","quality_evaluated":False,"total":1,
            "input_dir":str(source),"output_dir":str(output),"source_job":"frozen-development",
            "code_revision":"test","options":{"output_width":0},
            "controlnet":{"sha256":"fixed","tensors":180,"layers":[0,2],"channels":129,"kv_cache":False,"seconds":i+1},
            "controlnet_settings":{"scale":1.,"resolution":1024,"kv_cache":False},
            "server_isolation":{"gpus":[1],"code_revision":"test"},
            "probe_shard":{"count":2,"index":i,"all_files":["village/0.png","village/1.png"]},
            "seconds":10+i,"sources_sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest()}}
        (folder/"summary.json").write_text(json.dumps(summary))
        (folder/"results.jsonl").write_text(json.dumps({"file":name,"ok":True,"quality_attempts":1,
            "model_inferences":1,"output_size":[12,8],"controlnet_probe":{"denoising_steps":8},"harmonization":{}})+"\n")
    return home,state,source,jobs


def run(monkeypatch,home,tmp_path,jobs):
    script=Path(__file__).resolve().parents[1]/"scripts/collect-controlnet-probe.py"
    monkeypatch.setattr(Path,"home",lambda:home)
    monkeypatch.setattr(sys,"argv",[str(script),"--jobs",*jobs,"--tag","collection","--work-root",str(tmp_path)])
    runpy.run_path(str(script),run_name="__main__")


def test_collects_each_original_once_without_new_generations_or_image_conversion(tmp_path,monkeypatch):
    home,state,source,jobs=fixture(tmp_path)
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.png")}
    run(monkeypatch,home,tmp_path,jobs)
    result=json.loads((tmp_path/"collection/summary.json").read_text())
    rows=list(map(json.loads,(state/result["job"]/"results.jsonl").read_text().splitlines()))
    assert result["total"]==2 and result["quality_evaluated"] is False
    assert {r["file"] for r in rows}=={"village/0.png","village/1.png"}
    assert (tmp_path/"collection/output/village/0.png").stat().st_ino==(tmp_path/"output0/village/0.png").stat().st_ino
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


@pytest.mark.parametrize("failure",["settings","checksum","dimensions","repeated_generation"])
def test_incompatible_or_changed_shards_cannot_create_an_approval_artifact(tmp_path,monkeypatch,failure):
    home,state,source,jobs=fixture(tmp_path);folder=state/jobs[1]
    if failure=="settings":
        data=json.loads((folder/"summary.json").read_text());data["controlnet_settings"]["scale"]=.5
        (folder/"summary.json").write_text(json.dumps(data))
    elif failure=="checksum":
        Image.new("RGB",(12,8),"black").save(source/"village/1.png")
    elif failure=="dimensions":
        Image.new("RGB",(8,8),"black").save(tmp_path/"output1/village/1.png")
    else:
        data=json.loads((folder/"results.jsonl").read_text());data["model_inferences"]=2
        (folder/"results.jsonl").write_text(json.dumps(data)+"\n")
    with pytest.raises(ValueError):
        run(monkeypatch,home,tmp_path,jobs)
    assert not (tmp_path/"collection").exists()
