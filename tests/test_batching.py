import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from watermark_studio.batching import override_methods
from watermark_studio.config import DeploymentConfig, ProcessingConfig, Region
from watermark_studio.runtime import Runtime
from watermark_studio.worker import process_batch


def test_scoped_override_restores_on_exception():
    class Pipe:
        def encode(self):
            return "original"
    pipe = Pipe()
    with pytest.raises(RuntimeError), override_methods(pipe, encode=lambda: "temporary"):
        assert pipe.encode() == "temporary"
        raise RuntimeError("inference failed")
    assert pipe.encode() == "original" and "encode" not in pipe.__dict__


@pytest.mark.parametrize("concurrency", [1, 3, 8, 16, 31])
def test_scheduler_counts_images_not_workers(tmp_path, concurrency):
    config = DeploymentConfig(backend="mock", state_dir=str(tmp_path), gpus=[0,1,2,3],
                              batch_size=2, batch_window=4)
    runtime = Runtime(config)
    class Inbox:
        def put(self, value):
            self.value = value
    runtime.workers = [{"state":"idle", "task":None, "queue":Inbox()} for _ in range(4)]
    runtime.stage = "ready"
    runtime.pending = deque({"relative":str(i)} for i in range(50))
    runtime.job = {"id":"test","state":"running","options":{"concurrency":concurrency}}
    runtime._dispatch()
    assigned = [w["task"] for w in runtime.workers if w["task"]]
    assert sum(map(len, assigned)) == concurrency
    assert all(len(tasks) <= 8 for tasks in assigned)
    assert len({t["relative"] for tasks in assigned for t in tasks}) == concurrency


@pytest.mark.parametrize("split", [False, True])
def test_batch_oom_isolation_and_original_dimensions(tmp_path, monkeypatch, split):
    class OutOfMemoryError(RuntimeError):
        pass
    cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 100,
        empty_cache=lambda: None, OutOfMemoryError=OutOfMemoryError,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr("watermark_studio.batching.dimensions", lambda image, resolution: (64,64))
    calls = []
    def inference(pipe, items, resolution, kv):
        calls.append(len(items))
        if len(items) > 1:
            raise OutOfMemoryError("test OOM")
        return [Image.new("RGB", (64,64), (250, 20, 30))]
    monkeypatch.setattr("watermark_studio.batching.independent_batch", inference)
    tasks = []
    for i, size in enumerate([(173,129), (251,187), (173,129)]):
        source, dest = tmp_path/f"in-{i}.png", tmp_path/f"out-{i}.png"
        Image.new("RGBA", size, (30,60,90,170)).save(source)
        tasks.append({"source":str(source),"destination":str(dest),"relative":str(i)})
    options = ProcessingConfig(preset="custom", regions=[Region(box=(.2,.2,.5,.5))])
    results = process_batch(tasks, options, DeploymentConfig(
        batch_size=2, batch_bucketing=True, oom_split_batch=split,
    ), SimpleNamespace(_execution_device="cuda:0"))
    assert calls == ([2,1,1,1] if split else [2,1])
    assert sum(result["ok"] for result in results) == (3 if split else 1)
    for task, result in zip(tasks, results):
        if result["ok"]:
            with Image.open(task["source"]) as original, Image.open(task["destination"]) as output:
                assert original.size == output.size
                assert np.array_equal(np.asarray(original)[:,:,3], np.asarray(output)[:,:,3])
                assert original.getpixel((0,0)) == output.getpixel((0,0))

def test_preparation_failure_never_writes_partial_repair(tmp_path, monkeypatch):
    cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    calls = []
    def dimensions(image, resolution):
        calls.append(image.size)
        if len(calls) == 2:
            raise ValueError("invalid second crop")
        return (64,64)
    monkeypatch.setattr("watermark_studio.batching.dimensions", dimensions)
    source, target = tmp_path/"source.png", tmp_path/"output.png"
    Image.new("RGB",(200,150)).save(source)
    options = ProcessingConfig(
        preset="custom", regions=[Region(box=(.1,.1,.2,.2)),Region(box=(.7,.7,.9,.9))],
    )
    results = process_batch(
        [{"source":str(source),"destination":str(target),"relative":"source.png"}],
        options, DeploymentConfig(), SimpleNamespace(_execution_device="cuda:0"),
    )
    assert not results[0]["ok"] and "invalid second crop" in results[0]["error"]
    assert not target.exists()


@pytest.mark.parametrize("small_oom", [False, True])
def test_small_input_batch_cap_never_increases_large_input_batch_or_leaks_images(tmp_path, monkeypatch, small_oom):
    class OutOfMemoryError(RuntimeError): pass
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
        empty_cache=lambda: None, OutOfMemoryError=OutOfMemoryError)))
    regions = [Region(kind="panel", box=(0,.6,.4,1)),
               Region(kind="logo", box=(.85,.95,1,1))]
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",
                        lambda images, *a: [(regions, '{"scene":"outdoor"}') for _ in images])
    monkeypatch.setattr("watermark_studio.batching.dimensions", lambda image, resolution: (resolution,resolution))
    calls = []
    def inference(pipe, items, resolution, kv):
        calls.append((resolution, len(items)))
        assert len({item["seed"] for item in items}) == len(items)
        if resolution == 512 and len(items) > 1 and small_oom:
            raise OutOfMemoryError("small batch also respects OOM isolation")
        return [Image.new("RGB", (32,32), "gray") for item in items]
    monkeypatch.setattr("watermark_studio.batching.independent_batch", inference)
    tasks = []
    for index in range(2):
        source, target = tmp_path/f"in-{index}.png", tmp_path/f"out-{index}.png"
        Image.new("RGB", (800,600), (30+index*40,60,90)).save(source)
        tasks.append({"source": str(source), "destination": str(target), "relative": str(index)})
    originals = [Path(t["source"]).read_bytes() for t in tasks]
    pipe = SimpleNamespace(text_encoder=None, processor=None, _execution_device="cpu")
    results = process_batch(tasks, ProcessingConfig(scope="full_image", preserve_background=True,
        first_pass_mode="focused", resolution=1024), DeploymentConfig(
            batch_size=1, small_batch_size=2, batch_bucketing=True, oom_split_batch=True), pipe)
    assert [n for budget,n in calls if budget == 1024] == [1,1]
    assert [n for budget,n in calls if budget == 512] == ([2,1,1] if small_oom else [2])
    assert all(r["ok"] for r in results)
    for i, task in enumerate(tasks):
        assert Path(task["source"]).read_bytes() == originals[i]
        with Image.open(task["destination"]) as output, Image.open(task["source"]) as source:
            assert output.size == source.size
            assert output.getpixel((450,50)) == source.getpixel((450,50))


def test_window_prefill_keeps_independent_images_and_prompts_with_diffusion_b1(monkeypatch):
    from contextlib import nullcontext

    from watermark_studio.batching import preencode_window
    class OOM(RuntimeError):
        pass
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(inference_mode=nullcontext,
        cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))
    pipe=SimpleNamespace(image_processor=SimpleNamespace(resize=lambda image,width,height:image.resize((width,height))))
    items=[{"image":Image.new("RGB",(400,300),"red"),"size":(128,96),"prompt":"first"},
           {"image":Image.new("RGB",(300,400),"blue"),"size":(96,128),"prompt":"second"}]
    calls=[]
    def encode(pipe,prepared):
        calls.append([(x["prompt"],x["image"].size) for x in prepared])
        return [(x["prompt"],x["image"].getpixel((0,0)),x["image"].size) for x in prepared]
    stats=preencode_window(pipe,items,4,encode)
    assert calls==[[("first",(128,96)),("second",(96,128))]]
    assert stats["cached_images"]==2 and stats["fallback_images"]==0
    assert items[0]["_encoded_prompt"][0]=="first" and items[0]["_encoded_prompt"][1][:3]==(255,0,0)
    assert items[1]["_encoded_prompt"][0]=="second" and items[1]["_encoded_prompt"][1][:3]==(0,0,255)
    assert [x["image"].size for x in items]==[(400,300),(300,400)]
    assert [x["_prefill_batch_size"] for x in items]==[2,2]


def test_window_prefill_oom_splits_and_single_failure_falls_back_without_cross_image_cache(monkeypatch):
    from contextlib import nullcontext

    from watermark_studio.batching import preencode_window
    class OOM(RuntimeError):
        pass
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(inference_mode=nullcontext,
        cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))
    pipe=SimpleNamespace(image_processor=SimpleNamespace(resize=lambda image,**kwargs:image.copy()))
    items=[{"image":Image.new("RGB",(32,32)),"size":(32,32),"prompt":text} for text in ("valid","bad")]
    def encode(pipe,prepared):
        if len(prepared)>1:
            raise OOM("split")
        if prepared[0]["prompt"]=="bad":
            raise ValueError("bad encoding")
        return [("own-vector","own-mask","own-layout")]
    stats=preencode_window(pipe,items,2,encode)
    assert stats=={"cached_images":1,"oom_splits":1,"fallback_images":1}
    assert items[0]["_encoded_prompt"][0]=="own-vector"
    assert "_encoded_prompt" not in items[1]


def test_original_semantics_preserve_independent_full_photos_without_changing_vae_images(monkeypatch):
    from contextlib import nullcontext

    from watermark_studio.batching import preencode_window, semantic_reference

    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(inference_mode=nullcontext,
        cuda=SimpleNamespace(empty_cache=lambda:None)))
    pipe=SimpleNamespace(image_processor=SimpleNamespace(
        resize=lambda image,width,height:image.resize((width,height))))
    items=[{"image":Image.new("RGB",(320,240),"gray"),
            "_semantic_reference":Image.new("RGB",(320,240),color),
            "size":(160,120),"prompt":str(i)} for i,color in enumerate(["red","blue"])]
    def encode(pipe,prepared):
        return [(p["image"].getpixel((0,0)),p["prompt"],p["image"].size) for p in prepared]
    stats=preencode_window(pipe,items,2,encode)
    assert stats["cached_images"]==2
    assert items[0]["_encoded_prompt"][0][:3]==(255,0,0)
    assert items[1]["_encoded_prompt"][0][:3]==(0,0,255)
    assert all(item["image"].getpixel((0,0))==(128,128,128) for item in items)
    with pytest.raises(ValueError,match="complete conditioning frame"):
        semantic_reference({"image":Image.new("RGB",(320,240)),
                            "_semantic_reference":Image.new("RGB",(160,120))})
