import sys
from collections import deque
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
    ), object())
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
        options, DeploymentConfig(), object(),
    )
    assert not results[0]["ok"] and "invalid second crop" in results[0]["error"]
    assert not target.exists()
