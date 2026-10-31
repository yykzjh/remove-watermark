"""Bounded real-service smoke test: all fixtures and artifacts stay on Linux."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8765")
args = parser.parse_args()
state = request(args.url, "/health")
assert state["stage"] == "ready", state["stage"]
assert not state.get("job") or state["job"]["state"] not in {"scanning", "running", "paused", "cancelling"}
work = Path(".run").resolve()/f"mirror-smoke-{time.time_ns()}"
source, target = work/"input", work/"output"
leaf = source/"甲镇"/"甲村"/"一级"/"二级"/"三级"
other = source/"乙乡"/"乙村"/"分组"
(leaf/"空目录").mkdir(parents=True)
other.mkdir(parents=True)
(leaf/"说明.txt").write_text("完整保留非图像内容", encoding="utf-8")
(leaf/"data.bin").write_bytes(bytes(range(256))*128)
(leaf/"无扩展名").write_bytes(b"opaque payload")
(leaf/"坏图.jpg").write_bytes(b"")
(leaf/".hidden").write_bytes(b"hidden")
(leaf/"相对链接").symlink_to("说明.txt")
(leaf/"目录链接").symlink_to("空目录", target_is_directory=True)
image = Image.new("RGB", (173, 127), (60, 90, 70))
exif = image.getexif(); exif[274] = 6
image.save(leaf/"照片.JPG", exif=exif)
Image.new("RGB", (91, 69), (80, 100, 120)).save(other/"照片.JPG")
Image.new("RGBA", (79, 53), (30, 70, 50, 180)).save(other/"透明.PNG")
Image.new("RGB", (9, 7)).save(leaf/"多页.tiff", save_all=True,
                                     append_images=[Image.new("RGB", (9, 7))])

def snapshot(root):
    directories, files, links = set(), {}, {}
    for folder, dirs, names in os.walk(root, followlinks=False):
        directories.add(Path(folder).relative_to(root).as_posix())
        for name in dirs + names:
            path = Path(folder)/name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                links[rel] = os.readlink(path)
            elif path.is_file():
                files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return directories, files, links

before = snapshot(source)
start = time.monotonic()
job_id = request(args.url, "/jobs", {"input_dir":str(source), "output_dir":str(target),
    "options":{"concurrency":3, "preserve_background":True}})["id"]
while True:
    job = request(args.url, "/health")["job"]
    assert job["id"] == job_id
    if job["state"] in {"completed", "failed", "cancelled"}:
        break
    if time.monotonic()-start > 240:
        raise TimeoutError(job_id)
    time.sleep(.25)
after = snapshot(target)
assert job["state"] == "completed", job
assert before == snapshot(source), "Input changed"
assert before[0] == after[0], "Directory hierarchy changed"
assert before[1].keys() == after[1].keys(), "File names changed or files lost"
assert before[2] == after[2], "Links changed"
images = {p.relative_to(source).as_posix() for p in (leaf/"照片.JPG",other/"照片.JPG",other/"透明.PNG")}
for rel, digest in before[1].items():
    if rel not in images:
        assert after[1][rel] == digest, rel
    else:
        with Image.open(source/rel) as original, Image.open(target/rel) as output:
            assert original.size == output.size, rel
            assert original.getexif().get(274, 1) == output.getexif().get(274, 1), rel
result = {"revision":subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
          "job":job_id, "state":job["state"], "seconds":round(time.monotonic()-start,3),
          "files":len(before[1]), "directories":len(before[0]), "links":len(before[2]),
          "processed_images":job["succeeded"], "invalid":job["invalid"], "mirror":job["mirror"],
          "source_unchanged":True,"all_relative_paths_equal":True,
          "auxiliary_sha256_equal":True,"image_dimensions_equal":True,"exif_orientation_equal":True}
Path(".run/mirror-smoke-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False),flush=True)
