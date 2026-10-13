from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import PRESETS, ProcessingConfig

# Animated/multipage images are deliberately excluded instead of silently losing frames.
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageTask:
    source: str
    destination: str
    relative: str


def validate_directories(input_dir: str, output_dir: str) -> tuple[Path, Path]:
    source, target = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("输入目录不存在")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("输入和输出目录必须互不包含，避免覆盖原图或重复遍历")
    return source, target


def scan_images(source: Path, target: Path, overwrite: bool, stop=None):
    tasks, skipped, invalid = [], 0, []
    target.mkdir(parents=True, exist_ok=True)

    def scan_error(error):
        invalid.append({"file": str(error.filename), "error": str(error)[:300]})

    for folder, dirs, names in os.walk(source, followlinks=False, onerror=scan_error):
        if stop and stop.is_set():
            break
        dirs[:] = sorted(d for d in dirs if not (Path(folder) / d).is_symlink())
        relative_folder = Path(folder).relative_to(source)
        destination_folder = target / relative_folder
        destination_folder.mkdir(parents=True, exist_ok=True)
        if destination_folder.resolve() != destination_folder:
            raise ValueError("输出目录包含符号链接，无法保证安全写入")
        for name in sorted(names):
            path = Path(folder) / name
            if path.is_symlink() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(source)
            destination = target / relative
            if destination.is_symlink():
                raise ValueError("输出文件不能是符号链接")
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            try:
                with Image.open(path) as image:
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("不支持多帧图片")
                    image.verify()
                tasks.append(ImageTask(str(path), str(destination), relative.as_posix()))
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                invalid.append({"file": relative.as_posix(), "error": str(error)[:300]})
    return tasks, skipped, invalid


def regions_for(image: Image.Image, options: ProcessingConfig):
    preset = options.preset
    if preset == "auto":
        # Template classifier only: it is NOT a general watermark detector.
        sample = np.asarray(image.resize((256, 192)).convert("RGB"), dtype=np.int16)
        corner = sample[135:185, :120]
        blue = (corner[..., 2] > corner[..., 0] + 35) & (corner[..., 1] > corner[..., 0] + 10)
        preset = "blue" if float(blue.mean()) > 0.06 else "gray"
    regions = options.regions if preset == "custom" else PRESETS[preset]
    return [r for r in regions if options.remove_logo or r.kind != "logo"]


def pixel_box(box, size):
    w, h = size
    return (
        max(0, math.floor(box[0] * w)),
        max(0, math.floor(box[1] * h)),
        min(w, math.ceil(box[2] * w)),
        min(h, math.ceil(box[3] * h)),
    )


def context_box(box, size, factor):
    x0, y0, x1, y1 = box
    dx, dy = round((x1 - x0) * factor), round((y1 - y0) * factor)
    return (max(0, x0 - dx), max(0, y0 - dy), min(size[0], x1 + dx), min(size[1], y1 + dy))


def blend_patch(original: Image.Image, patch: Image.Image, box, crop, feather: int):
    # Alpha is 1 throughout the specified watermark, fading OUTSIDE it.
    # A blurred mask that fades inside the watermark would leave text ghosts.
    x0, y0, x1, y1 = box
    cx0, cy0, cx1, cy1 = crop
    xx, yy = np.meshgrid(np.arange(cx0, cx1), np.arange(cy0, cy1))
    distance = np.maximum.reduce(
        [x0 - xx, xx - (x1 - 1), y0 - yy, yy - (y1 - 1), np.zeros_like(xx)]
    )
    alpha = (
        (distance == 0).astype(np.float32)
        if feather == 0
        else np.clip(1 - distance / feather, 0, 1)
    )
    alpha = alpha * alpha * (3 - 2 * alpha)
    mask = Image.fromarray(np.uint8(np.round(alpha * 255)))
    resized = patch.convert("RGB").resize((cx1 - cx0, cy1 - cy0), Image.Resampling.LANCZOS)
    result = original.copy()
    result.paste(resized, (cx0, cy0), mask)
    return result


def load_photo(path: str):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        exif = image.getexif()
        exif.pop(274, None)
        alpha = (
            image.convert("RGBA").getchannel("A")
            if ("A" in image.getbands() or "transparency" in image.info)
            else None
        )
        return image.convert("RGB"), exif.tobytes(), raw.info.get("icc_profile"), alpha


def atomic_save(image, destination, exif=b"", icc=None, overwrite=False):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fmt = Image.registered_extensions()[path.suffix.lower()]
    kwargs = {}
    if fmt == "JPEG":
        kwargs.update(quality=97, subsampling=0)
    if fmt in {"JPEG", "PNG", "WEBP", "TIFF"}:
        if exif:
            kwargs["exif"] = exif
        if icc:
            kwargs["icc_profile"] = icc
    try:
        image.save(temp, format=fmt, **kwargs)
        if overwrite:
            os.replace(temp, path)
        else:
            # Atomic no-overwrite even if another process creates the target meanwhile.
            os.link(temp, path)
            temp.unlink()
    finally:
        temp.unlink(missing_ok=True)
