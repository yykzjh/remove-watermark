from __future__ import annotations

import json
import math
import os
import shutil
import time
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


def scan_images(source: Path, target: Path, overwrite: bool, stop=None, workers=1,
                *, copy_overwrite=None, mirror_stats=None, progress=None):
    """Mirror the complete tree; queue only supported, valid single-frame images.

    Non-images and unsupported/invalid images are copied byte-for-byte. Symlinks
    are preserved as links, never followed. Copy failures remain explicit failures.
    Image admission and auxiliary copying use separate overwrite policies because
    quality-only jobs may inspect existing images without replacing other files.
    """
    tasks, skipped, invalid, candidates = [], 0, [], []
    stats = mirror_stats if mirror_stats is not None else {}
    stats.update(directories=0, copied_files=0, copied_bytes=0, copied_invalid_images=0,
                 copied_links=0, existing_files=0, copy_failures=0, errors=[])
    copy_overwrite = overwrite if copy_overwrite is None else copy_overwrite
    last_progress = 0
    target.mkdir(parents=True, exist_ok=True)

    def report(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if progress and (force or now-last_progress >= .25):
            progress({key: value for key, value in stats.items() if key != "errors"})
            last_progress = now

    def copy_error(path, error):
        stats["copy_failures"] += 1
        stats["errors"].append({"file": str(path), "error": str(error)[:300]})
        report()

    def preserve(path, invalid_image=False):
        if stop and stop.is_set():
            return
        relative = path.relative_to(source)
        destination = target / relative
        try:
            is_link = path.is_symlink()
            if destination.is_symlink() and not is_link:
                raise ValueError("输出文件不能是符号链接")
            if destination.is_dir() and not destination.is_symlink():
                raise IsADirectoryError(f"输出同名路径是目录: {relative}")
            if (destination.exists() or destination.is_symlink()) and not copy_overwrite:
                stats["existing_files"] += 1
                report()
                return
            if is_link:
                atomic_symlink(path, destination, copy_overwrite)
                stats["copied_links"] += 1
            elif path.is_file():
                atomic_copy(path, destination, copy_overwrite, preserve_metadata=True)
                stats["copied_files"] += 1
                stats["copied_bytes"] += destination.stat().st_size
                stats["copied_invalid_images"] += int(invalid_image)
            else:
                raise ValueError("不支持复制此特殊文件类型")
        except (OSError, ValueError) as error:
            copy_error(relative, error)
        report()

    def scan_error(error):
        invalid.append({"file": str(error.filename), "error": str(error)[:300]})
        copy_error(error.filename, error)

    for folder, dirs, names in os.walk(source, followlinks=False, onerror=scan_error):
        if stop and stop.is_set():
            break
        relative_folder = Path(folder).relative_to(source)
        destination_folder = target / relative_folder
        if destination_folder.resolve() != destination_folder:
            raise ValueError("输出目录包含符号链接，无法保证安全写入")
        destination_folder.mkdir(parents=True, exist_ok=True)
        stats["directories"] += 1
        for name in sorted(dirs):
            path = Path(folder) / name
            if path.is_symlink():
                preserve(path)
        dirs[:] = sorted(d for d in dirs if not (Path(folder) / d).is_symlink())
        for name in sorted(names):
            if stop and stop.is_set():
                break
            path = Path(folder) / name
            if path.is_symlink() or path.suffix.lower() not in EXTENSIONS:
                preserve(path)
                continue
            relative = path.relative_to(source)
            destination = target / relative
            if destination.is_symlink():
                raise ValueError("输出文件不能是符号链接")
            candidates.append(ImageTask(str(path), str(destination), relative.as_posix()))
        report()

    def verify(task):
        if stop and stop.is_set():
            return None
        try:
            with Image.open(task.source) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("不支持多帧图片")
                image.verify()
            return task
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            return {"file": task.relative, "error": str(error)[:300]}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task, result in zip(candidates, pool.map(verify, candidates)):
            if isinstance(result, ImageTask):
                if Path(result.destination).exists() and not overwrite:
                    skipped += 1
                else:
                    tasks.append(result)
            elif result is not None:
                invalid.append(result)
                preserve(Path(task.source), invalid_image=True)
    report(force=True)
    return tasks, skipped, invalid


def regions_for(image: Image.Image, options: ProcessingConfig):
    preset = options.preset
    if preset == "auto":
        raise ValueError("auto 模式必须逐图执行视觉检测，不能使用固定模板")
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


def harmonize_repair(original, generated, regions, reference_resolution=512, *, conditioning_initialization=False):
    """Fit a smooth RGB offset from unoccluded pixels, never from watermark pixels.

    Coarse-to-fine harmonic extension uses Neumann boundaries at image edges.
    Only low-frequency colour changes; composition still controls the edit footprint.
    """
    if original.size != generated.size:
        raise ValueError("Harmonization requires matching image dimensions")
    if not regions:
        return generated
    capture=os.environ.get("WATERMARK_REPAIR_CAPTURE")
    if capture:
        folder=Path(capture)/f"{os.getpid()}-{original.width}x{original.height}"
        folder.mkdir(parents=True,exist_ok=True)
        original.save(folder/"original.png",compress_level=1)
        generated.save(folder/"generated.png",compress_level=1)
        (folder/"regions.json").write_text(json.dumps([r.model_dump() for r in regions]))
    highest_scale=min(1,reference_resolution/max(original.size))
    highest_size=(max(4,round(original.width*highest_scale)),max(4,round(original.height*highest_scale)))
    source_pyramid=original.convert("RGB").resize(highest_size,Image.Resampling.BOX)
    candidate_pyramid=generated.convert("RGB").resize(highest_size,Image.Resampling.BOX)
    previous = None
    levels=[(16,96),(32,64),(64,40),(128,24),(256,24),(512,16)]
    if reference_resolution>=1024:
        levels.append((1024,16))
    for longest, iterations in levels:
        scale = min(1, longest/max(original.size))
        size = (max(4, round(original.width*scale)), max(4, round(original.height*scale)))
        source = np.asarray(source_pyramid.resize(size, Image.Resampling.BOX), dtype=np.float32)
        candidate = np.asarray(candidate_pyramid.resize(size, Image.Resampling.BOX), dtype=np.float32)
        valid = np.ones((size[1], size[0]), dtype=bool)
        for region in regions:
            x0, y0, x1, y1 = region.box
            valid[max(0, int(y0*size[1])):min(size[1], math.ceil(y1*size[1])),
                  max(0, int(x0*size[0])):min(size[0], math.ceil(x1*size[0]))] = False
        if valid.sum() < 4:
            return generated
        raw_delta=source-candidate
        weighted=np.where(valid[...,None],raw_delta,0)
        weights=np.pad(valid.astype(np.float32),1,mode="edge")
        padded_delta=np.pad(weighted,((1,1),(1,1),(0,0)),mode="edge")
        numerator=np.zeros_like(weighted)
        denominator=np.zeros_like(valid,dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                numerator+=padded_delta[dy:dy+size[1],dx:dx+size[0]]
                denominator+=weights[dy:dy+size[1],dx:dx+size[0]]
        boundary=numerator/np.maximum(denominator[...,None],1)

        if previous is None:
            field = np.broadcast_to(np.median(boundary[valid], axis=0), boundary.shape).copy()
        else:
            field = np.stack([
                np.asarray(Image.fromarray(previous[..., channel]).resize(size, Image.Resampling.BILINEAR))
                for channel in range(3)
            ], axis=2).astype(np.float32)
        field[valid] = boundary[valid]
        yy, xx = np.indices(valid.shape)
        phases = [(~valid) & ((xx+yy)%2 == parity) for parity in (0, 1)]
        for _ in range(iterations):
            largest_update = 0.
            for active in phases:
                if not active.any():
                    continue
                padded = np.pad(field, ((1,1),(1,1),(0,0)), mode="edge")
                average = (padded[:-2,1:-1]+padded[2:,1:-1]+padded[1:-1,:-2]+padded[1:-1,2:])*.25
                delta = 1.65*(average[active]-field[active])
                field[active] += delta
                largest_update = max(largest_update, float(np.abs(delta).max()))
            if largest_update < .02:
                break
        previous = np.clip(field, -255, 255)
    result = np.asarray(generated.convert("RGB"), dtype=np.float32).copy()
    for channel in range(3):
        offset = Image.fromarray(previous[..., channel]).resize(original.size, Image.Resampling.BILINEAR)
        result[..., channel] += np.asarray(offset, dtype=np.float32)
    output=Image.fromarray(np.clip(np.rint(result),0,255).astype(np.uint8))
    # A smooth colour field cannot repair a generated scene/material change.
    # Do not disguise catastrophic whole-frame drift as plausible local texture.
    # Preserve the diagnostic even when correction is withheld, so QA/retry can
    # veto it rather than treating a colour-matched reconstruction as evidence.
    quantiles=np.quantile(raw_delta[valid],[.1,.9],axis=0)
    shift_spread=float(np.median(quantiles[1]-quantiles[0]))
    excessive_drift=float((np.abs(raw_delta[valid])>64).mean())>=.5 and shift_spread>=32
    if excessive_drift and not conditioning_initialization:
        output=generated.convert("RGB").copy()
    output.info["harmonization"]={
        "method":"masked_harmonic_rgb_v3","reference_grid":list(size),
        "correction_withheld":excessive_drift and not conditioning_initialization,
        "conditioning_only":conditioning_initialization,"reference_shift_spread":round(shift_spread,3),
        "reference_max_abs_delta":round(float(np.abs(raw_delta[valid]).max()),3),
        "reference_over_64_fraction":round(float((np.abs(raw_delta[valid])>64).mean()),5),
        "offset_min":round(float(previous.min()),3),"offset_max":round(float(previous.max()),3),
    }
    return output


def blend_patch(original: Image.Image, patch: Image.Image, box, crop, feather: int, blending="feather",
                *, reference_original=None, exclude_regions=(), core_box=None):
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
    if core_box is not None:
        kx0,ky0,kx1,ky1=core_box
        if not (x0<=kx0<kx1<=x1 and y0<=ky0<ky1<=y1):
            raise ValueError("Opaque repair core must be inside the padded overlay")
        def ramp(values,outer0,inner0,inner1,outer1):
            left=(values>=inner0).astype(np.float32) if inner0==outer0 else np.clip((values-outer0).astype(np.float32)/(inner0-outer0),0,1)
            right=(values<inner1).astype(np.float32) if inner1==outer1 else np.clip((outer1-1-values).astype(np.float32)/(outer1-inner1),0,1)
            return np.minimum(left,right)
        alpha=np.minimum(ramp(xx,x0,kx0,kx1,x1),ramp(yy,y0,ky0,ky1,y1))
    alpha = alpha * alpha * (3 - 2 * alpha)
    mask = Image.fromarray(np.uint8(np.round(alpha * 255)))
    resized = patch.convert("RGB").resize((cx1 - cx0, cy1 - cy0), Image.Resampling.LANCZOS)
    if blending in {"harmonic","harmonic-fine"}:
        from .config import Region
        local = Region(box=((x0-cx0)/(cx1-cx0), (y0-cy0)/(cy1-cy0),
                            (x1-cx0)/(cx1-cx0), (y1-cy0)/(cy1-cy0)))
        excluded=[local]
        for region in exclude_regions:
            rx0,ry0,rx1,ry1=pixel_box(region.box,original.size)
            left,top,right,bottom=max(cx0,rx0),max(cy0,ry0),min(cx1,rx1),min(cy1,ry1)
            if left<right and top<bottom:
                excluded.append(Region(box=((left-cx0)/(cx1-cx0),(top-cy0)/(cy1-cy0),
                                            (right-cx0)/(cx1-cx0),(bottom-cy0)/(cy1-cy0))))
        reference=reference_original if reference_original is not None else original
        resized = harmonize_repair(reference.crop(crop),resized,excluded,1024 if blending=="harmonic-fine" else 512)
    result = original.copy()
    result.paste(resized, (cx0, cy0), mask)
    if "harmonization" in resized.info:
        result.info["harmonization"]=resized.info["harmonization"]
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



def restore_storage_orientation(image, original_path):
    """Inverse EXIF display transform; preserve both stored width/height and orientation."""
    with Image.open(original_path) as raw:
        original_size = raw.size
        exif = raw.getexif()
        orientation = exif.get(274, 1)
    inverse = {
        2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_270,
    }
    if orientation in inverse:
        image = image.transpose(inverse[orientation])
    if image.size != original_size:
        raise ValueError("保存尺寸与原文件不一致，停止写入")
    return image, exif.tobytes()

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


def atomic_copy(source, destination, overwrite=False, *, preserve_metadata=False):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temp)
        if preserve_metadata:
            shutil.copystat(source, temp)
        if overwrite:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_symlink(source, destination, overwrite=False):
    """Recreate a link verbatim, including dangling/cyclic links, without traversal."""
    path = Path(destination)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        os.symlink(os.readlink(source), temp)
        if overwrite:
            os.replace(temp, path)
        else:
            os.link(temp, path, follow_symlinks=False)
    finally:
        temp.unlink(missing_ok=True)

def save_output(image, source, destination, exif, icc, options):
    """Resize only the completed display-oriented image, never an individual patch."""
    if options.output_width:
        image = image.resize(
            (options.output_width, options.output_height), Image.Resampling.LANCZOS
        )
        metadata = Image.Exif()
        metadata.load(exif)
        metadata.pop(274, None)
        for key in (256, 257, 40962, 40963):
            metadata.pop(key, None)
        exif = metadata.tobytes()
    else:
        image, exif = restore_storage_orientation(image, source)
    atomic_save(image, destination, exif, icc, options.overwrite)
    return list(image.size)

def inference_shape(path, resolution):
    """Same 32-pixel rounding as the pinned pipeline, using display aspect ratio."""
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274, 1) in {5, 6, 7, 8}:
            width, height = height, width
    ratio = width / height
    width = math.sqrt(resolution * resolution * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32

def composite_protected(original, generated, regions, feather=12, blending="feather", *, core_boxes=None):
    """Keep the source outside automatically detected overlay masks."""
    if original.size != generated.size:
        raise ValueError("Protected composition requires source-sized generated image")
    if blending in {"harmonic","harmonic-fine"}:
        generated = harmonize_repair(original,generated,regions,1024 if blending=="harmonic-fine" else 512)
    result = original.copy()
    w, h = original.size
    result.info.update({k:v for k,v in generated.info.items() if k=="harmonization"})
    for region in regions:
        box = pixel_box(region.box, original.size)
        x0, y0, x1, y1 = box
        crop = (max(0, x0-feather), max(0, y0-feather),
                min(w, x1+feather), min(h, y1+feather))
        core=(core_boxes or {}).get(tuple(region.box))
        core=pixel_box(core,original.size) if core is not None else None
        result = blend_patch(result, generated.crop(crop), box, crop, feather, core_box=core)
    return result


def prefill_overlay_reference(original, box, crop, regions):
    """Erase the input overlay's appearance using only real unoccluded SOURCE colors.

    This is a conditioning initialization, never an accepted repair or an extra candidate.
    Original pixels outside the target rectangle remain exact.
    """
    from .config import Region
    cx0,cy0,cx1,cy1=crop
    reference=original.crop(crop).convert("RGB")
    width,height=reference.size
    exclusions=[]
    for region in regions:
        left,top,right,bottom=pixel_box(region.box,original.size)
        left,top,right,bottom=max(cx0,left),max(cy0,top),min(cx1,right),min(cy1,bottom)
        if left<right and top<bottom:
            exclusions.append(Region(box=((left-cx0)/width,(top-cy0)/height,
                                           (right-cx0)/width,(bottom-cy0)/height)))
    x0,y0,x1,y1=box
    if not (cx0<=x0<x1<=cx1 and cy0<=y0<y1<=cy1):
        raise ValueError("Prefill target must stay inside the original crop")
    exclusions.append(Region(box=((x0-cx0)/width,(y0-cy0)/height,
                                  (x1-cx0)/width,(y1-cy0)/height)))
    scale=min(1,256/max(reference.size))
    small_size=(max(4,round(width*scale)),max(4,round(height*scale)))
    known=np.ones((small_size[1],small_size[0]),dtype=bool)
    for region in exclusions:
        left,top,right,bottom=pixel_box(region.box,small_size)
        known[top:bottom,left:right]=False
    if known.sum()<4:
        return reference,False
    small=reference.resize(small_size,Image.Resampling.BOX)
    neutral=Image.new("RGB",small_size,(128,128,128))
    field=harmonize_repair(small,neutral,exclusions,reference_resolution=256)
    field=field.resize(reference.size,Image.Resampling.BILINEAR)
    target=(x0-cx0,y0-cy0,x1-cx0,y1-cy0)
    result=reference.copy()
    result.paste(field.crop(target),target)
    return result,True



def protect_edge_completion_context(original, edited, evidence):
    """Use generated extension pixels only for source stroke bounds plus 2px."""
    from PIL import ImageDraw

    if not evidence:
        return edited,0
    if original.size!=edited.size:
        raise ValueError("Edge context protection requires the original pixel grid")
    result=edited.copy()
    restored=0
    for record in evidence:
        outer=pixel_box(record["box"],original.size)
        inner=pixel_box(record["original_box"],original.size)
        if not (outer[0]<=inner[0]<inner[2]<=outer[2] and outer[1]<=inner[1]<inner[3]<=outer[3]):
            raise ValueError("Original edge mask must stay inside its completion")
        width,height=outer[2]-outer[0],outer[3]-outer[1]
        mask=Image.new("L",(width,height),0)
        draw=ImageDraw.Draw(mask)
        draw.rectangle((inner[0]-outer[0],inner[1]-outer[1],
                        inner[2]-outer[0]-1,inner[3]-outer[1]-1),fill=255)
        for side in record["sides"]:
            boxes=side.get("source_stroke_boxes")
            if not boxes:
                raise ValueError("Source edge completion needs explicit stroke bounds")
            for x0,y0,x1,y1 in boxes:
                if not (0<=x0<x1<=original.width and 0<=y0<y1<=original.height):
                    raise ValueError("Source stroke bounds must stay inside the photo")
                draw.rectangle((max(0,x0-outer[0]-2),max(0,y0-outer[1]-2),
                                min(width-1,x1-outer[0]+1),min(height-1,y1-outer[1]+1)),fill=255)
        restore=ImageOps.invert(mask)
        restored+=int(np.count_nonzero(np.asarray(restore)))
        result.paste(original.crop(outer).convert(result.mode),outer,restore)
    return result,restored

def prefill_whole_reference(original, regions, core_mapping):
    """Optional full-frame conditioning; erase only validated digital source cores.

    The unmodified original remains the QA/compositing reference. One field is
    computed for all cores; no model ROI crops or extra generated candidates.
    """
    from .config import Region
    from .surface_reference import _box
    cores=[]
    for region in regions:
        if region.kind!="panel":
            continue
        # Small footer/code detections deliberately have no feather core. Leave
        # their ORIGINAL appearance for Qwen to remove; their missing metadata
        # must not cancel initialization of a separate validated large card.
        l,t,r,b=region.box
        if (r-l)*(b-t)<.035:
            continue
        raw=core_mapping.get(tuple(region.box))
        core=_box(raw)
        if (core is None or not (region.box[0]<=core[0]<core[2]<=region.box[2]
                               and region.box[1]<=core[1]<core[3]<=region.box[3])):
            return original,False
        cores.append(Region(kind=region.kind,box=core))
    if not cores:
        return original,False
    scale=min(1,256/max(original.size))
    size=(max(4,round(original.width*scale)),max(4,round(original.height*scale)))
    known=np.ones((size[1],size[0]),dtype=bool)
    # Exclude the full safety margins from the field's SOURCE references.
    for region in regions:
        l,t,r,b=pixel_box(region.box,size)
        known[t:b,l:r]=False
    if known.sum()<4:
        return original,False
    small=original.convert("RGB").resize(size,Image.Resampling.BOX)
    field=harmonize_repair(small,Image.new("RGB",size,(128,128,128)),regions,256,
                           conditioning_initialization=True)
    field=field.resize(original.size,Image.Resampling.BILINEAR)
    result=original.convert("RGB").copy()
    for core in cores:
        box=pixel_box(core.box,original.size)
        result.paste(field.crop(box),box)
    return result,True
