from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.images import (
    atomic_save,
    blend_patch,
    regions_for,
    scan_images,
    validate_directories,
)


def test_walk_filters_and_preserves_tree(tmp_path):
    source, target = tmp_path / "input", tmp_path / "output"
    (source / "nested" / "empty").mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(source / "nested" / "photo.PNG")
    (source / "nested" / "notes.txt").write_text("not an image")
    (source / "bad.jpg").write_text("broken")
    (source / "cycle").symlink_to(source, target_is_directory=True)
    tasks, skipped, invalid = scan_images(source, target, False)
    assert [t.relative for t in tasks] == ["nested/photo.PNG"]
    assert Path(tasks[0].destination) == target / "nested" / "photo.PNG"
    assert (target / "nested" / "empty").is_dir()
    assert (target / "nested" / "notes.txt").read_bytes() == b"not an image"
    assert (target / "bad.jpg").read_bytes() == b"broken"
    assert (target / "cycle").is_symlink()
    assert skipped == 0 and len(invalid) == 1
    Image.new("RGB", (20, 20)).save(target / "nested" / "photo.PNG")
    assert scan_images(source, target, False)[1] == 1


def test_no_nested_or_identical_output(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    for destination in [source, source / "out", tmp_path]:
        with pytest.raises(ValueError):
            validate_directories(str(source), str(destination))


def test_blending_opaque_core_and_exact_outside():
    original = Image.new("RGB", (100, 100), "black")
    patch = Image.new("RGB", (80, 80), "white")
    result = np.asarray(blend_patch(original, patch, (30, 30, 60, 60), (10, 10, 90, 90), 8))
    assert (result[30:60, 30:60] == 255).all()
    assert (result[:22] == 0).all()
    assert (result[68:] == 0).all()
    assert 0 < result[26, 40, 0] < 255


def test_atomic_save_does_not_overwrite_or_leave_temporary_files(tmp_path):
    destination = tmp_path / "photo.png"
    Image.new("RGB", (5, 5), "red").save(destination)
    old = destination.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_save(Image.new("RGB", (5, 5), "blue"), destination)
    assert destination.read_bytes() == old
    assert len(list(tmp_path.iterdir())) == 1


def test_explicit_templates_and_parameter_validation():
    image = Image.new("RGB", (256, 192), (70, 80, 50))
    assert regions_for(image, ProcessingConfig(preset="gray"))[0].box[1] == 0.727
    image.paste((50, 130, 220), (0, 150, 100, 170))
    assert regions_for(image, ProcessingConfig(preset="blue"))[0].box[1] == 0.781
    with pytest.raises(ValueError):
        regions_for(image, ProcessingConfig())
    with pytest.raises(ValueError):
        DeploymentConfig(attention="flex", compile=False)
    with pytest.raises(ValueError):
        DeploymentConfig(compile=True, offload="model")
    with pytest.raises(ValueError):
        ProcessingConfig(preset="custom", regions=[])


def test_all_exif_orientations_preserve_stored_and_display_dimensions(tmp_path):
    from PIL import ImageOps

    from watermark_studio.images import load_photo, restore_storage_orientation
    for orientation in range(1, 9):
        image = Image.new("RGB", (173, 127))
        image.putpixel((10, 20), (255, 70, 20))
        exif = image.getexif()
        exif[274] = orientation
        path = tmp_path / f"orientation-{orientation}.png"
        image.save(path, exif=exif)
        displayed, _, _, _ = load_photo(path)
        stored, restored_exif = restore_storage_orientation(displayed, path)
        assert stored.size == (173, 127)
        assert stored.tobytes() == image.tobytes()
        output = tmp_path / f"output-{orientation}.png"
        stored.save(output, exif=restored_exif)
        with Image.open(path) as a, Image.open(output) as b:
            assert ImageOps.exif_transpose(a).tobytes() == ImageOps.exif_transpose(b).tobytes()


def test_deep_complete_mirror_and_nonprocessable_images(tmp_path):
    import os

    source, target = tmp_path/"input", tmp_path/"output"
    leaf = source/"某镇"/"某村"/"施工照片"/"第五层"/"第六层"
    (leaf/"隐藏目录"/"空目录").mkdir(parents=True)
    (source/"另一乡"/"另一村").mkdir(parents=True)
    payloads = {"说明.txt": "保留内容及命名".encode(), "表格.xlsx": bytes(range(256))*30,
                "无扩展名": b"opaque", ".hidden": b"hidden", "坏图.JPG": b"broken"}
    for name, payload in payloads.items():
        (leaf/name).write_bytes(payload)
    os.utime(leaf/"说明.txt", ns=(1234567890123456789, 1234567890123456789))
    Image.new("RGB", (73, 41), "red").save(leaf/"照片.JPG")
    Image.new("RGB", (39, 71), "blue").save(source/"另一乡"/"另一村"/"照片.JPG")
    Image.new("RGB", (7, 9)).save(leaf/"动画.gif")
    Image.new("RGB", (9, 7)).save(leaf/"多页.tiff", save_all=True,
                                         append_images=[Image.new("RGB", (9, 7))])
    (leaf/"相对链接").symlink_to("说明.txt")
    (leaf/"失效链接").symlink_to("missing")
    stats = {}
    tasks, skipped, invalid = scan_images(source, target, False, workers=2, mirror_stats=stats)
    rel = leaf.relative_to(source)
    assert {t.relative for t in tasks} == {(rel/"照片.JPG").as_posix(), "另一乡/另一村/照片.JPG"}
    assert len(invalid) == 2 and skipped == 0
    assert (target/rel/"隐藏目录"/"空目录").is_dir()
    for name in (*payloads, "动画.gif", "多页.tiff"):
        assert (target/rel/name).read_bytes() == (leaf/name).read_bytes()
    for name in ("相对链接", "失效链接"):
        assert os.readlink(target/rel/name) == os.readlink(leaf/name)
    assert (target/rel/"说明.txt").stat().st_mtime_ns == (leaf/"说明.txt").stat().st_mtime_ns
    assert stats["copied_files"] == 7 and stats["copied_invalid_images"] == 2
    assert stats["copied_links"] == 2 and stats["copy_failures"] == 0


def test_auxiliary_copy_overwrite_is_separate_from_image_review(tmp_path):
    source, target = tmp_path/"input", tmp_path/"output"
    source.mkdir(); target.mkdir()
    (source/"notes.txt").write_bytes(b"new")
    (target/"notes.txt").write_bytes(b"old")
    Image.new("RGB", (4, 5)).save(source/"photo.png")
    stats = {}
    assert len(scan_images(source, target, True, copy_overwrite=False, mirror_stats=stats)[0]) == 1
    assert (target/"notes.txt").read_bytes() == b"old" and stats["existing_files"] == 1
    scan_images(source, target, True, mirror_stats=stats)
    assert (target/"notes.txt").read_bytes() == b"new" and stats["copied_files"] == 1


def test_mirror_copy_failure_preserves_source_and_leaves_no_partial_file(tmp_path, monkeypatch):
    from watermark_studio import images

    source, target = tmp_path/"input", tmp_path/"output"
    source.mkdir()
    (source/"notes.txt").write_bytes(b"unchanged source")
    def interrupted_copy(_source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(images.shutil, "copyfile", interrupted_copy)
    stats = {}
    scan_images(source, target, False, mirror_stats=stats)
    assert stats["copy_failures"] == 1 and stats["errors"][0]["file"] == "notes.txt"
    assert not list(target.iterdir()) and (source/"notes.txt").read_bytes() == b"unchanged source"


def test_mirror_rejects_output_link_escape_and_file_directory_conflicts(tmp_path):
    source, target, outside = tmp_path/"input", tmp_path/"output", tmp_path/"outside"
    source.mkdir(); target.mkdir(); outside.mkdir()
    (source/"notes.txt").write_bytes(b"input")
    (outside/"sentinel").write_bytes(b"protected")
    (target/"notes.txt").symlink_to(outside/"sentinel")
    stats = {}
    scan_images(source, target, True, mirror_stats=stats)
    assert stats["copy_failures"] == 1
    assert (outside/"sentinel").read_bytes() == b"protected"
    (target/"notes.txt").unlink(); (target/"notes.txt").mkdir()
    scan_images(source, target, False, mirror_stats=stats)
    assert stats["copy_failures"] == 1 and stats["existing_files"] == 0
    (source/"sub").mkdir(); (target/"sub").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        scan_images(source, target, False)
