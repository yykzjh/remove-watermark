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
