import time

import pytest
from PIL import Image, ImageDraw

from watermark_studio.codex_photo_repair import native_output, protected_repair, repair_masks


def test_only_session_created_images_are_accepted_as_native_edits(tmp_path):
    root = tmp_path / "generated"
    folder = root / "this-session"
    folder.mkdir(parents=True)
    path = folder / "exec-012abc.png"
    Image.new("RGB", (256, 256), "green").save(path)
    assert native_output(str(path), [{"type": "thread.started", "thread_id": "this-session"}],
                         time.time() - 1, root) == path
    with pytest.raises(ValueError, match="session"):
        native_output(str(path), [{"type": "thread.started", "thread_id": "other-session"}],
                      time.time() - 1, root)
    outside = tmp_path / "exec-012abc.png"
    outside.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        native_output(str(outside), [], time.time() - 1, root)


def test_protected_composition_preserves_pixels_outside_repair(tmp_path):
    original, candidate, generated = [tmp_path / n for n in ["original.png", "candidate.png", "native.png"]]
    Image.new("RGB", (512, 512), "green").save(original)
    Image.new("RGB", (512, 512), "blue").save(candidate)
    native = Image.new("RGB", (768, 768), "blue")
    ImageDraw.Draw(native).rectangle((310, 310, 460, 460), fill="red")
    native.save(generated)
    defects = [{"kind": "residual_watermark", "box": [.4, .4, .6, .6]}]
    result, _, _ = protected_repair(original, candidate, generated, defects, [])
    assert result.size == (512, 512)
    assert result.getpixel((256, 256)) == (255, 0, 0)
    for xy in [(0, 0), (511, 511), (0, 256), (256, 0)]:
        assert result.getpixel(xy) == (0, 0, 255)


def test_visible_content_restored_exactly_but_occluded_content_requires_generation(tmp_path):
    regions = [{"box": [.5, .5, .9, .9]}]
    defects = [{"kind": "content_change", "box": [.1, .1, .8, .8]}]
    restore, repaint = repair_masks((512, 512), defects, regions)
    assert restore.getpixel((100, 100)) == 255
    assert restore.getpixel((320, 320)) == 0
    assert repaint.getpixel((320, 320)) == 255
    assert repaint.getpixel((100, 100)) == 0
    original, candidate = tmp_path / "original.png", tmp_path / "candidate.png"
    Image.new("RGB", (512, 512), "green").save(original)
    Image.new("RGB", (512, 512), "red").save(candidate)
    result, _, _ = protected_repair(original, candidate, None, defects, regions)
    assert result.getpixel((100, 100)) == (0, 128, 0)
    assert result.getpixel((320, 320)) == (255, 0, 0)


def case_verdict(kind=None):
    value = {"acceptable": kind is None, "watermark_removed": kind != "residual_watermark",
             "natural_fill": kind != "unnatural_fill", "content_preserved": kind != "content_change",
             "uncertain": False, "evidence": ["Observed image-grounded comparison evidence."], "defects": []}
    if kind:
        value["defects"] = [{"kind": kind, "box": [.1, .1, .4, .4],
                            "evidence": "Specific visible defect.", "repair_instruction": "Fix only this area."}]
    return value


def test_worse_generated_candidate_never_overwrites_best_output(tmp_path, monkeypatch):
    from watermark_studio.codex_photo_repair import repair_one
    from watermark_studio.production_archives import file_sha256
    original, target, generated = [tmp_path / n for n in ["original.png", "target.png", "native.png"]]
    Image.new("RGB", (512, 512), "green").save(original)
    Image.new("RGB", (512, 512), "blue").save(target)
    Image.new("RGB", (512, 512), "red").save(generated)
    initial = target.read_bytes()
    monkeypatch.setattr("watermark_studio.codex_photo_repair.generate", lambda *a: generated)
    monkeypatch.setattr("watermark_studio.codex_photo_repair.review_photo",
                        lambda *a: {"verdict": case_verdict("content_change")})
    entry = {"file": "target.png", "identity": {"output_sha256": file_sha256(target)},
             "verdict": case_verdict("residual_watermark")}
    report = repair_one("unused", original, target, entry, [], tmp_path / "repair", attempts=1)
    assert report["status"] == "needs_review"
    assert target.read_bytes() == initial
    assert report["history"][0]["kept_previous_best"] is True


def test_visible_source_content_can_be_restored_without_extra_generation(tmp_path, monkeypatch):
    from watermark_studio.codex_photo_repair import repair_one
    from watermark_studio.production_archives import file_sha256
    original, target = tmp_path / "original.png", tmp_path / "target.png"
    Image.new("RGB", (512, 512), "green").save(original)
    Image.new("RGB", (512, 512), "red").save(target)
    monkeypatch.setattr("watermark_studio.codex_photo_repair.generate",
                        lambda *a: pytest.fail("Visible source restoration must not invent content"))
    monkeypatch.setattr("watermark_studio.codex_photo_repair.review_photo",
                        lambda *a: {"verdict": case_verdict()})
    entry = {"file": "target.png", "identity": {"output_sha256": file_sha256(target)},
             "verdict": case_verdict("content_change")}
    report = repair_one("unused", original, target, entry, [{"box": [.6, .6, .9, .9]}],
                        tmp_path / "repair", attempts=1)
    assert report["status"] == "passed"
    with Image.open(target) as image:
        assert image.getpixel((100, 100)) == (0, 128, 0)
        assert image.getpixel((500, 500)) == (255, 0, 0)


def test_early_verified_repair_can_be_reapplied_only_to_its_exact_baseline(tmp_path, monkeypatch):
    from watermark_studio.codex_photo_repair import repair_one
    from watermark_studio.production_archives import file_sha256
    original, target = tmp_path / "original.png", tmp_path / "target.png"
    Image.new("RGB", (512, 512), "green").save(original)
    Image.new("RGB", (512, 512), "red").save(target)
    baseline = target.read_bytes()
    monkeypatch.setattr("watermark_studio.codex_photo_repair.review_photo",
                        lambda *a: {"verdict": case_verdict()})
    entry = {"file": "target.png", "identity": {"output_sha256": file_sha256(target)},
             "verdict": case_verdict("content_change")}
    folder = tmp_path / "repair"
    report = repair_one("unused", original, target, entry, [], folder, attempts=1)
    target.write_bytes(baseline)  # final full assembly restores the original Qwen output.
    monkeypatch.setattr("watermark_studio.codex_photo_repair.review_photo",
                        lambda *a: pytest.fail("Exact cached repair should not generate a new grade"))
    assert repair_one("unused", original, target, entry, [], folder) == report
    assert file_sha256(target) == report["final_sha256"]
    Image.new("RGB", (512, 512), "yellow").save(target)
    with pytest.raises(ValueError, match="changed"):
        repair_one("unused", original, target, entry, [], folder)


def test_tiny_defects_receive_a_context_view_without_losing_source_coordinates():
    from watermark_studio.codex_photo_repair import context_view
    box = [.86, .969, .998, .999]
    view = context_view((4096, 3072), [{"box": box}])
    assert view is not None
    x0, y0, x1, y1 = view
    assert x1 - x0 == y1 - y0
    assert 0 <= x0 < box[0] * 4096 < box[2] * 4096 <= x1 <= 4096
    assert 0 <= y0 < box[1] * 3072 < box[3] * 3072 <= y1 <= 3072
    assert (x1-x0)*(y1-y0) < .2 * 4096 * 3072
    assert context_view((4096, 3072), [{"box": [.01, .55, .9, .98]}]) is None


def test_context_patch_maps_back_to_original_canvas_and_preserves_every_other_area(tmp_path):
    original, candidate, generated = [tmp_path / n for n in ["original.png", "candidate.png", "patch.png"]]
    Image.new("RGB", (512, 384), "green").save(original)
    Image.new("RGB", (512, 384), "blue").save(candidate)
    patch = Image.new("RGB", (256, 256), "blue")
    ImageDraw.Draw(patch).rectangle((110, 110, 150, 150), fill="red")
    patch.save(generated)
    view = {"path": generated, "view_box": [256, 128, 512, 384], "display_size": [512, 384]}
    defects = [{"kind": "residual_watermark", "box": [.70, .60, .80, .75]}]
    result, _, _ = protected_repair(original, candidate, view, defects, [])
    assert result.size == (512, 384)
    assert result.getpixel((384, 256)) == (255, 0, 0)
    assert result.getpixel((10, 10)) == (0, 0, 255)
    assert result.getpixel((300, 350)) == (0, 0, 255)
    with pytest.raises(ValueError, match="original"):
        protected_repair(original, candidate, view | {"display_size": [4096, 3072]}, defects, [])
