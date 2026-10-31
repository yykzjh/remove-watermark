import json

import pytest
from PIL import Image

from watermark_studio.config import ProcessingConfig
from watermark_studio.quality import deterministic_warnings, parse_review, pixel_change


def verdict(**changes):
    return json.dumps({"watermark_removed": True, "natural": True,
                       "content_preserved": True, "uncertain": False, **changes})


def test_pass_requires_all_checks_and_no_uncertainty():
    assert parse_review(verdict())["status"] == "passed"
    for flag in ("watermark_removed", "natural", "content_preserved"):
        assert parse_review(verdict(**{flag: False}))["status"] == "review"
    assert parse_review(verdict(uncertain=True))["status"] == "review"


@pytest.mark.parametrize("text", ['{}', 'not json', '{"watermark_removed":"true"}'])
def test_incomplete_or_invalid_verdict_is_never_a_pass(text):
    with pytest.raises((ValueError, TypeError)):
        parse_review(text)


def test_size_change_alone_has_no_change_in_constant_color():
    assert pixel_change(Image.new("RGB", (400, 300), "blue"),
                        Image.new("RGB", (800, 600), "blue")) == 0


def test_quality_is_opt_in_and_parameters_are_bounded():
    assert not ProcessingConfig().quality_check
    with pytest.raises(ValueError):
        ProcessingConfig(quality_batch_size=16)


def test_unchanged_and_introduced_flat_fill_require_review():
    before = Image.new("RGB", (400, 300), "blue")
    assert deterministic_warnings(before, before.copy())
    after = before.copy()
    from PIL import ImageDraw
    ImageDraw.Draw(after).rectangle((0, 0, 200, 150), fill="lime")
    assert deterministic_warnings(before, after)

def test_detail_crops_include_a_small_local_change():
    from PIL import ImageDraw

    from watermark_studio.quality import detail_sheets
    before = Image.new("RGB", (1024, 768), "gray")
    after = before.copy()
    ImageDraw.Draw(after).rectangle((700, 350, 730, 380), fill="red")
    sheets, boxes = detail_sheets(before, after, 4)
    assert len(sheets) == 2 and len(boxes) == 4
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 715/1024 <= x1 and y0 <= 365/768 <= y1


def test_review_grammar_only_allows_complete_typed_json():
    from watermark_studio.quality import allowed_review_tokens

    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    tree = allowed_review_tokens(CharTokenizer(), [0])
    outputs = []
    def walk(node, prefix):
        for token, child in node.items():
            if token == 0:
                outputs.append(json.loads("".join(map(chr, prefix))))
            else:
                walk(child, prefix+[token])
    walk(tree, [])
    assert len(outputs) == 160
    for row in outputs:
        assert all(type(row[k]) is bool for k in
                   ("watermark_removed", "natural", "content_preserved", "uncertain"))
        assert parse_review(json.dumps(row))["status"] in {"passed", "review"}

def test_protected_composition_does_not_change_pixels_outside_feather():
    from watermark_studio.config import Region
    from watermark_studio.images import composite_protected
    before = Image.new("RGB", (100, 80), "blue")
    generated = Image.new("RGB", before.size, "red")
    result = composite_protected(before, generated, [Region(box=(.2, .2, .4, .4))], feather=3)
    assert result.getpixel((30, 24)) == (255, 0, 0)
    assert result.getpixel((10, 10)) == before.getpixel((10, 10))
    import numpy as np
    changed = np.any(np.array(result) != np.array(before), axis=2)
    changed[13:35, 17:43] = False
    assert not changed.any()

def test_local_guard_detects_subtle_change_outside_removed_overlay():
    from PIL import ImageDraw

    from watermark_studio.config import Region
    from watermark_studio.quality import outside_repair_changes
    before = Image.new("RGB", (1024, 768), "gray")
    after = before.copy()
    ImageDraw.Draw(before).rectangle((20, 550, 300, 750), fill="blue")
    regions = [Region(box=(.01, .70, .32, 1.))]
    assert outside_repair_changes(before, after, regions)["tiles"] == 0
    ImageDraw.Draw(after).text((700, 300), "CAMERA", fill=(160, 160, 160))
    change=outside_repair_changes(before, after, regions)
    assert change["tiles"] > 0
    from watermark_studio.evidence import local_findings
    assert local_findings({"outside_repair_change":change})[0]["box"]


def test_local_guard_tolerates_requested_output_resize_and_jpeg():
    import io

    import numpy as np

    from watermark_studio.quality import outside_repair_changes
    rng = np.random.default_rng(42)
    before = Image.fromarray(rng.integers(0, 256, (768, 1024, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    before.save(buffer, format="JPEG", quality=97, subsampling=0)
    after = Image.open(io.BytesIO(buffer.getvalue()))
    assert outside_repair_changes(before, after, [])["tiles"] == 0
    after = before.resize((512, 384), Image.Resampling.LANCZOS)
    assert outside_repair_changes(before, after, [])["tiles"] == 0

def test_local_guard_accounts_for_actual_composition_feather():
    from watermark_studio.config import Region
    from watermark_studio.images import composite_protected
    from watermark_studio.quality import outside_repair_changes
    before = Image.new("RGB", (512, 384), "blue")
    regions = [Region(box=(.4, .4, .6, .6))]
    after = composite_protected(before, Image.new("RGB", before.size, "red"), regions, feather=64)
    assert outside_repair_changes(before, after, regions, feather=64)["tiles"] == 0
    assert outside_repair_changes(before, after, regions, feather=0)["tiles"] > 0

@pytest.mark.parametrize("orientation", range(1, 9))
def test_quality_size_check_uses_display_orientation(tmp_path, monkeypatch, orientation):
    import shutil
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.worker import process_batch

    source, target = tmp_path/"source.jpg", tmp_path/"output.jpg"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (60, 40), "blue").save(source, exif=exif)
    shutil.copyfile(source, target)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
    )))
    monkeypatch.setattr("watermark_studio.quality.verify_batch",
                        lambda *a, **kw: [{"status": "passed", "reason": ""}])
    pipe = SimpleNamespace(_quality_encoder=None, _quality_processor=None, _execution_device="cpu")
    result = process_batch([{"source": str(source), "destination": str(target),
                             "relative": "source.jpg", "reuse_existing": True}],
                           ProcessingConfig(quality_check=True, scope="full_image"), DeploymentConfig(), pipe)[0]
    assert result["quality"]["status"] == "passed"
    assert result["quality"]["dimensions_match"]
    assert result["quality"]["display_output_size"] == ([40, 60] if orientation >= 5 else [60, 40])
    assert result["output_size"] == [60, 40]

@pytest.mark.parametrize("batch_error",["oom","malformed"])
def test_quality_oom_releases_traceback_before_splitting(tmp_path, monkeypatch,batch_error):
    import sys
    import weakref
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.worker import process_batch

    class OOM(Exception):
        pass
    class HeldTensor:
        pass
    refs=[]
    def verify(pairs,*a,**kw):
        if len(pairs)>1:
            held=HeldTensor()
            refs.append(weakref.ref(held))
            raise (OOM if batch_error=="oom" else ValueError)("simulated batch failure")
        assert refs[0]() is None
        return [{"status":"passed","reason":""}]
    monkeypatch.setattr("watermark_studio.quality.verify_batch",verify)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,
        empty_cache=lambda:None,OutOfMemoryError=OOM,
    )))
    tasks=[]
    for name in ("a.png","b.png"):
        path=tmp_path/name
        Image.new("RGB",(64,48),"blue").save(path)
        tasks.append({"source":str(path),"destination":str(path),"relative":name,"reuse_existing":True})
    pipe=SimpleNamespace(_quality_encoder=None,_quality_processor=None,_execution_device="cpu")
    rows=process_batch(tasks,ProcessingConfig(scope="full_image",quality_check=True,quality_batch_size=2),
                       DeploymentConfig(),pipe)
    assert all(r["quality"]["status"]=="passed" and r["quality_oom_splits"]==int(batch_error=="oom") for r in rows)
    assert all(r["quality_validation_splits"]==int(batch_error=="malformed") for r in rows)


def test_material_context_is_per_image_and_does_not_leak_between_documents_and_outdoors(monkeypatch):
    from PIL import Image

    from watermark_studio.quality import verify_batch

    observed=[]
    def structured(chats,fields,*args):
        observed.append(chats)
        return [{field:field!="uncertain" for field in fields}|{"area":"none"} for _ in chats],.01
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
                        lambda *a:[([],""),([],"")])
    photo=Image.new("RGB",(200,180),"gray")
    results=verify_batch([(photo,photo.copy()),(photo,photo.copy())],None,None,"cpu",
        detail_patches=0,source_regions=[[],[]],
        editor_context=[{"source_scene":"document"},{"source_scene":"outdoor"}])
    for batch in observed:
        texts=[" ".join(item.get("text","") for item in chat[0]["content"]) for chat in batch]
        assert "完整原图是票据" in texts[0] and "完整原图是票据" not in texts[1]
    assert [r["scene_context"] for r in results]==["document","outdoor"]


@pytest.mark.parametrize("visible,expected",[(False,"passed"),(True,"review")])
def test_new_inset_cues_need_visual_confirmation_and_cannot_exempt_confirmed_damage(monkeypatch,visible,expected):
    from watermark_studio.config import Region
    from watermark_studio.quality import verify_batch

    def structured(chats,fields,*args):
        return [{f: (visible if f=="defect_visible" else f!="uncertain") for f in fields}|{"area":"none"}
                for _ in chats],.01
    def seam(*args,**kwargs):
        return {"count":int(kwargs.get("inset_search",False)),
                "suspect_segments":[{"box":[.2,.2,.3,.3],"side":"left","luma_step":5}]}
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    monkeypatch.setattr("watermark_studio.quality.repair_boundary_warnings",seam)
    monkeypatch.setattr("watermark_studio.quality.repair_texture_warnings",
                        lambda *a,**k:{"count":0,"suspect_segments":[]})
    monkeypatch.setattr("watermark_studio.quality.deterministic_warnings",lambda *a:[])
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",lambda *a:[([],"")])
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda chats,*a:[
        {"status":"ok","assessment":"no_clear_defect","summary":"No clear defect","findings":[]} for _ in chats])
    photo=Image.new("RGB",(400,300),"gray")
    q=verify_batch([(photo,photo.copy())],None,None,"cpu",detail_patches=0,
                   source_regions=[[Region(box=(.15,.15,.5,.5))]])[0]
    assert q["status"]==expected
    assert q["repair_cue_assessment"]["model"]["defect_visible"]==visible
    # Even a simultaneous natural_boundary=true cannot excuse defect_visible=true.
    assert ("repair_inset_confirmed" in q.get("guard_codes",[]))==visible


def test_low_amplitude_inset_rectangles_are_only_soft_candidates_and_real_existing_edges_are_retained():
    from PIL import ImageDraw

    from watermark_studio.config import Region
    from watermark_studio.quality import repair_boundary_warnings
    source=Image.new("RGB",(600,450),(100,100,100))
    after=source.copy()
    box=(120,180,420,400)
    ImageDraw.Draw(source).rectangle(box,fill="blue")
    ImageDraw.Draw(after).rectangle(box,fill=(94,94,94))
    regions=[Region(box=(.2,.4,.7,400/450))]
    assert repair_boundary_warnings(after,regions,before=source)["count"]==0
    assert repair_boundary_warnings(after,regions,before=source,inset_search=True,minimum_step=4)["count"]>0
    assert repair_boundary_warnings(after,regions,before=after,inset_search=True,minimum_step=4)["count"]==0
