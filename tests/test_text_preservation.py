import io
from types import SimpleNamespace

from PIL import Image, ImageDraw

from watermark_studio.text_preservation import OutputTextPreservation


def detector(boxes,status="text_present"):
    return SimpleNamespace(inspect=lambda *args:{"status":status,"regions":[{"textlike_boxes":boxes}]})


def photo():
    image=Image.new("RGB",(400,200),(105,115,125))
    ImageDraw.Draw(image).text((50,50),"PHYSICAL SIGN ABC 123",fill="white",font_size=25)
    return image


def test_unchanged_lettering_survives_small_jpeg_noise():
    before=photo();buffer=io.BytesIO();before.save(buffer,format="JPEG",quality=97,subsampling=0)
    after=Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    proof=OutputTextPreservation(detector([[.1,.2,.9,.5]])).inspect(before,after)
    assert proof["status"]=="unchanged"
    assert proof["not_a_watermark_absence_proof"] is True


def test_rewritten_or_missing_lettering_cannot_support_acceptance():
    before=photo();after=before.copy()
    ImageDraw.Draw(after).rectangle((40,40,360,100),fill=(105,115,125))
    ImageDraw.Draw(after).text((50,50),"PHYS1CAL FAKE XX 999",fill="white",font_size=25)
    proof=OutputTextPreservation(detector([[.1,.2,.9,.5]])).inspect(before,after)
    assert proof["status"]=="changed_text"
    assert any(not box["unchanged"] for box in proof["text_boxes"])


def test_absent_text_or_detector_error_never_provides_positive_witness():
    before=photo()
    assert OutputTextPreservation(detector([],"clear")).inspect(before,before)["status"]=="no_text_witness"
    assert OutputTextPreservation(detector([],"error")).inspect(before,before)["status"]=="error"


def test_unchanged_pixels_are_not_a_digital_overlay_classifier():
    image=photo()
    proof=OutputTextPreservation(detector([[.1,.2,.9,.5]])).inspect(image,image)
    assert proof["status"]=="unchanged" and proof["not_a_watermark_absence_proof"]
    assert "watermark_removed" not in proof

def test_physical_text_inquiry_cannot_waive_any_existing_nonwatermark_veto():
    from copy import deepcopy

    from watermark_studio.physical_text_review import eligible
    context={"background_preserved":True,"source_scene":"outdoor"}
    verdict={"status":"review","watermark_removed":False,"natural":True,"content_preserved":True,
        "uncertain":False,"guard_codes":[],"outside_repair_change":{"tiles":0},"residual_regions":[],
        "evidence_conflict":True,"model_explanation":{"status":"ok","assessment":"no_clear_defect","findings":[]}}
    assert eligible(verdict,context)
    for patch in [{"natural":False},{"content_preserved":False},{"uncertain":True},
                  {"guard_codes":["repair_rectangle_phase"]},{"guard_codes":["source_physical_ambiguity"]},
                  {"outside_repair_change":{"tiles":1}},{"residual_regions":[{"box":[0,0,.2,.2]}]},
                  {"evidence_conflict":False},{"model_explanation":{"status":"ok","assessment":"defect","findings":[{}]}}]:
        modified=deepcopy(verdict);modified.update(patch)
        assert not eligible(modified,context)
    assert not eligible(verdict,dict(context,source_scene="document"))
    assert not eligible(verdict,dict(context,background_preserved=False))

    assert not eligible(verdict,dict(context,source_carrier_ambiguous=True))
