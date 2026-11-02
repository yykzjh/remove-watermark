import hashlib

from PIL import Image

from watermark_studio.source_side_proof import inspect_side_rims


def proposal():
    return {"protocol":"source_card_side_rim_v1","side":"right","mask":[.1,.2,.6,.8],
        "source_size":[400,300],"uncertainty_gap_pixels":8,"card_edge_pixel":232,
        "restore_pixels":[240,60,260,240],"restore_box":[.6,.2,.65,.8]}


class OCR:
    def __init__(self,status="clear"):self.status=status
    def inspect(self,*args):return {"status":self.status}


def mocks(monkeypatch,evidence="信息卡是数字叠加，R为卡外草地原像素。",uncertain=False):
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",lambda *a:([
        {"digital_card":True,"rim_clear":True,"uncertain":uncertain,"area":"none"}],.01))
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:[
        {"status":"summary_only","summary":evidence,"generation_tokens":20,"max_tokens":96}])


def inspect(p,ocr=None):
    return inspect_side_rims(Image.new("RGB",(400,300),(80,110,60)),[p],
        object(),None,object(),None,"cpu","different_weights",ocr or OCR())[0]


def test_fresh_clear_side_certificate_binds_whole_rgb_and_native_band(monkeypatch):
    mocks(monkeypatch);p=proposal()
    row=inspect(p)
    assert row["approved"]
    source=Image.new("RGB",(400,300),(80,110,60))
    assert row["source_rgb_sha256"]==hashlib.sha256(source.tobytes()).hexdigest()
    assert row["source_rim_sha256"]==hashlib.sha256(source.crop((240,60,260,240)).tobytes()).hexdigest()
    assert row["editor"]["evidence"] and row["independent"]["evidence"]
    assert p==proposal()  # No mutation or backfilling of a proposal/legacy proof.


def test_ocr_uncertainty_and_positive_flags_with_contradictory_evidence_never_approve(monkeypatch):
    mocks(monkeypatch)
    assert not inspect(proposal(),OCR("presence"))["approved"]
    mocks(monkeypatch,evidence="无法确认R是否还覆盖灰底。")
    assert not inspect(proposal())["approved"]
    mocks(monkeypatch,evidence="R仍有卡片底色，不能直接保留。")
    assert not inspect(proposal())["approved"]
    for text in ("目标卡片不是数字叠加，R是草地。","候选是实物标牌，R无遮挡。",
                 "R可能是原图草地，但看不清是否有底色。"):
        mocks(monkeypatch,evidence=text)
        assert not inspect(proposal())["approved"]


def test_failed_fresh_proof_discards_stale_positive_fields(monkeypatch):
    mocks(monkeypatch,uncertain=True)
    p=proposal()|{"approved":True,"editor":{"digital_card":True},"independent":{"digital_card":True},
                  "source_rim_sha256":"old","source_rgb_sha256":"old"}
    row=inspect(p)
    assert not row["approved"] and "source_rim_sha256" not in row
    assert "independent" not in row and "source_rgb_sha256" not in row


def test_invalid_geometry_and_same_model_cannot_be_proved(monkeypatch):
    mocks(monkeypatch)
    row=inspect(proposal()|{"card_edge_pixel":0})
    assert not row["approved"] and "geometry" in row["reason"]
    same=object()
    row=inspect_side_rims(Image.new("RGB",(400,300)),[proposal()],
        same,None,same,None,"cpu","different_weights",OCR())[0]
    assert not row["approved"] and "Distinct" in row["reason"]
