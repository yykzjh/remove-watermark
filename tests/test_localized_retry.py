from pathlib import Path

import pytest
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.retry import localized_repair_supported, process_attempts, retry_allowed


def quality(status="review", **extra):
    return {"status":status,"watermark_removed":True,"natural":False,"content_preserved":True,
            "uncertain":True}|extra


def measured():
    return quality(repair_boundary={"suspect_segments":[
        {"box":[.1,.6,.4,.9],"side":"top","luma_step":12}]})


@pytest.mark.parametrize("q",[
    quality(watermark_removed=False,natural=True,residual_regions=[]),
    quality(status="error"),
    quality(evidence=[{"source":"vlm","issue":"blur_smear","box":[.1,.1,.3,.3],"certainty":"uncertain"}]),
    quality(evidence=[{"source":"vlm","issue":"physical_text","box":[.1,.1,.3,.3],"certainty":"visible"}]),
    measured()|{"guard_codes":["source_physical_ambiguity"]},
])
def test_unknown_or_physical_evidence_never_grants_regeneration(q):
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_retry_localized=True,
                          quality_max_attempts=0)
    assert not localized_repair_supported(q)
    assert not retry_allowed({"quality":q},opts,1)
    assert q["status"]!="passed"


def test_only_originally_localized_residuals_can_use_scan_as_repair_evidence():
    q=quality(residual_regions=[{"box":[.8,.8,.95,.95]}],source_regions=[{"box":[0,.7,.5,1]}])
    assert not localized_repair_supported(q)
    q["source_regions"]=[{"box":[.75,.75,1,1]}]
    assert localized_repair_supported(q)
    assert localized_repair_supported(measured())


def test_successful_first_candidates_are_written_once_and_not_copied(tmp_path,monkeypatch):
    src=tmp_path/"source.png";dst=tmp_path/"out.png"
    Image.new("RGB",(12,12),"blue").save(src)
    before=src.read_bytes()
    def process(tasks,*args):
        assert tasks[0]["destination"]==str(dst)
        Image.new("RGB",(12,12),"gray").save(dst)
        return [{"file":"a.png","ok":True,"quality":quality("passed",natural=True,uncertain=False)}]
    monkeypatch.setattr("watermark_studio.retry.atomic_copy",lambda *a,**k:pytest.fail("successful first pass must not be copied"))
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_retry_localized=True)
    row=process_attempts([{"source":str(src),"destination":str(dst),"relative":"a.png"}],opts,
        DeploymentConfig(backend="mock",state_dir=str(tmp_path/"state")),None,{},"job",process)[0]
    assert row["quality_attempts"]==1 and row["quality"]["status"]=="passed"
    assert dst.is_file() and src.read_bytes()==before


def test_direct_rejected_first_is_retained_and_retry_uses_original(tmp_path):
    source=tmp_path/"source.png";dest=tmp_path/"out.png"
    Image.new("RGB",(16,12),"blue").save(source);before=source.read_bytes()
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_retry_localized=True,quality_max_attempts=2)
    cfg=DeploymentConfig(backend="mock",state_dir=str(tmp_path/"state"),retain_rejected_first=True)
    seen=[]
    def process(tasks,settings,*args):
        seen.append(tasks[0]["source"])
        is_retry=tasks[0].get("_retry",False)
        Image.new("RGB",(16,12),"green" if is_retry else "gray").save(tasks[0]["destination"])
        return [{"file":"a.png","ok":True,"model_inferences":1,
                 "quality":quality("passed",natural=True,uncertain=False) if is_retry else measured()}]
    first=process_attempts([{"source":str(source),"destination":str(dest),"relative":"a.png"}],
                           opts,cfg,None,{},"job",process)[0]
    retained=Path(first["first_rejected_candidate"]);retained_bytes=retained.read_bytes()
    assert retained_bytes==dest.read_bytes() and "_retry_task" in first
    final=process_attempts([first["_retry_task"]],opts,cfg,None,{},"job",process)[0]
    assert final["quality"]["status"]=="passed" and final["quality_attempts"]==2
    assert retained.read_bytes()==retained_bytes and dest.read_bytes()!=retained_bytes
    assert source.read_bytes()==before and seen==[str(source),str(source)]


def test_unlocalized_rejection_stays_review_without_blind_recheck(tmp_path):
    source=tmp_path/"source.png";dest=tmp_path/"out.png"
    Image.new("RGB",(8,8),"blue").save(source)
    opts=ProcessingConfig(quality_check=True,quality_auto_retry=True,quality_retry_localized=True)
    def process(tasks,*args):
        Image.new("RGB",(8,8),"gray").save(tasks[0]["destination"])
        return [{"file":"a.png","ok":True,"quality":quality(watermark_removed=False,natural=True,residual_regions=[])}]
    row=process_attempts([{"source":str(source),"destination":str(dest),"relative":"a.png"}],opts,
        DeploymentConfig(backend="mock",state_dir=str(tmp_path/"state"),retain_rejected_first=True),
        None,{},"job",process)[0]
    assert row["quality"]["status"]=="review" and "_retry_task" not in row
    assert row["retry_blocked"]=="unlocalized_or_ambiguous_evidence"
    assert Path(row["first_rejected_candidate"]).is_file()


def test_localized_retry_is_opt_in_and_needs_verification_and_retry():
    assert ProcessingConfig().quality_retry_localized is False
    with pytest.raises(ValueError):
        ProcessingConfig(quality_retry_localized=True)
