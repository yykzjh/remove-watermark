from PIL import Image

from watermark_studio.glyph_hypotheses import overlapping_physical_candidates


class Detector:
    def __init__(self,glyphs):
        self.glyphs=glyphs
    def inspect(self,image,boxes):
        return {"status":"text_present","regions":[{"textlike_boxes":self.glyphs}]}


def test_mixed_coarse_hypothesis_can_only_propose_unchanged_outside_glyphs():
    before=Image.new("RGB",(1000,1000),"gray")
    proposals=[{"issue":"physical_text","box":[.3,.65,.55,.72]}]
    verdict={"source_regions":[{"kind":"panel","box":[0.,.6,.4,1.]}]}
    found=overlapping_physical_candidates(before,before,proposals,verdict,Detector([[.44,.66,.5,.70]]))
    assert len(found["findings"])==1
    assert found["findings"][0]["box"][0]>.416
    assert found["quality_permission"] is False
    changed=before.copy();changed.paste("black",(440,660,500,700))
    result=overlapping_physical_candidates(before,changed,proposals,verdict,Detector([[.44,.66,.5,.70]]))
    assert not result["findings"] and result["reason"]=="original_glyph_or_pixel_identity_not_established"


def test_any_residual_glyph_vetoes_whole_set_and_residual_claim_cannot_move():
    before=Image.new("RGB",(1000,1000),"gray")
    proposals=[{"issue":"physical_text","box":[.3,.65,.55,.72]}]
    verdict={"source_regions":[{"kind":"panel","box":[0.,.6,.4,1.]}]}
    found=overlapping_physical_candidates(before,before,proposals,verdict,
        Detector([[.44,.66,.5,.70],[.35,.66,.38,.70]]))
    assert not found["findings"] and found["reason"]=="output_glyph_intersects_original_edit"
    proposals[0]["issue"]="residual_text"
    found=overlapping_physical_candidates(before,before,proposals,verdict,Detector([[.44,.66,.5,.70]]))
    assert not found["findings"] and found["reason"]=="explicit_bounded_physical_hypotheses_only"


def test_refinement_cannot_jump_to_an_adjacent_unchanged_label():
    before=Image.new("RGB",(1000,1000),"gray")
    proposals=[{"issue":"physical_text","box":[.3,.65,.55,.72]}]
    verdict={"source_regions":[{"kind":"panel","box":[0.,.6,.4,1.]}]}
    found=overlapping_physical_candidates(before,before,proposals,verdict,Detector([[.54,.66,.6,.70]]))
    assert not found["findings"] and found["reason"]=="native_glyph_escapes_coarse_hypothesis"
    assert len(found["source_rgb_sha256"])==64 and found["quality_permission"] is False
