"""Source geometry and mask support of the experimental auxiliary route."""
import numpy as np
import pytest
from PIL import Image

from watermark_studio.tiny_inpaint import masked_inputs, tiny_eligible


def example():
    image = Image.new("RGB", (80, 40), "green")
    return {"state": {"source": Image.new("RGB", (1000, 1000)), "whole_generation": False,
                      "source_scene": "outdoor"}, "box": (10, 20, 40, 40),
            "crop": (0, 0, 80, 40), "image": image, "resolution": 512, "size": (83, 43)}


def test_mask_uses_arbitrary_source_coordinates_and_half_open_box():
    source, mask = masked_inputs(example())
    assert source.shape == (48, 88, 3) and mask.shape == (48, 88)
    assert mask[25, 20] and not mask[25, 50] and not mask[5, 20]
    assert np.all(source[:, :, 1] > source[:, :, 0])


def test_route_excludes_whole_large_and_physical_documents():
    edit = example()
    assert tiny_eligible(edit)
    edit["state"]["whole_generation"] = True
    assert not tiny_eligible(edit)
    edit["state"]["whole_generation"] = False
    edit["state"]["source_scene"] = "document"
    assert not tiny_eligible(edit)
    edit["state"]["source_scene"] = "outdoor"
    edit["box"] = (0, 0, 500, 500)
    assert not tiny_eligible(edit)


def test_geometry_mismatch_is_an_error():
    edit = example()
    edit["crop"] = (0, 0, 79, 40)
    with pytest.raises(ValueError):
        masked_inputs(edit)


def test_white_stencil_preserves_visible_gaps_and_excludes_outside_box():
    from watermark_studio.tiny_inpaint import white_glyph_support

    rgb=np.full((80,120,3),65,dtype=np.uint8)
    rgb[25:55,35:38]=255
    rgb[25:55,65:68]=255
    mask,method=white_glyph_support(Image.fromarray(rgb),(20,10,100,70))
    actual=np.asarray(mask)
    assert method=="white_glyph_source_stencil"
    assert actual[40,36] and actual[40,66]
    assert not actual[40,50] and not actual[:10].any() and not actual[:,100:].any()


def test_white_background_is_ambiguous_and_never_claimed_to_be_a_text_stencil():
    from watermark_studio.tiny_inpaint import white_glyph_support

    mask,method=white_glyph_support(Image.new("RGB",(80,60),"white"),(10,10,70,50))
    assert method=="box_ambiguous_white_coverage"
    assert np.count_nonzero(np.asarray(mask))==60*40


def test_native_stencil_composition_does_not_resample_uncovered_source_pixels():
    from watermark_studio.tiny_inpaint import native_stencil_composite, white_glyph_support

    edit=example()
    rng=np.random.default_rng(12)
    rgb=rng.integers(20,100,(40,80,3),dtype=np.uint8)
    rgb[25:35,20:23]=255
    edit["image"]=Image.fromarray(rgb)
    fixed=native_stencil_composite(edit,Image.new("RGB",(32,16),"red"))
    mask,_=white_glyph_support(edit["image"],edit["box"])
    outside=np.asarray(mask)==0
    assert np.array_equal(np.asarray(fixed)[outside],rgb[outside])
    assert fixed.size==edit["image"].size
