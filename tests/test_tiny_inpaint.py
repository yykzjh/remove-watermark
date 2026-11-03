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
