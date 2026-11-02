from types import SimpleNamespace

import numpy as np
import pytest

from watermark_studio.config import Region
from watermark_studio.denoising_context import pixel_edit_masks


def test_independent_pixel_masks_have_no_cross_image_regions():
    items = [{"_denoising_context_regions": [Region(box=(0, 0, .25, .5))]},
             {"_denoising_context_regions": [Region(box=(.75, .5, 1, 1))]}]
    values = pixel_edit_masks(items, (80, 60))
    assert values.shape == (2, 60, 80)
    assert values[0, :30, :20].sum() == 600
    assert values[1, 30:, 60:].sum() == 600
    assert np.count_nonzero(values[0]*values[1]) == 0
    with pytest.raises(ValueError):
        pixel_edit_masks([items[0], {}], (80, 60))


def test_full_flow_preserves_each_original_at_actual_sigma_and_keeps_editable_values():
    torch = pytest.importorskip("torch")
    from watermark_studio.denoising_context import DenoisingContext

    items = [{"_denoising_context_regions": []},
             {"_denoising_context_regions": [Region(box=(0, 0, 1, 1))]}]
    state = DenoisingContext(items, 64, 64, 8)
    noise = torch.stack([torch.ones((64, 4)), torch.ones((64, 4))*2])
    clean = torch.stack([torch.ones((64, 4))*3, torch.ones((64, 4))*7])
    clean_before = clean.clone()
    state.capture(noise, clean)
    state.schedule(torch.tensor([.8, .35, 0.]))
    assert torch.allclose(noise[0], torch.ones((64, 4))*1.4)
    assert torch.equal(noise[1], torch.ones((64, 4))*2)
    pipe = SimpleNamespace(scheduler=SimpleNamespace(step_index=1))
    result = state.after_step(pipe, 0, 0, {"latents": torch.ones_like(noise)*11})["latents"]
    assert torch.allclose(result[0], clean[0]*.65+torch.ones((64, 4))*.35)
    assert torch.equal(result[1], torch.ones((64, 4))*11)
    pipe.scheduler.step_index = 2
    result = state.after_step(pipe, 1, 0, {"latents": torch.ones_like(noise)*13})["latents"]
    assert torch.equal(result[0], clean[0]) and torch.equal(result[1], torch.ones((64, 4))*13)
    assert torch.equal(clean, clean_before)
    state.finish()
    assert all(item["_denoising_context_log"]["steps"] == 2 for item in items)
    assert all(item["_denoising_context_log"]["extra_vae_encodes"] == 0 for item in items)


def test_wrong_reference_layout_or_scheduler_fails_closed():
    torch = pytest.importorskip("torch")
    from watermark_studio.denoising_context import DenoisingContext

    state = DenoisingContext([{"_denoising_context_regions": []}], 64, 64, 8)
    with pytest.raises(ValueError, match="matching full-frame"):
        state.capture(torch.ones((1, 64, 4)), torch.ones((1, 63, 4)))
    state.capture(torch.ones((1, 64, 4)), torch.zeros((1, 64, 4)))
    with pytest.raises(ValueError, match="Descending flow"):
        state.schedule(torch.tensor([.5, .8, 0.]))
    state.schedule(torch.tensor([1., .5, 0.]))
    with pytest.raises(ValueError, match="position"):
        state.after_step(SimpleNamespace(scheduler=SimpleNamespace(step_index=2)), 0, 0,
                         {"latents": torch.ones((1, 64, 4))})
    with pytest.raises(ValueError, match="Incomplete"):
        state.finish()


def test_explicit_latent_context_buffer_does_not_modify_the_pixel_region_definition():
    import copy

    import pytest
    torch=pytest.importorskip("torch")
    from watermark_studio.denoising_context import DenoisingContext
    original={"_denoising_context_regions":[{"kind":"panel","box":[.4,.4,.5,.5]}]}
    items=[copy.deepcopy(original)]
    narrow=DenoisingContext(items,128,128,16)
    wide_items=[copy.deepcopy(original)|{"_denoising_context_margin":3}]
    wide=DenoisingContext(wide_items,128,128,16)
    noise=torch.zeros(1,64,64);reference=torch.ones_like(noise)
    narrow.capture(noise.clone(),reference);wide.capture(noise.clone(),reference)
    assert narrow.margin_cells==1 and wide.margin_cells==3
    assert torch.all(wide.mask>=narrow.mask) and wide.mask.sum()>narrow.mask.sum()
    assert items[0]["_denoising_context_regions"]==wide_items[0]["_denoising_context_regions"]==original["_denoising_context_regions"]
    for value in (True,-1,17,2.0):
        with pytest.raises(ValueError):DenoisingContext([original|{"_denoising_context_margin":value}],128,128,16)
    with pytest.raises(ValueError):
        DenoisingContext([original,original|{"_denoising_context_margin":3}],128,128,16)
