import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from watermark_studio.translucent_recovery import fit_transmission, recover_translucent_body


def test_physical_transmission_fit_ignores_opaque_text_outliers():
    rng=np.random.default_rng(8)
    background=rng.uniform(30,220,(400,3))
    observed=.6*background+.4*np.array([240,240,242])
    observed[:30]=0
    fit=fit_transmission(background,observed)
    assert fit["ok"]
    assert abs(fit["transmission"]-.6)<.01
    assert np.max(np.abs(np.array(fit["foreground"])-[240,240,242]))<1
    assert fit_transmission(background,np.ones_like(background)*235)["ok"] is False


def test_transmission_does_not_claim_an_opaque_or_colored_card_is_recoverable():
    rng=np.random.default_rng(9)
    x=rng.uniform(20,230,(400,3))
    assert not fit_transmission(x,.1*x+230)["ok"]
    assert not fit_transmission(x,.6*x+[5,30,100])["ok"]
    assert not fit_transmission(x,rng.uniform(20,230,x.shape))["ok"]
    assert not fit_transmission(x[:8],x[:8])["ok"]


def test_source_transmission_recovery_preserves_glyph_candidate_and_outside():
    rng=np.random.default_rng(10)
    yy,xx=np.mgrid[:600,:800]
    levels=np.clip(45+.25*yy+.025*xx+rng.normal(0,1,(600,800)),0,255)
    clean=np.stack([levels+5,levels,levels-5],axis=2).astype(np.uint8)
    source=clean.copy()
    source[120:480,160:560]=np.rint(.6*source[120:480,160:560]+.4*240).astype(np.uint8)
    photo=Image.fromarray(source)
    ImageDraw.Draw(photo).rectangle((160,120,559,200),fill=(30,100,220))
    ImageDraw.Draw(photo).text((220,300),"123 ABC",fill="black",stroke_width=1)
    candidate=Image.fromarray(clean).filter(ImageFilter.GaussianBlur(1))
    before=photo.tobytes()
    result,log=recover_translucent_body(photo,candidate,[.2,.2,.7,.8])
    assert log["applied"],log
    assert abs(log["combined_fit"]["transmission"]-.6)<.05
    assert np.abs(np.asarray(result)[250:400,180:500].astype(float)-clean[250:400,180:500]).mean()<3
    assert result.getpixel((5,5))==candidate.getpixel((5,5))
    assert photo.tobytes()==before
    assert result.size==photo.size


def test_local_alpha_support_survives_a_wrong_candidate_half_but_never_certifies_pixels():
    from watermark_studio.translucent_recovery import local_transmission_support
    rng = np.random.default_rng(117)
    clean = rng.uniform(30,220,(256,384,3))
    observed = .6*clean+.4*250
    candidate = clean.copy()
    candidate[:,:192] = rng.uniform(30,220,candidate[:,:192].shape)
    report = local_transmission_support(observed, candidate)
    assert report["supported"] and not report["quality_permission"]
    assert abs(report["transmission_median"]-.6) < .001
    assert all(tile["box"][0]>=192 for tile in report["tiles"])


def test_local_fit_rejects_opaque_wrong_identity_and_inconsistent_alpha():
    from watermark_studio.translucent_recovery import local_transmission_support
    rng = np.random.default_rng(118)
    clean = rng.uniform(30,220,(256,384,3))
    assert not local_transmission_support(np.full_like(clean,230),clean)["supported"]
    assert not local_transmission_support(rng.uniform(30,220,clean.shape),clean)["supported"]
    observed = .6*clean+.4*250
    observed[:,:192] = .4*clean[:,:192]+.6*250
    assert not local_transmission_support(observed,clean)["supported"]


def test_local_alpha_support_requires_several_well_separated_visible_tiles():
    from watermark_studio.translucent_recovery import local_transmission_support
    rng = np.random.default_rng(119)
    clean = rng.uniform(30,220,(256,384,3))
    observed = .6*clean+.4*250
    valid = np.zeros(clean.shape[:2],bool)
    valid[:64,:128] = True
    assert not local_transmission_support(observed,clean,valid=valid)["supported"]
    assert not local_transmission_support(observed[:32],clean[:32])["supported"]
