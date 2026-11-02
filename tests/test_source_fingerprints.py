import numpy as np
from PIL import Image, PngImagePlugin

from watermark_studio.source_fingerprints import (
    FingerprintIndex,
    compare_fingerprints,
    fingerprint_file,
)


def photo():
    y,x=np.mgrid[:192,:256]
    values=np.stack([80+x*.4+8*np.sin(y*.1),90+y*.5+12*np.cos(x*.1),100+x*.2+y*.25],axis=-1)
    return Image.fromarray(np.clip(np.rint(values),0,255).astype(np.uint8))


def test_metadata_aliases_are_exact_pixels_and_exif_display_rotation_is_normalized(tmp_path):
    original=photo();first=tmp_path/"a.png";second=tmp_path/"b.png";rotated=tmp_path/"r.png"
    original.save(first);info=PngImagePlugin.PngInfo();info.add_text("version","different")
    original.save(second,pnginfo=info)
    exif=Image.Exif();exif[274]=6
    original.transpose(Image.Transpose.ROTATE_90).save(rotated,exif=exif)
    a,b,c=[fingerprint_file(p) for p in (first,second,rotated)]
    assert a["encoded_sha256"]!=b["encoded_sha256"]
    assert compare_fingerprints(a,b)["relation"]=="decoded_rgb_equal"
    assert compare_fingerprints(a,c)["relation"]=="decoded_rgb_equal"


def test_recompression_is_a_candidate_but_different_colors_are_not(tmp_path):
    image=photo();first=tmp_path/"a.png";second=tmp_path/"b.jpg";different=tmp_path/"different.png"
    image.save(first);image.save(second,quality=93)
    Image.fromarray(np.asarray(image)[:,::-1].copy()).save(different)
    a,b,c=[fingerprint_file(p) for p in (first,second,different)]
    assert compare_fingerprints(a,b)["relation"]=="near_duplicate_candidate"
    assert compare_fingerprints(a,c) is None
    b["phash64"]="ffffffffffffffff";b["dhash64"]="ffffffffffffffff"
    index=FingerprintIndex();index.add({**a,"partition":"history"})
    matches=index.matches(b)
    assert len(matches)==1 and matches[0][1]["relation"]=="near_duplicate_candidate"


def test_history_inventory_excludes_sealed_reservation_but_includes_api_sources_and_reports_missing(tmp_path):
    import json
    import runpy
    from pathlib import Path

    script=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/audit-evaluation-isolation.py"))
    root=tmp_path/"root";root.mkdir();raw=root/"八标照片集";raw.mkdir()
    first=raw/"first.png";second=raw/"second.png";secret=raw/"sealed.png"
    for path in (first,second,secret):photo().save(path)
    old=root/"prior";old.mkdir();missing=raw/"missing.jpg"
    (old/"sources.json").write_text(json.dumps([str(first),str(missing)]))
    sealed=root/"sealed";sealed.mkdir();(sealed/"sources.json").write_text(json.dumps([str(secret)]))
    state=tmp_path/"state";job=state/("a"*32);job.mkdir(parents=True)
    (job/"summary.json").write_text(json.dumps({"input_dir":str(raw)}))
    (job/"results.jsonl").write_text(json.dumps({"file":"second.png"})+"\n")
    paths,_,unresolved,missing_paths=script["discover"](root,sealed,state)
    assert set(paths)=={str(first),str(second)}
    assert str(secret) not in paths and not unresolved
    assert missing_paths==[str(missing)]
