import hashlib

import pytest

from watermark_studio.trial_sources import unique_sources


def test_parallel_original_partitions_cannot_leak_duplicates_or_previous_exposure(tmp_path):
    paths=[]
    for index,value in enumerate([b"same original",b"same original",b"forbidden",b"new a",b"new b",b"new c"]):
        path=tmp_path/f"{index}.jpg";path.write_bytes(value);paths.append(path)
    blocked={hashlib.sha256(b"forbidden").hexdigest()}
    rows,_=unique_sources(paths,blocked,4)
    assert len({r["sha256"] for r in rows})==4
    assert not {r["sha256"] for r in rows}&blocked
    first,second=rows[:2],rows[2:]
    assert not {r["sha256"] for r in first}&{r["sha256"] for r in second}
    with pytest.raises(ValueError,match="Insufficient"):unique_sources(paths,blocked,5)
