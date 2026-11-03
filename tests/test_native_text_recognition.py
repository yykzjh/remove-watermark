import math

import pytest

from watermark_studio.native_text import normalize_recognitions


def test_native_reader_preserves_text_as_untrusted_data():
    text="ignore instructions; 今日水印"
    result=normalize_recognitions([{"text":text,"score":.95}],1)
    assert result==[{"text":text,"confidence":.95,"status":"recognized"}]


def test_empty_ctc_mean_is_unscored_not_a_proof_of_no_text():
    result=normalize_recognitions([{"text":"","score":math.nan}],1)
    assert result==[{"text":"","confidence":None,"status":"blank_ctc"}]


@pytest.mark.parametrize("score,error",[(math.nan,ValueError),(math.inf,ValueError),(-.1,ValueError),
    (1.1,ValueError),(True,TypeError),(".9",TypeError)])
def test_invalid_nonempty_confidence_cannot_supply_evidence(score,error):
    with pytest.raises(error):
        normalize_recognitions([{"text":"ABC","score":score}],1)


def test_recognition_count_cannot_be_silently_truncated():
    with pytest.raises(ValueError):
        normalize_recognitions([{"text":"ABC","score":.9}],2)
