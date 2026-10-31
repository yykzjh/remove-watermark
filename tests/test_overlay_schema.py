import json

import pytest

from watermark_studio.overlay_schema import overlay_grammar


class Characters:
    def encode(self,text,add_special_tokens=False):
        return [ord(c) for c in text]


def feed(text,scene=False):
    grammar=overlay_grammar(Characters(),[0],scene)
    prefix=[]
    for token in map(ord,text):
        assert token in grammar(prefix),text[:len(prefix)+1]
        prefix.append(token)
    return grammar(prefix)


@pytest.mark.parametrize("value",[
    {"watermarks":[]},
    {"watermarks":[{"kind":"logo","box":[0,910,1000,1000]}]},
    {"watermarks":[{"kind":"panel","box":[12,30,600,800]}]*8},
])
def test_complete_json_is_legal_but_never_forces_a_detection(value):
    assert feed(json.dumps(value,separators=(",",":")))==[0]


def test_scene_choices_are_bounded_and_keep_geometry_json_valid():
    text='{"scene":"document","watermarks":[{"kind":"panel","box":[0,600,400,1000]}]}'
    assert feed(text,True)==[0]


def test_ninth_group_and_non_json_text_are_impossible():
    text=json.dumps({"watermarks":[{"kind":"logo","box":[0,0,10,10]}]*8},separators=(",",":"))
    assert ord(",") not in feed(text[:-2])
    assert ord("S") not in overlay_grammar(Characters(),[0])([])


def test_coordinates_cannot_exceed_the_normalized_integer_range():
    grammar=overlay_grammar(Characters(),[0])
    prefix=list(map(ord,'{"watermarks":[{"kind":"logo","box":[1000'))
    assert ord("1") not in grammar(prefix)
    assert ord(".") not in grammar(prefix)


def test_finished_batch_rows_tolerate_decoder_padding_after_eos():
    grammar=overlay_grammar(Characters(),[0])
    assert grammar(list(map(ord,'{"watermarks":[]}'))+[0,999])==[0]
