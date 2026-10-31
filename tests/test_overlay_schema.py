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


class MergedPunctuation:
    def __init__(self):
        self.pieces=['":[]}',': []}',']}]}' ,']},{"kind":"', '{"kind":"', '":[', '":[]']
        self.tokens={piece:10000+i for i,piece in enumerate(self.pieces)}
    def encode(self,text,add_special_tokens=False):
        result=[]
        while text:
            match=next((piece for piece in self.pieces if text.startswith(piece)),None)
            if match:
                result.append(self.tokens[match]);text=text[len(match):]
            else:
                result.append(ord(text[0]));text=text[1:]
        return result


@pytest.mark.parametrize("scene",[False,True])
@pytest.mark.parametrize("count",[0,1,2,8])
def test_grammar_keeps_natural_bpe_paths_for_empty_lists_and_joint_closing_tokens(scene,count):
    tokenizer=MergedPunctuation()
    value={"scene":"document","watermarks":[]} if scene else {"watermarks":[]}
    value["watermarks"]=[{"kind":"logo","box":[0,900,1000,1000]}]*count
    tokens=tokenizer.encode(json.dumps(value,separators=(",",":")))
    grammar=overlay_grammar(tokenizer,[0],scene)
    prefix=[]
    for token in tokens:
        assert token in grammar(prefix),prefix
        prefix.append(token)
    assert grammar(prefix)==[0]
