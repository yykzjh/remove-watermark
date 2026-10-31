"""Small token-fragment automaton for bounded overlay JSON, without a new dependency."""
from __future__ import annotations

_CACHE = {}


def overlay_grammar(tokenizer, eos_ids, scene=False):
    key=(id(tokenizer),tuple(eos_ids))
    if key not in _CACHE:
        roots={}
        def add(stage,text,next_stage):
            node=roots.setdefault(stage,{})
            for token in tokenizer.encode(text,add_special_tokens=False):
                node=node.setdefault(token,{})
            node["_next"]=next_stage
        add("header",'{"watermarks":[',"list")
        add("scene_header",'{"scene":',"scene")
        for value in ("outdoor","document","indoor","unknown"):
            add("scene",'"'+value+'","watermarks":[',"list")
        add("list","]}","done")
        add("list",'{"kind":',"kind")
        add("kind",'"panel","box":[',"n0")
        add("kind",'"logo","box":[',"n0")
        for index in range(4):
            for number in range(1001):
                add("n"+str(index),str(number)+("," if index<3 else "]}"),
                    "n"+str(index+1) if index<3 else "separator")
        add("separator",",","object")
        add("separator","]}","done")
        add("object",'{"kind":',"kind")
        add("finish","]}","done")
        _CACHE[key]=(tokenizer,roots)
    roots=_CACHE[key][1]
    def allowed(suffix):
        if any(token in eos_ids for token in suffix):
            return list(eos_ids)  # Finished batch rows can contain padding after EOS.
        stage="scene_header" if scene else "header"
        node=roots[stage]; groups=0
        for token in suffix:
            if stage=="done":
                if token not in eos_ids: raise ValueError("Tokens after overlay JSON")
                continue
            node=node[token]
            if "_next" in node:
                stage=node["_next"]
                if stage=="separator":
                    groups+=1
                    if groups==8: stage="finish"
                if stage!="done": node=roots[stage]
        return list(eos_ids) if stage=="done" else [token for token in node if isinstance(token,int)]
    return allowed
