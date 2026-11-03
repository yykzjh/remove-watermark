"""Offline evidence-first reviewer experiment; default online protocol is unchanged."""
from __future__ import annotations

import json
import time

FIELDS = {"watermark_removed", "content_preserved", "natural", "uncertain"}
EVIDENCE_PROMPT = """
补充检验协议：先用图像证据作观察记录，再给布尔结论。
observations字符串最多80个汉字，禁止冗长逐图复述。只简要记录：实际可读的数字覆盖文字/图案和整图位置；若是实物印字，
记录支持它的实体载体、透视和连接关系；若无数字覆盖则写未见。逐项记录可见修补异常或未见异常。
只能引用当前结果照片中的实际证据，不把提示词中的水印例子当成图像内容，
也不把原图的水印、裁切坐标和检验排版当成结果残留。看不清或载体分类冲突则uncertain=true。
微弱文字、只剩底纹、无字暗板亮板仍是不合格；有实体印字也不能豁免其他数字覆盖。
原图已可见的内容发生改变必须不合格。检验标准保持不变。
输出一个JSON对象，顺序为observations（简短字符串），然后原协议要求的所有布尔字段和area。
不要输出思考标签、Markdown或JSON以外的文本。"""

def parse_grounded(text,fields):
    value=json.loads(text.strip())
    if (not isinstance(value,dict) or not isinstance(value.get("observations"),str)
            or not value["observations"].strip() or len(value["observations"])>1600):
        raise ValueError("Grounded inspection requires bounded visual observations")
    from .quality import _verdict_json
    result=_verdict_json(json.dumps(value),fields)
    result["grounded_observations"]=value["observations"]
    return result

def inspect_grounded(chats,fields,encoder,processor,device,max_tokens):
    import torch
    if not set(fields)<=FIELDS:
        raise ValueError("Unknown grounded review fields")
    started=time.monotonic()
    prepared=[]
    for chat in chats:
        messages=[dict(m,content=[dict(item) for item in m["content"]]) for m in chat]
        tail=messages[-1]["content"][-1]
        if tail.get("type")!="text":
            raise ValueError("Inspection must end in instructions")
        tail["text"]+=EVIDENCE_PROMPT+(
            "\n必须输出的完整字段名："+", ".join(("observations",*fields,"area"))+
            "。每个布尔字段必须是JSON true或false；area必须是位置枚举中的字符串，"
            "没有缺陷用字符串none，禁止null或省略area。先写observations，再写这些字段，不加其他字段。")
        prepared.append(messages)
    tokenizer=processor.tokenizer;previous=tokenizer.padding_side
    try:
        tokenizer.padding_side="left"
        inputs=processor.apply_chat_template(prepared,tokenize=True,add_generation_prompt=True,
            processor_kwargs={"padding":True},return_dict=True,return_tensors="pt")
        inputs.pop("token_type_ids",None);inputs=inputs.to(device)
        width=inputs.input_ids.shape[1]
        with torch.inference_mode():
            generated=encoder.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=max(384,max_tokens))
        texts=processor.batch_decode(generated[:,width:],skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        if len(texts)!=len(chats):
            raise ValueError("Grounded response count mismatch")
        rows=[parse_grounded(t,fields) for t in texts]
        return rows,time.monotonic()-started
    finally:
        tokenizer.padding_side=previous
