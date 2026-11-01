"""Rare source-only reconciliation of excessively broad outdoor overlay masks."""
from __future__ import annotations

import json

from .cross_review import CLASSIFY_PROMPT
from .detection import DETECTION_PROMPT, locate_watermarks_batch, parse_detection, source_scene

SOURCE_RELOCALIZATION_PROMPT=DETECTION_PROMPT+(
    "\n这是原始照片的独立几何复核，不预设水印是否存在。"
    "只定位实际后期数字覆盖层的完整外接框，不把邻近场景一并圈入。"
    "设备操作规程、安全告示、企业标牌、证书、包装和衣物上的真实文字及背板、木框、"
    "设备边缘、物体表面均须保留，即使它们有蓝白标题、绿黄色装饰、密集黑字或矩形外形。"
    "相机悬浮信息卡及防伪码仍是数字覆盖层，不能把真实水印也豁免。"
    "不要为了缩小框而漏掉任何水印，也不按旧检测或固定角落猜坐标。")


def area(region):
    l,t,r,b=region.box
    return (r-l)*(b-t)


def broad_masks(detection):
    return [r for r in detection[0] if r.kind=="panel" and area(r)>=.35]


def _overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def _within(a,b):
    return b[0]-1e-7<=a[0]<a[2]<=b[2]+1e-7 and b[1]-1e-7<=a[1]<a[3]<=b[3]+1e-7


def _raw(text):
    text=text.strip()
    if text.startswith(chr(96)*3):
        text=text.split("\n",1)[1].rsplit(chr(96)*3,1)[0].strip()
    parse_detection(text)
    return json.loads(text)


def reconcile_broad(original,first,second):
    """Require one-to-one agreeing tighter components; never clip a detector."""
    broad=broad_masks(original)
    selected=[]
    for old in broad:
        left=[r for r in first[0] if _within(r.box,old.box)]
        right=[r for r in second[0] if _within(r.box,old.box)]
        if not left or len(left)!=len(right):
            raise ValueError("Source detectors disagree on component count")
        used=set()
        for one in left:
            if area(one)>=.35:
                raise ValueError("Source footprint remains ambiguous")
            matches=[(index,two) for index,two in enumerate(right) if index not in used and
                     _overlap(one.box,two.box)/(area(one)+area(two)-_overlap(one.box,two.box))>=.75]
            if len(matches)!=1:
                raise ValueError("Source detector geometry has no unique agreement")
            index,two=matches[0];used.add(index)
            if area(two)>=.35:
                raise ValueError("Source footprint remains ambiguous")
            raw=[]
            for detection,region in [(first,one),(second,two)]:
                raw.extend(c for c in parse_detection(detection[1],padding=0,relative_padding=0)
                           if _within(c.box,region.box))
            if not raw:
                raise ValueError("Missing raw source geometry")
            box=[min(c.box[0] for c in raw),min(c.box[1] for c in raw),
                 max(c.box[2] for c in raw),max(c.box[3] for c in raw)]
            selected.append({"kind":"panel" if one.kind=="panel" or two.kind=="panel" else "logo",
                             "box":[1000*v for v in box]})
    payload=_raw(original[1])
    kept=[item for item in payload["watermarks"] if not any(
        _within([v/1000 for v in item["box"]],old.box) for old in broad)]
    raw=json.dumps({"scene":source_scene(original[1]),"watermarks":kept+selected})
    return parse_detection(raw),raw



def _digital_claim(chats,encoder,processor,device):
    """Digital identity has its own uncertainty semantics, not missing carrier proof."""
    from .evidence import _generate_explanations
    from .quality import _structured_vqa
    content=chats[0][0]["content"][:-1]
    prompt=(CLASSIFY_PROMPT.split("只返回JSON。")[0]+
        "\n本次判断中心候选本身是数字叠加还是实物印字，不预设它属于哪一类。"
        "object_attachment和surface_evidence仅指承载候选文字的真实实物，不是它后面的场景。"
        "确认数字覆盖层须有可见正证据，例如悬浮叠加关系、半透明底板或不随场景透视的版式；"
        "不能只因为没找到载体就断言数字。若数字身份已有清楚证据，缺少实物载体不等于uncertain。"
        "只有分类看不清、证据冲突或混合实物/数字文字时uncertain=true。"
        "目标外标牌、设备和背景不能代替候选自身的载体证据。"
        "只返回JSON，physical_scene_text、digital_overlay、object_attachment、surface_evidence、uncertain"
        "为布尔值，area为none/top_left/top/top_right/left/center/right/bottom_left/bottom/bottom_right。")
    fields=("physical_scene_text","digital_overlay","object_attachment","surface_evidence","uncertain")
    rows,_=_structured_vqa([[{"role":"user","content":content+[{"type":"text","text":prompt}]}]],
                           fields,encoder,processor,device,128)
    # Flags are not supplied to the observation call; it must inspect the pixels.
    observed=_generate_explanations([[{"role":"user","content":content+[{"type":"text","text":
        "只用一句中文具体描述中心候选文字与真实场景的位置、边缘、表面、透视或透明叠加关系。"
        "不预设类别，不写是否通过，不复述布尔判定，不转录姓名地址日期代码。看不清就说明。"}]}]],
        encoder,processor,device,128,True)[0]
    if observed.get("status")!="summary_only":
        raise ValueError("Source digital identity lacks an observable description")
    return rows[0]|{"evidence":observed["summary"],"evidence_origin":"independent_visible_observation"}

def refine_broad_source(image,detection,editor,editor_processor,verifier,verifier_processor,device,
                        weights_relation):
    broad=broad_masks(detection)
    if not broad or source_scene(detection[1])!="outdoor" or weights_relation!="different_weights":
        return detection,[]
    audits=[{"box":list(r.box),"version":"source-broad-consensus-v1",
             "status":"uncertain_physical_carrier","requires_review":True} for r in broad]
    protected=([r for r in detection[0] if r not in broad],detection[1])
    try:
        first=locate_watermarks_batch([image],editor,editor_processor,device,1536,1,True,
                                      SOURCE_RELOCALIZATION_PROMPT)[0]
        second=locate_watermarks_batch([image],verifier,verifier_processor,device,1536,1,True,
                                       SOURCE_RELOCALIZATION_PROMPT)[0]
        if isinstance(first,Exception):raise first
        if isinstance(second,Exception):raise second
        for audit in audits:
            audit["source_detectors"]={"editor":[r.model_dump() for r in first[0]],
                                       "independent":[r.model_dump() for r in second[0]]}
        updated=reconcile_broad(detection,first,second)
        from .source_vetting import source_carrier_chats
        candidates=[r for r in updated[0] if any(_within(r.box,old.box) for old in broad)]
        if not candidates or len(candidates)>4:
            raise ValueError("Unbounded or absent replacement geometry")
        proofs=[]
        for region in candidates:
            chats=source_carrier_chats(image,region.box,context=.15)
            chats[0][0]["content"][-1]["text"]+=(
                "\n只核实标明的中心候选文字。目标外的实体牌、设备和背景，"
                "不能代替目标本身的物理载体证据。")
            reports=[_digital_claim(chats,model,processor,device) for model,processor in
                     [(editor,editor_processor),(verifier,verifier_processor)]]
            proofs.append({"box":list(region.box),"editor":reports[0],"independent":reports[1]})
            for audit in audits:audit["digital_proofs"]=list(proofs)
            if not all(r.get("digital_overlay") is True and r.get("physical_scene_text") is False
                       and r.get("object_attachment") is False and r.get("uncertain") is False for r in reports):
                raise ValueError("Tighter source component lacks dual digital-overlay proof")
        for audit in audits:
            audit.update(status="refined_digital_geometry",requires_review=False,
                         replacement_regions=[r.model_dump() for r in updated[0]],digital_proofs=proofs)
        return updated,audits
    except Exception as error:  # noqa: BLE001 - uncertain source regions are never blindly redrawn
        for audit in audits:audit["error"]=f"{type(error).__name__}: {error}"[:240]
        return protected,audits
