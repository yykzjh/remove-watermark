"""Source-only physical carrier veto; never approves generated outputs."""
from .cross_review import _classify, dispute_chats, physical

SOURCE_CARRIER_CONTEXT = """这是尚未去水印的同一原图位置，两份图相同只用于载体检查，不能因像素相同就判为实物。
实物文字不要求现场手写、喷涂或雕刻：机械铭牌、品牌贴纸、包装印字、卷尺/测量杆刻度和印刷数字，
只要确实依附于原有物体，且能看到对应物体的材质、边缘或透视，就是实物内容。
不能因为它是印刷字、字体清晰、位置居中、面积很小或呈矩形就当成数字水印。
相机应用的悬浮信息卡、时间地点、品牌水印和防伪码依然是数字覆盖层；仅看似贴在物体上不构成实物证据。
依据原图可见载体判定，每个布尔字段均按实际观察填写；载体或表面证据不足必须uncertain。
"""


def vet_small_panels(image, detection, editor, editor_processor, verifier, verifier_processor, device,
                     weights_relation):
    regions,raw=detection
    audit=[]
    if weights_relation!="different_weights":
        return detection,[{"status":"disabled","reason":"distinct_weights_required"}]
    candidates=[r for r in regions if r.kind=="panel"
                and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])<.035]
    kept=list(regions)
    for region in candidates[:4]:
        record={"box":list(region.box),"status":"retained_for_edit","version":"source-carrier-v2"}
        try:
            chats=dispute_chats(image,image,[region.box])
            chats[0][0]["content"][-1]["text"]+="\n"+SOURCE_CARRIER_CONTEXT
            first=_classify(chats,editor,editor_processor,device)[0]
            record["editor"]=first
            if physical(first) or first.get("object_attachment") or first.get("physical_scene_text"):
                second=_classify(chats,verifier,verifier_processor,device)[0]
                record["independent"]=second
                if physical(first) and physical(second):
                    kept.remove(region)
                    record["status"]="protected_physical_text"
                else:
                    kept.remove(region)
                    record["status"]="uncertain_physical_carrier"
                    record["requires_review"]=True
        except Exception as error:  # noqa: BLE001 - ambiguity cannot discharge a detected overlay
            record["error"]=f"{type(error).__name__}: {error}"[:300]
        audit.append(record)
    return (kept,raw),audit
