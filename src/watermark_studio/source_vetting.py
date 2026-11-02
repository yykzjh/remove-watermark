"""Source-only physical carrier veto; never approves generated outputs."""
from .cross_review import CLASSIFY_PROMPT, _classify, physical

SOURCE_CARRIER_CONTEXT = """这是尚未去水印的原图全貌与同位置局部，只用于载体检查，不能因原图也有就判为实物。
实物文字不要求现场手写、喷涂或雕刻：机械铭牌、品牌贴纸、包装印字、卷尺/测量杆刻度和印刷数字，
只要确实依附于原有物体，且能看到对应物体的材质、边缘或透视，就是实物内容。
不能因为它是印刷字、字体清晰、位置居中、面积很小或呈矩形就当成数字水印。
相机应用的悬浮信息卡、时间地点、品牌水印和防伪码依然是数字覆盖层；仅看似贴在物体上不构成实物证据。
依据原图可见载体判定，每个布尔字段均按实际观察填写；载体或表面证据不足必须uncertain。
"""


def source_carrier_chats(image,box,context=.9,*,wide=False):
    from .images import context_box, pixel_box
    from .quality import preview

    crop=context_box(pixel_box(box,image.size),image.size,context)
    if wide:
        x0,y0,x1,y1=pixel_box(box,image.size)
        pad=max(48,min((x1-x0)*.6,min(image.size)*.3))
        crop=(max(0,int(x0-pad)),max(0,int(y0-pad)),
              min(image.width,int(x1+pad)),min(image.height,int(y1+pad)))
    local=[(box[0]*image.width-crop[0])/(crop[2]-crop[0]),
           (box[1]*image.height-crop[1])/(crop[3]-crop[1]),
           (box[2]*image.width-crop[0])/(crop[2]-crop[0]),
           (box[3]*image.height-crop[1])/(crop[3]-crop[1])]
    instruction=SOURCE_CARRIER_CONTEXT
    if wide:
        instruction=instruction.replace("载体或表面证据不足必须uncertain。",
            "uncertain表示数字覆盖层与实物文字的身份仍无法确定。若声称实物，但载体或表面证据不足，必须uncertain=true。"
            "若明确观察到数字叠加层，则没有实物载体/表面是分类证据，不能仅因此标uncertain；"
            "仍需实际可见的数字覆盖层依据，不能因缺少实物证据就自动断言是水印。"
            "无法分辨文字是否贴在真实物体上时仍必须uncertain=true。")
    return [[{"role":"user","content":[
        {"type":"text","text":f"原图完整视野；只核实0..1坐标{list(box)}的候选文字。全图只用于辨认候选所附物体，不要把别处水印当成此候选。"},
        {"type":"image","image":preview(image,640)},
        {"type":"text","text":f"同一原图候选的局部，目标位于此局部0..1范围{[round(v,4) for v in local]}。观察实际物体材质、边缘、透视，不要求读出文字。"},
        {"type":"image","image":preview(image.crop(crop),640)},
        {"type":"text","text":CLASSIFY_PROMPT+"\n"+instruction}]}]]


def vet_small_panels(image, detection, editor, editor_processor, verifier, verifier_processor, device,
                     weights_relation, first_reports=None,*,recheck=False):
    regions,raw=detection
    audit=[]
    if weights_relation!="different_weights":
        return detection,[{"status":"disabled","reason":"distinct_weights_required"}]
    candidates=[r for r in regions if r.kind=="panel"
                and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])<.035]
    kept=list(regions)
    for region in candidates[:4]:
        record={"box":list(region.box),"status":"retained_for_edit","version":"source-carrier-global-context-v3"}
        try:
            chats=None
            if first_reports is None:
                chats=source_carrier_chats(image,region.box)
                first=_classify(chats,editor,editor_processor,device)[0]
                record["primary_batch_size"]=1
            else:
                first,record["primary_batch_size"]=first_reports[tuple(region.box)]
                if isinstance(first,Exception):
                    raise first
            record["editor"]=first
            if physical(first) or first.get("object_attachment") or first.get("physical_scene_text"):
                if chats is None:
                    chats=source_carrier_chats(image,region.box)
                second=_classify(chats,verifier,verifier_processor,device)[0]
                record["independent"]=second
                if physical(first) and physical(second):
                    kept.remove(region)
                    record["status"]="protected_physical_text"
                else:
                    proof=None
                    if recheck:
                        try:
                            chats=source_carrier_chats(image,region.box,wide=True)
                            again=_classify(chats,editor,editor_processor,device)[0]
                            independent=_classify(chats,verifier,verifier_processor,device)[0]
                            proof={"editor":again,"independent":independent,"view":"wide_original_context_v1"}
                            record["carrier_recheck"]=proof
                        except Exception as error:  # noqa: BLE001 - failed evidence keeps the original protected
                            record["carrier_recheck"]={"error":f"{type(error).__name__}: {error}"[:200]}
                    if proof and all(confirmed_digital(proof[k]) for k in ("editor","independent")):
                        record["status"]="confirmed_digital_recheck"
                    else:
                        kept.remove(region)
                        record["status"]="uncertain_physical_carrier"
                        record["requires_review"]=True
        except Exception as error:  # noqa: BLE001 - ambiguity cannot authorize erasing possible physical text
            record["error"]=f"{type(error).__name__}: {error}"[:300]
            prior=record.get("editor",{})
            if physical(prior) or prior.get("object_attachment") or prior.get("physical_scene_text"):
                if region in kept:
                    kept.remove(region)
                record.update(status="uncertain_physical_carrier",requires_review=True)
        audit.append(record)
    return (kept,raw),audit


def vet_sources(images,detections,editor,editor_processor,verifier,verifier_processor,device,
                weights_relation,batch_size=1,*,recheck=False):
    """Batch only independent editor-side source claims; keep per-image carrier proofs."""
    if len(images)!=len(detections):
        raise ValueError("Source vet image/detection counts differ")
    from .source_geometry import refine_broad_source
    broad_audits={}
    prepared=[]
    for index,(image,detection) in enumerate(zip(images,detections)):
        if isinstance(detection,Exception):
            prepared.append(detection)
            continue
        updated,audit=refine_broad_source(image,detection,editor,editor_processor,verifier,
            verifier_processor,device,weights_relation)
        prepared.append(updated);broad_audits[index]=audit
    detections=prepared
    cache=[{} for _ in images]
    jobs=[]
    if batch_size>1 and weights_relation=="different_weights":
        for index,(image,detection) in enumerate(zip(images,detections)):
            if isinstance(detection,Exception):
                continue
            candidates=[r for r in detection[0] if r.kind=="panel"
                        and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])<.035]
            for region in candidates[:4]:
                jobs.append((index,region.box,source_carrier_chats(image,region.box)[0]))
        def run(group):
            failure=None
            try:
                reports=_classify([j[2] for j in group],editor,editor_processor,device)
                if len(reports)!=len(group):
                    raise ValueError("Source carrier classification count mismatch")
                for (index,box,_),report in zip(group,reports):
                    cache[index][tuple(box)]=(report,len(group))
                return
            except Exception as error:  # noqa: BLE001 - split before retaining an explicit per-candidate error
                failure=f"{type(error).__name__}: {error}"[:300]
            if "out of memory" in failure.lower():
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            if len(group)>1:
                middle=len(group)//2
                run(group[:middle]);run(group[middle:])
            else:
                index,box,_=group[0]
                cache[index][tuple(box)]=(RuntimeError(failure),1)
        for start in range(0,len(jobs),batch_size):
            run(jobs[start:start+batch_size])
    output=[];audits={}
    for index,(image,detection) in enumerate(zip(images,detections)):
        if isinstance(detection,Exception):
            output.append(detection)
            continue
        result,audit=vet_small_panels(image,detection,editor,editor_processor,verifier,
            verifier_processor,device,weights_relation,cache[index] if batch_size>1 and
            weights_relation=="different_weights" else None,recheck=recheck)
        output.append(result);audits[index]=broad_audits.get(index,[])+audit
    return output,audits


def confirmed_digital(report):
    """All fields need explicit agreement; missing/uncertain never grants edit authority."""
    return (report.get("digital_overlay") is True
            and all(report.get(k) is False for k in
                    ("physical_scene_text","object_attachment","surface_evidence","uncertain")))
