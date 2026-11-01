"""Source-only physical carrier veto; never approves generated outputs."""
from .cross_review import _classify, dispute_chats, physical


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
        record={"box":list(region.box),"status":"retained_for_edit"}
        try:
            chats=dispute_chats(image,image,[region.box])
            first=_classify(chats,editor,editor_processor,device)[0]
            record["editor"]=first
            if physical(first):
                second=_classify(chats,verifier,verifier_processor,device)[0]
                record["independent"]=second
                if physical(second):
                    kept.remove(region)
                    record["status"]="protected_physical_text"
        except Exception as error:  # noqa: BLE001 - ambiguity cannot discharge a detected overlay
            record["error"]=f"{type(error).__name__}: {error}"[:300]
        audit.append(record)
    return (kept,raw),audit
