"""Experimental semantic clarification; identical images and downstream QA vetoes."""
from __future__ import annotations

OUTPUT_PROMPT = """图1是完整结果照片，后续图是相同结果的放大局部，裁切和白边不属于照片缺陷。
只按可见证据检查整张结果：
overlay_present：仍存在后期数字水印则为true，包括蓝白或灰黑信息卡、黄底文字、绿蓝应用图标、
今日水印、相机、日期地点、防伪码、细码、透明底板或残影。真实物体上的标牌、手持验收牌、
尺子刻度、合格证、包装袋文字、条码、机械品牌、衣服文字和施工标记不属于数字水印。
repair_defect：修补不自然则为true，包括接缝、矩形填块或明暗板、模糊抹痕、重复纹理、
断裂工程线条、人体畸形、异常绿紫杂点或规则颗粒。人物手脚、泥水、植被、履带印、模板钢筋、
原本的物体和阴影必须合理连续，不能将工地美化成光滑地面。
cannot_assess：细节不足或证据含糊、无法可靠判断时为true。
若没有可见数字水印，overlay_present=false；若没有修补异常，repair_defect=false。
根据实际照片判断，不能因为提示词提到某类缺陷便假设照片存在该缺陷。看不清必须cannot_assess=true。
只返回JSON：overlay_present、repair_defect、cannot_assess为布尔值；
area为主要问题在完整照片的位置，none/top_left/top/top_right/left/center/right/bottom_left/bottom/bottom_right。"""

REFERENCE_PROMPT = """图1为原照片，图2为结果；后续局部和辅助图均有来源标签。
只验收结果，不把原图已有水印或辅助图处理效果当成结果缺陷。
visible_content_changed：原来可见的场景内容发生变化时为true。检查人物数量、脸、姿势、服装、
机械、建筑、道路沟渠边线、树木、位置以及测量刻度读数、钢筋根数间距、手持验收牌、合格证、
包装印字、临时标记和真实投影。删除数字水印、合理重建原来不透明水印完全挡住且不可见的部分、
均匀缩放或压缩本身不属于可见内容变化；不能要求未知遮挡内历史像素一致。
repair_defect：结果有矩形暗板亮板、低纹理矩形、模糊断层、接缝、材质断层、工程线条断裂
或人体异常时为true。真实阴影须有原未遮挡部分的连续性或可见遮挡几何支持，不能凭空猜测开脱。
cannot_assess：细节不足、证据含糊或关键测量证书被水印挡住无法核实时为true，不得臆造可靠读数。
若可见场景一致，visible_content_changed=false；若修补自然，repair_defect=false。
只返回JSON：visible_content_changed、repair_defect、cannot_assess为布尔值；
area为主要问题在完整照片的位置，none/top_left/top/top_right/left/center/right/bottom_left/bottom/bottom_right。"""


def inspect_defects(chats, kind, encoder, processor, device, max_tokens):
    from .quality import CONTENT_PROMPT as legacy_reference
    from .quality import OUTPUT_PROMPT as legacy_output
    from .quality import _structured_vqa
    if kind not in {"output", "reference"}:
        raise ValueError("Unknown inspection kind")
    first = "overlay_present" if kind == "output" else "visible_content_changed"
    canonical = "watermark_removed" if kind == "output" else "content_preserved"
    fields = (first, "repair_defect", "cannot_assess")
    prepared = []
    for chat in chats:
        # Copy dictionaries only: never duplicate or transform any evidence image.
        messages = [dict(message, content=[dict(item) for item in message["content"]]) for message in chat]
        tail = messages[-1]["content"][-1]
        if tail.get("type") != "text":
            raise ValueError("Inspection must end with its protocol")
        # Scene constraints (when present) are retained after the original protocol.
        text = tail["text"]
        original = legacy_reference if kind == "reference" else legacy_output.replace(
            "这是一张去水印结果的检验画布：上方为整图，下方（如有）为同一结果图的放大细节。",
            "图1是完整结果照片，后续图是标明原照片坐标的放大局部。结合整图透视和物体关系区分真实印刷文字。")
        if not text.startswith(original):
            raise ValueError("Unknown original inspection instructions")
        suffix = text[len(original):]
        tail["text"] = (OUTPUT_PROMPT if kind == "output" else REFERENCE_PROMPT) + suffix
        prepared.append(messages)
    values, seconds = _structured_vqa(prepared, fields, encoder, processor, device, max_tokens)
    results = []
    for value in values:
        if any(type(value.get(field)) is not bool for field in fields):
            raise ValueError("Incomplete defect evidence")
        results.append({canonical: not value[first], "natural": not value["repair_defect"],
                        "uncertain": value["cannot_assess"], "area": value["area"],
                        "raw_defect_checks": value})
    return results, seconds
