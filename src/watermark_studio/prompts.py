"""Short, spatially constrained instructions for the supplied outdoor photo styles."""

BASE_PROMPT = """Edit this photograph only to remove the digitally overlaid watermark.
Remove the entire {target}, including all text, digits, logos, colored backgrounds,
translucent gray or white panels, rounded borders, shadows and faint residual strokes.
The watermark occupies {location} in this crop. Reconstruct the scene hidden beneath it
using the immediately surrounding visual evidence. Continue the same soil, gravel,
concrete path, grass, weeds, fallen leaves, branches or construction materials where
these surfaces meet the removed overlay. Match their local scale, direction,
perspective, irregular texture, focus, lighting, color, noise and compression.
Continue existing edges and shadows smoothly across the repaired area. Every area formerly covered by the watermark must be naturally filled, never left
blank, flat-colored, blurred or transparent. When the overlay hides part of a human
body, continue the visible body silhouette, limb position, clothing folds, trouser
legs, hands or shoes with plausible anatomy and proportions. Use only the visible
person as context; avoid missing, extra, duplicated or disconnected body parts.
For covered soil, grass, trees and other plants, naturally extend the existing
terrain, blades of grass, tree trunks, roots, branches and foliage, with coherent
depth, edge continuity and contact shadows. Do not insert unrelated vegetation.
If the overlay crosses a person's visible clothing, legs or shoes, continue only
that existing person's shape and fabric consistently with the visible parts. Preserve every person's identity,
pose and proportions and every existing object, building, road edge and plant.
Do not add people, objects, machinery, signage, writing, symbols, logos or decorative
details. Do not beautify, clean up, sharpen, recolor, relight, crop, rotate or redesign
the scene. Keep all unoccluded content and the original camera viewpoint unchanged.
The result must look like the same ordinary outdoor site photograph without the
digital overlay, with natural texture and no rectangular patch, smear or repeated pattern."""

TARGETS = {
    "panel": "detected site-information panel (blue header with white Chinese text and/or translucent pale-gray body with black Chinese text)",
    "logo": (
        "entire detected camera-app watermark cluster: the large white Chinese "
        'characters "今日水印" on the top line, every line and tag beneath them, '
        'including "相机", "真实时间", "防伪码", any timestamp or alphanumeric '
        "verification code, short horizontal separator dashes, small icons, "
        "outlines, shadows, and gray, black or translucent label backgrounds. "
        "Remove the complete cluster, including any portions at image edges "
        "where it extends there; do not stop after removing only the top title"
    ),
    "custom": "specified digital text or information-panel overlay",
}


def build_prompt(
    kind: str, box: tuple[int, int, int, int], crop: tuple[int, int, int, int], extra: str = ""
) -> str:
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    x0, y0, x1, y1 = box
    location = (
        f"x={100 * (x0 - crop[0]) / cw:.0f}% to {100 * (x1 - crop[0]) / cw:.0f}%, "
        f"y={100 * (y0 - crop[1]) / ch:.0f}% to {100 * (y1 - crop[1]) / ch:.0f}% "
        "(origin at top left)"
    )
    prompt = BASE_PROMPT.format(target=TARGETS.get(kind, TARGETS["custom"]), location=location)
    return prompt + (
        "\nAdditional scene-specific instruction: " + extra.strip() if extra.strip() else ""
    )
