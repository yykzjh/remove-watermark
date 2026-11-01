"""Read-only provenance helpers for held-out image evaluation."""
import json
import re


def exclusion_sources(root, base, explicit):
    """Explicit benchmark arguments must never replace the old development exclusions."""
    automatic=[base.name]
    old128=root/"first-pass-holdout-neutral-v1"
    if (old128/"sources.json").is_file():
        automatic.append(old128.name)
    # Every completed preparation is a development/inspection exposure, even
    # without an inference report. Discover its provenance instead of relying
    # on a growing, manually maintained list in a qualification command.
    for provenance in sorted(root.glob("*/sources.json")):
        folder = provenance.parent
        if folder.is_symlink() or not folder.resolve().is_relative_to(root.resolve()):
            continue
        if folder.name not in automatic:
            automatic.append(folder.name)
    seen=set()
    for name in dict.fromkeys([*automatic,*explicit]):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+",name):
            raise ValueError("Invalid excluded benchmark name")
        values=json.loads((root/name/"sources.json").read_text())
        if isinstance(values, list) and all(
                isinstance(v, dict) and isinstance(v.get("path"), str) for v in values):
            # Earlier visual-review manifests retain group/size alongside path.
            values = [v["path"] for v in values]
        if not isinstance(values,list) or any(not isinstance(v,str) for v in values):
            raise ValueError("Benchmark provenance must contain source path strings")
        seen.update(values)
    return seen,automatic
