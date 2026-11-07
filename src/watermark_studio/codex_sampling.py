"""Reproducible early sampling, with an exact final sample and no discarded grades."""
from __future__ import annotations

import hashlib
import json
from collections import deque

PROTOCOL = "input-frame-two-stage-sha256-v1"


def priority(name, seed, stage):
    value = json.dumps([PROTOCOL, seed, stage, name], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode()).digest()


def make_sampling_plan(paths, sample_size=1000, seed=20261106):
    paths = list(paths)
    if len(paths) != len(set(paths)) or any(not isinstance(p, str) for p in paths):
        raise ValueError("Sampling frame must contain unique relative image paths")
    if type(sample_size) is not int or sample_size < 0 or type(seed) is not int:
        raise ValueError("Sampling size must be nonnegative and seed must be an integer")
    encoded = json.dumps(sorted(paths), ensure_ascii=False, separators=(",", ":")).encode()
    return {"protocol": PROTOCOL, "population": len(paths), "sample_size": sample_size, "seed": seed,
            "frame_sha256": hashlib.sha256(encoded).hexdigest(),
            "primary": sorted(paths, key=lambda p: (priority(p, seed, "primary"), p))[:sample_size]}


def validate_sampling_plan(plan):
    if plan.get("protocol") != PROTOCOL:
        raise ValueError("Unknown streaming sampling protocol")
    size, population, seed = plan.get("sample_size"), plan.get("population"), plan.get("seed")
    if (type(size) is not int or size < 0 or type(population) is not int or population < 0
            or type(seed) is not int):
        raise ValueError("Invalid streaming sampling dimensions")
    primary = plan.get("primary", [])
    if (len(primary) != min(size, population) or len(set(primary)) != len(primary)
            or any(not isinstance(p, str) for p in primary)
            or sorted(primary, key=lambda p: (priority(p, seed, "primary"), p)) != primary):
        raise ValueError("Invalid frozen primary candidates")
    digest = plan.get("frame_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid sampling frame identity")


def final_sample(eligible, plan):
    """Keep every early passed candidate; uniformly top up from the remaining passed pool.

    The primary draw is uniform over the complete input frame. Conditional on its
    passed count, its intersection is uniform over passed images; the independent
    top-up draw therefore produces the requested uniform final sample.
    """
    validate_sampling_plan(plan)
    eligible = set(eligible)
    selected = [name for name in plan["primary"] if name in eligible]
    remainder = eligible - set(selected)
    count = min(plan["sample_size"], len(eligible)) - len(selected)
    return selected + sorted(remainder, key=lambda p: (priority(p, plan["seed"], "top-up"), p))[:count]


def interleave(*prepared):
    active = deque(map(iter, prepared))
    while active:
        items = active.popleft()
        try:
            item = next(items)
        except StopIteration:
            continue
        yield item
        active.append(items)
