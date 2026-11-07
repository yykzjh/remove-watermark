import copy

import pytest

from watermark_studio.codex_sampling import (
    final_sample,
    interleave,
    make_sampling_plan,
    validate_sampling_plan,
)


def test_input_frame_draw_is_order_independent_and_passed_intersection_is_never_discarded():
    frame = [f"town/village/{n}.jpg" for n in range(5000)]
    plan = make_sampling_plan(frame)
    assert make_sampling_plan(reversed(frame)) == plan
    assert len(plan["primary"]) == len(set(plan["primary"])) == 1000
    passed = {p for n, p in enumerate(frame) if n % 10}
    sampled = final_sample(passed, plan)
    assert len(sampled) == len(set(sampled)) == 1000
    assert set(plan["primary"]) & passed <= set(sampled) <= passed
    assert final_sample(sorted(passed, reverse=True), plan) == sampled
    assert make_sampling_plan(frame, seed=7)["primary"] != plan["primary"]


def test_final_top_up_keeps_all_initial_candidates_and_handles_small_populations():
    frame = [f"{n}.jpg" for n in range(20)]
    plan = make_sampling_plan(frame, 8)
    initial = set(plan["primary"][:3])
    passed = initial | (set(frame) - set(plan["primary"]))
    sample = final_sample(passed, plan)
    assert len(sample) == 8 and initial <= set(sample)
    assert set(final_sample(initial, plan)) == initial
    assert final_sample(passed, make_sampling_plan(frame, 0)) == []


def test_invalid_or_rebound_frame_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        make_sampling_plan(["same", "same"])
    with pytest.raises(ValueError, match="nonnegative"):
        make_sampling_plan(["a"], -1)
    plan = make_sampling_plan(["a", "b", "c"], 2)
    broken = copy.deepcopy(plan)
    broken["primary"] *= 2
    with pytest.raises(ValueError, match="primary"):
        validate_sampling_plan(broken)


def test_two_groups_interleave_lazily_without_unbounded_preparation():
    seen = []
    def tasks(prefix, count):
        for n in range(count):
            seen.append((prefix, n))
            yield prefix + str(n)
    prepared = interleave(tasks("bad", 2), tasks("passed", 3))
    assert seen == []
    assert next(prepared) == "bad0" and seen == [("bad", 0)]
    assert list(prepared) == ["passed0", "bad1", "passed1", "passed2"]


def test_frozen_streaming_sample_cannot_change_its_frame(tmp_path):
    from watermark_studio.codex_photo_audit import freeze_plan
    records = [{"file": f"{n}.jpg", "quality_status": "passed", "sha256": str(n)} for n in range(100)]
    frame = make_sampling_plan([r["file"] for r in records], 20)
    records[0]["quality_status"] = "review"
    plan = freeze_plan(records, tmp_path, sample_size=20, minimum_server_attempts=3,
                       stream_sampling_plan=frame)
    assert len(plan["sample"]) == 20
    assert set(frame["primary"]) - {"0.jpg"} <= set(plan["sample"])
    assert plan["deferred_server_review"] == ["0.jpg"]
    with pytest.raises(ValueError, match="frame changed"):
        freeze_plan(records, tmp_path, sample_size=20,
                    stream_sampling_plan=make_sampling_plan(["different"], 20))
    with pytest.raises(ValueError, match="frozen"):
        freeze_plan(records, tmp_path, sample_size=20, minimum_server_attempts=3)
