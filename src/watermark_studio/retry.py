"""Per-image original-based retry attempts with durable best-candidate selection."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .images import atomic_copy, load_photo


def quality_rank(result):
    """Conservative ordering, never trade visible content for removal alone."""
    q = result.get("quality", {})
    return (
        bool(result.get("ok")),
        q.get("status") == "passed",
        q.get("content_preserved") is True,
        "outside_repair_change" not in q.get("guard_codes", []),
        "new_flat_fill" not in q.get("guard_codes", []),
        "repair_boundary_step" not in q.get("guard_codes", []),
        "repair_texture_collapse" not in q.get("guard_codes", []),
        q.get("watermark_removed") is True,
        q.get("natural") is True,
        # Equal rejected candidates: prefer an explicitly empty residual scan.
        # Never override content/naturalness guards or promote review to passed.
        q.get("residual_regions") == [],
        q.get("status") != "error",
        q.get("uncertain") is False,
    )



def unsupported_residual_dispute(quality):
    """No evidence for a destructive re-edit; recheck once and retain review if disputed."""
    explanation = quality.get("model_explanation", {})
    return (
        quality.get("status") == "review"
        and quality.get("watermark_removed") is False
        and quality.get("natural") is True
        and quality.get("content_preserved") is True
        and quality.get("uncertain") is False
        and not quality.get("guard_codes")
        and quality.get("residual_regions") == []
        and explanation.get("status") == "ok"
        and explanation.get("assessment") == "no_clear_defect"
        and not any(f.get("issue") != "physical_text" for f in explanation.get("findings", []))
    )

def retry_allowed(result, options, attempt):
    return (
        options.quality_auto_retry and result.get("quality", {}).get("status") in {"review", "error"}
        and (options.quality_max_attempts == 0 or attempt < options.quality_max_attempts)
    )


def process_attempts(tasks, options, config, pipe, batch_limits, job, process):
    """Runtime schedules rounds; this function performs exactly one attempt per input."""
    if not options.quality_auto_retry:
        return process(tasks, options, config, pipe, batch_limits)
    # Retry feedback belongs to one photograph. Never share it across a batch.
    if len(tasks) > 1 and any(t.get("_retry") for t in tasks):
        return [result for task in tasks for result in process_attempts(
            [task], options, config, pipe, batch_limits, job, process)]
    outputs = []
    # Keep normal first-pass batching; retries have compatible per-request settings.
    prepared, records = [], []
    for task in tasks:
        if task.get("reuse_existing") and not task.get("_retry") and not options.overwrite:
            records.append((task, None, None))
            prepared.append(task)
            continue
        token = hashlib.sha256(task["relative"].encode()).hexdigest()
        folder = Path(config.state_dir).expanduser() / job / "candidates" / token
        folder.mkdir(parents=True, exist_ok=True)
        suffix = Path(task["destination"]).suffix
        candidate = folder / ("candidate" + suffix)
        best = folder / ("best" + suffix)
        history = folder / "attempts.json"
        saved = json.loads(history.read_text()) if history.exists() else {"attempts": [], "best": None}
        verify_existing = bool(task.get("reuse_existing") and not task.get("_retry"))
        if verify_existing:
            atomic_copy(task["destination"], candidate, True)
        prepared.append(task | {"destination": str(candidate), "reuse_existing": verify_existing})
        records.append((task, (folder, candidate, best, history), saved))
    attempt = int(tasks[0].get("_attempt", 1))
    settings = options.model_copy(deep=True)
    settings.overwrite = True
    if tasks[0].get("_retry"):
        settings.seed = (settings.seed + 104729 * (attempt - 1)) % (2**32)
        settings.resolution = options.quality_retry_resolution
        settings.quality_batch_size = 1
        if not (options.scope == "regions" and options.preset == "custom"):
            settings.scope, settings.preset, settings.regions = "regions", "auto", []
        settings.extra_prompt = (
            options.extra_prompt + "\n精确清除所有数字覆盖层及淡色文字、边框和小图标。"
            "只自然重建被遮挡区域，保留原来可见的人体、测量读数及真实印刷文字；"
            "避免模糊、重复纹理、直线错位、衣物和肢体变形。"
        ).strip()
        from .evidence import REPAIR_ACTIONS, feedback_from_quality
        validated = feedback_from_quality({"evidence": tasks[0].get("_retry_feedback", [])})
        guidance = "\n".join(dict.fromkeys(REPAIR_ACTIONS[item["issue"]] for item in validated))
        if guidance:
            settings.extra_prompt += ("\n上轮检验定位的返工重点（具体位置在当前裁切图提示中给出）：\n" + guidance +
                                      "\n始终从原图重做，只修改确认的数字覆盖层；不能为通过检验删除真实内容或降低标准。")
        # A verifier error first rechecks the best candidate instead of regenerating.
        if tasks[0].get("_recheck"):
            prepared[0]["destination"] = str(records[0][1][2])
            prepared[0]["reuse_existing"] = True
    results = process(prepared, settings, config, pipe, batch_limits)
    for (task, paths, saved), result in zip(records, results):
        if paths is None:
            result["retry_blocked"] = "existing_output_read_only"
            result["quality_attempts"] = 1
            outputs.append(result)
            continue
        folder, candidate, best, history = paths
        current_attempt = int(task.get("_attempt", 1))
        q = result.get("quality", {})
        if task.get("_retry"):
            result["repair_feedback_used"] = task.get("_retry_feedback", [])
            result["repair_feedback_attempt"] = task.get("_feedback_attempt")
        if (config.backend != "mock" and result["ok"] and q.get("status") == "review"
                and q.get("watermark_removed") is False and "residual_regions" not in q):
            # Only rejected candidates pay this extra scan. Never turn a rejection into a pass.
            from .detection import locate_watermarks
            started = time.monotonic()
            try:
                regions, _ = locate_watermarks(
                    load_photo(best if task.get("_recheck") else candidate)[0],
                    pipe._quality_encoder, pipe._quality_processor, pipe._execution_device,
                    max(1024, options.quality_resolution),
                )
                q["residual_regions"] = [r.model_dump() for r in regions]
            except Exception as error:  # noqa: BLE001 - retry hints are not acceptance evidence
                q["retry_hint_error"] = str(error)[:400]
            q["retry_hint_seconds"] = round(time.monotonic()-started, 3)
            q["seconds"] = round(q.get("seconds", 0)+q["retry_hint_seconds"], 3)
        from .evidence import refresh_evidence
        if q.get("status") in {"review", "error", "passed"}:
            refresh_evidence(q)
        if unsupported_residual_dispute(q):
            q["retry_recheck_only"] = True
            q["retry_recheck_reason"] = "unsupported_residual_dispute"
        record = {"attempt": current_attempt, "quality": result.get("quality"),
                  "ok": result["ok"], "action": result.get("action"), "error": result.get("error"),
                  "repair_feedback_used": result.get("repair_feedback_used", []),
                  "repair_feedback_attempt": result.get("repair_feedback_attempt")}
        saved["attempts"].append(record)
        chosen_path = best if task.get("_recheck") else candidate
        previous = saved["best"]
        if result["ok"] and (previous is None or quality_rank(result) > quality_rank(previous)):
            if chosen_path != best:
                atomic_copy(chosen_path, best, True)
            saved["best"] = result
        selected = dict(saved["best"] or result)
        selected.update(quality_attempts=current_attempt, attempt_history=saved["attempts"][-20:],
                        best_attempt=saved.get("best_attempt", 1),
                        quality_total_seconds=round(sum(
                            (r.get("quality") or {}).get("seconds", 0)
                            for r in saved["attempts"]), 3))
        if saved["best"] is result:
            saved["best_attempt"] = current_attempt
            selected["best_attempt"] = current_attempt
        if saved["best"] is not None:
            atomic_copy(best, task["destination"], options.overwrite or bool(task.get("_retry")))
        selected["cross_review_attempted_now"] = bool(q.get("cross_review",{}).get("attempted"))
        selected["cross_review_corrected_now"] = q.get("cross_review",{}).get("status")=="corrected"
        selected["cross_review_error_now"] = q.get("cross_review",{}).get("status")=="error"
        again = retry_allowed(selected, options, current_attempt)
        if (again and task.get("_protect_disputed_text") and task.get("_recheck")
                and selected.get("quality",{}).get("status") in {"review","error"}):
            again = False
            selected["retry_blocked"] = (
                "unresolved_residual_dispute"
                if selected.get("quality", {}).get("retry_recheck_reason") == "unsupported_residual_dispute"
                else "unresolved_physical_text_dispute"
            )
        selected["retry_exhausted"] = bool(
            not again and selected.get("quality", {}).get("status") in {"review", "error"}
            and not selected.get("retry_blocked")
        )
        if again:
            from .config import Region
            from .detection import merge_regions
            from .evidence import feedback_from_quality
            # Model coordinates are suggestions only; the source scan must confirm them.
            model_hints = [{"kind":"logo", "box":item["box"]} for item in q.get("evidence", [])
                           if item.get("source") == "vlm" and item.get("certainty") == "visible"
                           and item.get("issue") in {"residual_text", "residual_background"} and item.get("box")]
            hints = task.get("_extra_regions", []) + q.get("residual_regions", []) + model_hints
            try:
                hints = [r.model_dump() for r in merge_regions([
                    Region.model_validate(r) for r in hints
                ])]
            except ValueError:
                # Keep the previously validated hints if the union is implausibly large.
                hints = task.get("_extra_regions", [])
            selected["_retry_task"] = task | {
                "_extra_regions": hints,
                "_retry_feedback": feedback_from_quality(selected.get("quality", {})),
                "_feedback_attempt": selected["best_attempt"],
                "_cross_review_used": bool(task.get("_cross_review_used") or q.get("cross_review",{}).get("attempted")),
                "_protect_disputed_text": bool(task.get("_protect_disputed_text") or q.get("retry_recheck_only")),
                "_retry": True, "_attempt": current_attempt + 1,
                "_recheck": selected.get("quality", {}).get("status") == "error" or bool(
                    selected.get("quality", {}).get("retry_recheck_only") or q.get("retry_recheck_only") or task.get("_protect_disputed_text")),
            }
        temp = history.with_suffix(".tmp")
        temp.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        temp.replace(history)
        if not again:
            best.unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)
        outputs.append(selected)
    return outputs
