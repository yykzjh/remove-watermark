"""Bounded independent photo tasks; callers alone write shared journals and status."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


class PhotoTaskError(RuntimeError):
    pass


def photo_map(prepared, work, concurrency=1, stop=lambda: False, progress=lambda active: None):
    if type(concurrency) is not int or not 1 <= concurrency <= 16:
        raise ValueError("Codex concurrency must be an integer from 1 to 16")
    items = iter(prepared)
    seen, active = set(), {}
    exhausted, failure = False, None
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="codex-photo") as pool:
        while active or not exhausted:
            while not exhausted and failure is None and not stop() and len(active) < concurrency:
                try:
                    name, payload = next(items)
                    if name in seen:
                        raise ValueError("Duplicate photo in concurrent work: " + name)
                    seen.add(name)
                except StopIteration:
                    exhausted = True
                    break
                except Exception as error:  # noqa: BLE001 - drain other photos, then re-raise
                    failure = ("preparing photo", error)
                    break
                if stop():
                    exhausted = True
                    break
                active[pool.submit(work, payload)] = name
                progress(tuple(active.values()))
            if stop() or failure is not None:
                exhausted = True
            if not active:
                break
            completed, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED)
            for future in completed:
                name = active.pop(future)
                try:
                    value = future.result()
                except Exception as error:  # noqa: BLE001 - drain other photos, then re-raise
                    failure = failure or (name, error)
                    exhausted = True
                else:
                    # Persist each completed photo before preparing additional work.
                    yield name, value
                progress(tuple(active.values()))
    if failure is not None:
        name, error = failure
        raise PhotoTaskError("Individual Codex task failed: " + name + ": " + str(error)) from error
