import threading
import time

import pytest

from watermark_studio.codex_pool import PhotoTaskError, photo_map


def test_four_independent_photos_run_together_without_exceeding_bound():
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    running, peak = 0, 0
    def work(value):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        barrier.wait(timeout=3)
        time.sleep(.01)
        with lock:
            running -= 1
        return value * 2
    result = dict(photo_map(((str(n), n) for n in range(8)), work, 4))
    assert peak == 4
    assert result == {str(n): n * 2 for n in range(8)}


def test_handoff_finishes_all_active_photos_and_does_not_start_next_batch():
    barrier, stop = threading.Barrier(4), threading.Event()
    started = []
    def work(value):
        started.append(value)
        barrier.wait(timeout=3)
        if value == 0:
            stop.set()
        time.sleep(.02)
        return value
    result = dict(photo_map(((str(n), n) for n in range(12)), work, 4, stop=stop.is_set))
    assert set(started) == set(range(4))
    assert result == {str(n): n for n in range(4)}


def test_worker_failure_drains_other_results_before_propagating_without_resubmitting():
    barrier = threading.Barrier(4)
    def work(value):
        barrier.wait(timeout=3)
        if value == 0:
            raise RuntimeError("provider request failed")
        time.sleep(.03)
        return value
    completed = []
    with pytest.raises(PhotoTaskError, match="provider request failed"):
        for name, result in photo_map(((str(n), n) for n in range(12)), work, 4):
            completed.append((name, result))
    assert set(completed) == {(str(n), n) for n in range(1, 4)}


def test_duplicate_photo_cannot_enter_two_independent_sessions():
    with pytest.raises(PhotoTaskError, match="Duplicate photo"):
        list(photo_map([("same", 1), ("same", 2)], lambda value: value, 4))


@pytest.mark.parametrize("workers", [0, 17, True, 1.5])
def test_invalid_concurrency_is_rejected(workers):
    with pytest.raises(ValueError, match="concurrency"):
        list(photo_map([], lambda value: value, workers))
