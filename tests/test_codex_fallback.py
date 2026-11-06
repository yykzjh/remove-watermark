import pytest

from watermark_studio.codex_fallback import allowed_fallback, server_dispositions


def test_only_final_failed_matching_candidates_with_enough_attempts_enter_fallback():
    row = {"sha256": "pinned"}
    proof = {"sha256": "pinned", "status": "review", "attempts": 3}
    assert allowed_fallback(row, proof, 3)
    assert not allowed_fallback(row, proof | {"attempts": 1}, 3)
    assert not allowed_fallback(row, proof | {"status": "passed"}, 3)
    assert allowed_fallback(row, proof | {"attempts": 1}, 0)
    with pytest.raises(ValueError, match="pinned"):
        allowed_fallback(row, proof | {"sha256": "changed"}, 3)


def test_status_proofs_never_access_server_three_or_request_large_batches():
    with pytest.raises(ValueError, match="two records"):
        server_dispositions("cdu_dev3", "/state", "a" * 64, [])
    with pytest.raises(ValueError, match="two records"):
        server_dispositions("cdu_dev1", "/state", "a" * 64, ["b" * 64] * 3)
