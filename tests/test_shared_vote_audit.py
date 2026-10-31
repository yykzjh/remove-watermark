import json
import runpy
from pathlib import Path

import pytest

AUDIT=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/audit-shared-votes.py"))["audit_job"]


def fixture(tmp_path, relation="same_base_weights"):
    identity={"relation":relation,"editor_sha256":"a"*64,"verifier_sha256":"a"*64}
    (tmp_path/"weight-identity-audit.json").write_text(json.dumps({
        "job":"job","observed_revision":"old","identity":identity}))
    (tmp_path/"summary.json").write_text(json.dumps({"id":"job","code_revision":"old","state":"completed","total":1}))
    q={"status":"passed","cross_review":{"status":"corrected"},"watermark_removed":True,
       "pre_cross_review":{"status":"review","watermark_removed":False,"natural":True,
          "content_preserved":True,"uncertain":False,"guard_codes":None,"residual_regions":[],
          "model_explanation":{"status":"ok","summary":"disputed printing","findings":[]}},
       "evidence":[{"issue":"physical_text","box":[.1,.2,.2,.3],"observation":"later claim"}]}
    raw=json.dumps({"file":"image.png","ok":True,"quality":q})+"\n"
    (tmp_path/"results.jsonl").write_text(raw)
    return raw


def test_shared_vote_is_revoked_without_changing_source_evidence(tmp_path):
    original=fixture(tmp_path)
    result=AUDIT(tmp_path)
    assert result["strict_counts"]=={"review":1} and result["same_weight_votes_excluded"]==1
    assert (tmp_path/"results.jsonl").read_text()==original
    row=json.loads((tmp_path/"results-without-shared-votes.jsonl").read_text())
    assert row["quality"]["status"]=="review" and row["quality"]["watermark_removed"] is False
    assert row["quality"]["cross_review"]["status"]=="invalidated_same_base_weights"
    assert not row["quality"]["evidence"]


def test_distinct_model_consensus_cannot_be_invalidated_by_this_utility(tmp_path):
    fixture(tmp_path,"different_weights")
    with pytest.raises(ValueError,match="identical-weight"):
        AUDIT(tmp_path)


def test_missing_original_verdict_does_not_fabricate_a_refusal(tmp_path):
    fixture(tmp_path)
    path=tmp_path/"results.jsonl"
    row=json.loads(path.read_text());row["quality"].pop("pre_cross_review")
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="reconstruct"):
        AUDIT(tmp_path)
