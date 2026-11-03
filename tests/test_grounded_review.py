import json

import pytest

from watermark_studio.grounded_review import parse_grounded

FIELDS=("watermark_removed","natural","uncertain")
def test_grounded_review_preserves_negative_evidence_and_uncertainty():
    for changes in [{"watermark_removed":False},{"natural":False},{"uncertain":True},{}]:
        value={"observations":"右下有可见小字，需核实载体","watermark_removed":True,"natural":True,"uncertain":False,"area":"bottom_right",**changes}
        row=parse_grounded(json.dumps(value),FIELDS)
        assert all(row[k]==value[k] for k in FIELDS)
        assert row["grounded_observations"]==value["observations"]
@pytest.mark.parametrize("changes",[{"observations":""},{"observations":None},{"observations":"a"*1601},{"natural":"true"},{"area":"invalid"}])
def test_grounded_review_rejects_incomplete_or_malformed_proof(changes):
    value={"observations":"未见数字覆盖，实物标牌保持可见","watermark_removed":True,"natural":True,"uncertain":False,"area":"none",**changes}
    with pytest.raises((ValueError,TypeError)):
        parse_grounded(json.dumps(value),FIELDS)
