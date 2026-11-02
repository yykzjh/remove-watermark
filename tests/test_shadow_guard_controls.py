import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def test_shadow_controls_use_true_background_and_do_not_change_outside_the_injected_box():
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/probe-shadow-guard-synthetic.py"))
    clean=Image.new("RGB",(128,96),(130,150,170));box=(20,25,88,80);before=clean.tobytes()
    records=list(module["controls"](clean,box))
    assert [name for name,_,_ in records]==["exact","jpeg95","dark18","dark32","bright18"]
    outside=np.ones((96,128),bool);outside[25:80,20:88]=False
    for name,value,expected in records:
        assert clean.tobytes()==before
        if expected=="review":
            assert np.array_equal(np.asarray(value)[outside],np.asarray(clean)[outside])
            delta={"dark18":-18,"dark32":-32,"bright18":18}[name]
            assert np.array_equal(np.asarray(value)[25:80,20:88],np.asarray(clean,dtype=np.int16)[25:80,20:88]+delta)
