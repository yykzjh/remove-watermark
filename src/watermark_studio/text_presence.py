"""CPU text-presence veto for rare no-text appeals; never certifies repair quality."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .evidence import valid_box
from .images import pixel_box


class TargetTextPresence:
    def __init__(self,engine=None,max_edge=1536):
        self.engine=engine
        self.max_edge=max_edge

    def _engine(self):
        if self.engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as error:
                raise RuntimeError("Install uv extra quality-ocr with Python3.11/3.12") from error
            self.engine=RapidOCR(det_use_cuda=False,rec_use_cuda=False,cls_use_cuda=False,
                det_use_dml=False,rec_use_dml=False,cls_use_dml=False,
                intra_op_num_threads=1,inter_op_num_threads=1,
                det_limit_type="max",det_limit_side_len=self.max_edge,max_side_len=self.max_edge,
                det_box_thresh=.6,use_cls=False,use_rec=False)
        return self.engine

    def inspect(self,photo,boxes):
        """Any detected glyph-like region defeats an assertion of text absence.

        Detector errors also retain review. We do not transcribe personal text
        or infer physical/digital status from OCR, and never use an empty result
        as sole approval evidence.
        """
        began=time.monotonic()
        result={"method":"cpu_ppocrv4_text_presence_v1","status":"error","regions":[],
                "max_edge":self.max_edge,"det_box_threshold":.6}
        try:
            if not boxes or len(boxes)>32:
                raise ValueError("Bounded nonempty text-proof targets required")
            engine=self._engine()
            for value in boxes:
                box=valid_box(value)
                if box is None:
                    raise ValueError("Unknown text-proof target")
                native=pixel_box(box,photo.size)
                crop=photo.crop(native).convert("RGB")
                original=crop.size
                crop.thumbnail((self.max_edge,self.max_edge),Image.Resampling.LANCZOS)
                response=engine(np.asarray(crop)[...,::-1].copy(),
                                use_det=True,use_cls=False,use_rec=False)
                if not isinstance(response,tuple) or len(response)!=2:
                    raise ValueError("Invalid OCR presence response")
                rows,_=response
                findings=[]
                if rows is not None:
                    if not isinstance(rows,(list,tuple,np.ndarray)) or len(rows)>1000:
                        raise ValueError("Invalid OCR region list")
                    for row in rows:
                        quad=np.asarray(row,dtype=float)
                        if (quad.shape!=(4,2) or not np.isfinite(quad).all()
                                or np.any(quad<0) or np.any(quad[:,0]>crop.width)
                                or np.any(quad[:,1]>crop.height)):
                            raise ValueError("Invalid OCR text geometry")
                        lo,hi=quad.min(axis=0),quad.max(axis=0)
                        if np.any(hi<=lo):
                            raise ValueError("Degenerate OCR text region")
                        mapped=[(native[0]+lo[0]*original[0]/crop.width)/photo.width,
                                (native[1]+lo[1]*original[1]/crop.height)/photo.height,
                                (native[0]+hi[0]*original[0]/crop.width)/photo.width,
                                (native[1]+hi[1]*original[1]/crop.height)/photo.height]
                        findings.append(valid_box([float(value) for value in mapped]))
                result["regions"].append({"target":box,"textlike_boxes":findings})
                if findings:
                    result["status"]="text_present"
                    break
            else:
                result["status"]="clear"
        except Exception as error:  # noqa: BLE001 - failure is never proof of absence
            result["error"]=f"{type(error).__name__}: {error}"[:200]
        result["seconds"]=round(time.monotonic()-began,6)
        return result
