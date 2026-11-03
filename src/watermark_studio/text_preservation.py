"""Conservative pixel witness for unchanged OUTPUT lettering, never a removal verdict."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageFilter

from .images import pixel_box
from .text_presence import TargetTextPresence


def enhance_text_contrast(photo):
    """Diagnostic luminance-only local contrast, never written to the output."""
    import cv2
    gray=np.asarray(photo.convert("L"))
    clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
    return Image.fromarray(clahe.apply(gray)).convert("RGB")


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0


class OutputTextPreservation:
    def __init__(self,detector=None,*,enhance_low_contrast=False):
        self.detector=detector or TargetTextPresence(max_edge=2048)
        self.enhance_low_contrast=enhance_low_contrast

    def inspect(self,before,after):
        started=time.monotonic()
        report={"method":"output_text_native_pixel_witness_v1","status":"error","text_boxes":[],
            "not_a_watermark_absence_proof":True}
        try:
            detection=self.detector.inspect(after,[[0,0,1,1]])
            report["detection"]=detection
            if detection.get("status")=="error":
                raise ValueError(detection.get("error","Text detector failed"))
            boxes=[box for group in detection.get("regions",[]) for box in group.get("textlike_boxes",[])]
            report["enhanced_only_boxes"]=[]
            if self.enhance_low_contrast:
                diagnostic=self.detector.inspect(enhance_text_contrast(after),[[0,0,1,1]])
                report["contrast_detection"]=diagnostic
                if diagnostic.get("status")=="error":
                    raise ValueError(diagnostic.get("error","Contrast text detector failed"))
                for group in diagnostic.get("regions",[]):
                    for box in group.get("textlike_boxes",[]):
                        if all(box_iou(box,previous)<.5 for previous in boxes):
                            boxes.append(box);report["enhanced_only_boxes"].append(box)
                if len(report["enhanced_only_boxes"])>8:
                    raise ValueError("More than eight new low-contrast targets; no bounded proof")
            if not boxes:
                report["status"]="no_text_witness"
                return report
            if len(boxes)>128:
                raise ValueError("Too many text regions for bounded preservation proof")
            # Compare the actual saved output, tolerating only small resampling/JPEG errors.
            original=before.convert("RGB")
            if original.size!=after.size:
                original=original.resize(after.size,Image.Resampling.LANCZOS)
            output=after.convert("RGB")
            for box in boxes:
                native=pixel_box(box,after.size)
                values=[]
                for image in (original,output):
                    patch=image.crop(native).filter(ImageFilter.GaussianBlur(.45))
                    values.append(np.asarray(patch,dtype=np.float32))
                delta=np.abs(values[0]-values[1]).mean(axis=2)
                mae=float(delta.mean());high=float(np.quantile(delta,.95))
                fraction=float((delta>12).mean())
                stable=mae<=3 and high<=10 and fraction<=.03
                report["text_boxes"].append({"box":box,"mae":round(mae,4),
                    "p95":round(high,4),"over12_fraction":round(fraction,6),"unchanged":stable})
            report["status"]="unchanged" if all(b["unchanged"] for b in report["text_boxes"]) else "changed_text"
        except Exception as error:  # noqa: BLE001 - an OCR/pixel failure cannot provide positive evidence
            report["error"]=f"{type(error).__name__}: {error}"[:240]
        finally:
            report["seconds"]=round(time.monotonic()-started,6)
        return report
