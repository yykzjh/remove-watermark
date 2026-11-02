"""Hash-only source isolation checks; no model calls, labels, or image previews."""
from __future__ import annotations

import base64
import hashlib
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from .images import load_photo

COSINE=np.cos(np.pi*(2*np.arange(32)[None,:]+1)*np.arange(8)[:,None]/64)


def _bits(values):
    packed=np.packbits(np.asarray(values,dtype=np.uint8).reshape(-1),bitorder="big").tobytes()
    return packed.hex()


def image_fingerprint(photo):
    rgb=photo.convert("RGB");width,height=rgb.size
    digest=hashlib.sha256(struct.pack(">II",width,height)+rgb.tobytes()).hexdigest()
    gray=np.asarray(rgb.convert("L").resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    spectrum=COSINE@gray@COSINE.T;flat=spectrum.flatten();bits=flat>np.median(flat[1:]);bits[0]=False
    small=np.asarray(rgb.resize((16,16),Image.Resampling.LANCZOS),dtype=np.uint8)
    dgray=np.asarray(rgb.convert("L").resize((9,8),Image.Resampling.LANCZOS),dtype=np.int16)
    return {"width":width,"height":height,"decoded_rgb_sha256":digest,"phash64":_bits(bits),
        "dhash64":_bits(dgray[:,1:]>dgray[:,:-1]),
        "small_rgb16_base64":base64.b64encode(small.tobytes()).decode(),
        "method":"exif_display_rgb_sha256_rgb16_blockmean_v1"}


def fingerprint_file(path):
    path=Path(path);before=path.stat();raw=path.read_bytes()
    photo=load_photo(io.BytesIO(raw))[0]
    value=image_fingerprint(photo);after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError("Source changed during fingerprinting")
    return {**value,"path":str(path),"encoded_sha256":hashlib.sha256(raw).hexdigest(),
        "bytes":len(raw)}


def compare_fingerprints(first,second):
    if first["encoded_sha256"]==second["encoded_sha256"]:return {"relation":"encoded_equal"}
    if first["decoded_rgb_sha256"]==second["decoded_rgb_sha256"]:return {"relation":"decoded_rgb_equal"}
    distance=(int(first["phash64"],16)^int(second["phash64"],16)).bit_count()
    ratio=(first["width"]/first["height"])/(second["width"]/second["height"])
    if not .98<=ratio<=1.02:return None
    arrays=[]
    for value in (first,second):
        raw=base64.b64decode(value["small_rgb16_base64"],validate=True)
        if len(raw)!=768:raise ValueError("Bounded RGB16 fingerprint required")
        arrays.append(np.frombuffer(raw,dtype=np.uint8).astype(np.float32))
    a,b=arrays;error=np.abs(a-b);mae=float(error.mean());p95=float(np.quantile(error,.95))
    x,y=a-a.mean(),b-b.mean();norm=float(np.linalg.norm(x)*np.linalg.norm(y))
    correlation=float(x@y)/norm if norm>1e-8 else (1. if mae<=.5 else 0.)
    spatial_std=max(float(array.reshape(16,16,3).std(axis=(0,1)).mean()) for array in arrays)
    flat_candidate=mae<=.5 and p95<=2 and spatial_std<8
    if mae>3 or p95>8 or (correlation<.995 and not flat_candidate):return None
    return {"relation":"near_flat_candidate" if correlation<.995 else "near_duplicate_candidate","phash_distance":distance,
        "rgb16_mae":mae,"rgb16_p95":p95,"rgb16_correlation":correlation}


class FingerprintIndex:
    def __init__(self):
        self.values=[];self.exact={};self.encoded={};self.coarse=np.empty((1024,48),np.float32)
    @staticmethod
    def coarse_values(value):
        raw=base64.b64decode(value["small_rgb16_base64"],validate=True)
        if len(raw)!=768:raise ValueError("Bounded RGB16 fingerprint required")
        return np.frombuffer(raw,dtype=np.uint8).astype(np.float32).reshape(4,4,4,4,3).mean(axis=(1,3)).flatten()
    def add(self,value):
        index=len(self.values)
        if index>=len(self.coarse):
            larger=np.empty((len(self.coarse)*2,48),np.float32)
            larger[:index]=self.coarse;self.coarse=larger
        self.coarse[index]=self.coarse_values(value);self.values.append(value)
        self.exact.setdefault(value["decoded_rgb_sha256"],set()).add(index)
        self.encoded.setdefault(value["encoded_sha256"],set()).add(index)
    def matches(self,value):
        candidates=set(self.exact.get(value["decoded_rgb_sha256"],()))|set(self.encoded.get(value["encoded_sha256"],()))
        query=self.coarse_values(value)
        # Block averaging cannot increase mean absolute pixel difference
        # (triangle inequality). This filter preserves every RGB16-MAE<=3
        # candidate, including low-texture JPEGs with unstable perceptual hashes.
        if self.values:
            distances=np.abs(self.coarse[:len(self.values)]-query).mean(axis=1)
            candidates.update(np.flatnonzero(distances<=3.0001).tolist())
        output=[]
        for index in sorted(candidates):
            other=self.values[index];relation=compare_fingerprints(value,other)
            if relation:output.append((other,relation))
        return output
