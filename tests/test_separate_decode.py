import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from watermark_studio.batching import decode_latents


class Tensor:
    def __init__(self,values):
        self.values=np.asarray(values,dtype=np.float32)
        self.shape=self.values.shape
        self.ndim=self.values.ndim
        self.device="cpu"
        self.dtype="float32"
    def view(self,*shape):
        return Tensor(self.values.reshape(shape))
    def to(self,*args):
        return self
    def __mul__(self,other):
        return Tensor(self.values*other.values)
    def __add__(self,other):
        return Tensor(self.values+other.values)
    def __getitem__(self,key):
        return Tensor(self.values[key])


def test_separate_decode_matches_pinned_normalization_grid_and_postprocessing(monkeypatch):
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(inference_mode=nullcontext,tensor=Tensor))
    flat=Tensor(np.arange(8).reshape(1,4,2))
    captured=[]
    def decode(values,return_dict):
        captured.append(values.values)
        return (values,)
    pipe=SimpleNamespace(vae_scale_factor=2,
        _unpack_latents=lambda values,*args:Tensor(values.values.transpose(0,2,1).reshape(1,2,1,2,2)),
        vae=SimpleNamespace(dtype="float32",config=SimpleNamespace(z_dim=2,latents_mean=[10,20],latents_std=[2,3]),decode=decode),
        image_processor=SimpleNamespace(postprocess=lambda image,output_type:image.values))
    result=decode_latents(pipe,flat,4,4)
    expected=flat.values.transpose(0,2,1).reshape(1,2,1,2,2)*np.asarray([2,3])[None,:,None,None,None]
    expected+=np.asarray([10,20])[None,:,None,None,None]
    assert np.array_equal(captured[0],expected)
    assert np.array_equal(result,expected[:,:,0])
    with pytest.raises(ValueError,match="Latent sequence"):
        decode_latents(pipe,Tensor(np.zeros((1,5,2))),4,4)
