"""Exact masked SDPA without materializing a large padding-mask attention matrix.

Qwen's stock processor sends Bx1x1xK masks for target/condition image attention.
On this stack that misses the fast unmasked kernel. Remove only invalid keys per
sample for these calls; projection/MLP remain batched. Causal text masks retain
the stock implementation. This adapts the pinned Diffusers attention processor.
"""
import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    dispatch_attention_fn,
)


def attention(query, key, value, mask=None, backend=None):
    if mask is not None and mask.ndim == 4 and mask.shape[-2] == 1:
        outputs = []
        for index in range(query.shape[0]):
            valid = mask[min(index, mask.shape[0]-1), 0, 0].nonzero(as_tuple=True)[0]
            outputs.append(dispatch_attention_fn(
                query[index:index+1],
                key[index:index+1].index_select(1, valid),
                value[index:index+1].index_select(1, valid),
                attn_mask=None, dropout_p=0.0, backend=backend,
            ))
        return torch.cat(outputs)
    return dispatch_attention_fn(query, key, value, attn_mask=mask,
                                 dropout_p=0.0, backend=backend if mask is None else "native")


class IndependentSDPAProcessor(QwenImage21AttnProcessor):
    def __init__(self, backend=None):
        self._attention_backend = backend


    def _target_mask(self, mask, key):
        keep=getattr(self,"_watermark_target_key_keep",None)
        if keep is None:
            return mask
        if (keep.ndim!=2 or keep.dtype!=torch.bool or keep.device!=key.device
                or keep.shape!=(key.shape[0],key.shape[1])):
            raise ValueError("Target reference key mask differs from actual attention keys")
        if mask is not None and (mask.ndim!=4 or mask.shape[-2]!=1):
            raise ValueError("Target reference exclusion requires a key-validity mask")
        keep=keep[:,None,None,:]
        return keep if mask is None else mask & keep

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        from .optimizations import prepare_qkv
        query, key, value, seq_len_q = prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )
        if segments is None:
            if getattr(self,"_watermark_target_key_keep",None) is not None:
                if (kv_cache_mode!="cached" or query.shape[1]!=self._watermark_target_query_count):
                    raise ValueError("Reference exclusion without prefix segments requires cached target queries")
                attention_mask=self._target_mask(attention_mask,key)
            output = attention(query, key, value, attention_mask, self._attention_backend)
        else:
            outputs = []
            for start, end, is_text in segments:
                mask = None
                if is_text:
                    length = end-start
                    mask = torch.cat([
                        torch.ones(length,start,dtype=torch.bool,device=query.device),
                        torch.tril(torch.ones(length,length,dtype=torch.bool,device=query.device)),
                    ],dim=1)[None,None]
                if key_valid is not None:
                    valid = key_valid[:,None,None,:end]
                    mask = valid if mask is None else mask & valid
                outputs.append(attention(query[:,start:end],key[:,:end],value[:,:end],mask,self._attention_backend))
            prefix = segments[-1][1] if segments else 0
            mask = None if key_valid is None else key_valid[:,None,None,:]
            if getattr(self,"_watermark_target_key_keep",None) is not None:
                if query.shape[1]-prefix!=self._watermark_target_query_count:
                    raise ValueError("Prefill target query count differs from reference exclusion")
                mask=self._target_mask(mask,key)
            outputs.append(attention(query[:,prefix:],key,value,mask,self._attention_backend))
            output = torch.cat(outputs,dim=1)
        output = output[:,:seq_len_q].flatten(2,3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))
