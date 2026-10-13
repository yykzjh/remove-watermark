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


def attention(query, key, value, mask=None):
    if mask is not None and mask.ndim == 4 and mask.shape[-2] == 1:
        outputs = []
        for index in range(query.shape[0]):
            valid = mask[min(index, mask.shape[0]-1), 0, 0].nonzero(as_tuple=True)[0]
            outputs.append(dispatch_attention_fn(
                query[index:index+1],
                key[index:index+1].index_select(1, valid),
                value[index:index+1].index_select(1, valid),
                attn_mask=None, dropout_p=0.0, backend=None,
            ))
        return torch.cat(outputs)
    return dispatch_attention_fn(query, key, value, attn_mask=mask,
                                 dropout_p=0.0, backend=None)


class IndependentSDPAProcessor(QwenImage21AttnProcessor):
    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        from .optimizations import prepare_qkv
        query, key, value, seq_len_q = prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )
        if segments is None:
            output = attention(query, key, value, attention_mask)
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
                outputs.append(attention(query[:,start:end],key[:,:end],value[:,:end],mask))
            prefix = segments[-1][1] if segments else 0
            mask = None if key_valid is None else key_valid[:,None,None,:]
            outputs.append(attention(query[:,prefix:],key,value,mask))
            output = torch.cat(outputs,dim=1)
        output = output[:,:seq_len_q].flatten(2,3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))
