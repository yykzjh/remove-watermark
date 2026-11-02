"""Lossless offline prefix KV compaction for reference keys already excluded by attention."""
from __future__ import annotations


class ReferencePrefixCache:
    def __init__(self,keep,target_tokens):
        import torch

        if (keep.ndim!=2 or keep.dtype!=torch.bool or type(target_tokens) is not int
                or not 0<target_tokens<keep.shape[1] or not keep[:,-target_tokens:].all().item()):
            raise ValueError("Joint key validity and unmodified target keys required")
        self.prefix_tokens=keep.shape[1]-target_tokens;self.target_tokens=target_tokens
        self.batch=keep.shape[0];prefix=keep[:,:self.prefix_tokens]
        counts=prefix.sum(1);self.compact_tokens=int(counts.max().item())
        if self.batch<1 or counts.min().item()<1:
            raise ValueError("Every independent image requires prefix context")
        self.indices=torch.zeros(self.batch,self.compact_tokens,dtype=torch.long,device=keep.device)
        self.valid=torch.zeros(self.batch,self.compact_tokens,dtype=torch.bool,device=keep.device)
        for i in range(self.batch):
            selected=prefix[i].nonzero(as_tuple=True)[0]
            self.indices[i,:selected.numel()]=selected
            self.valid[i,:selected.numel()]=True
        self.dense=bool(self.valid.all().item())

    def pack(self,tensor):
        if tensor.ndim!=4 or tensor.shape[:2]!=(self.batch,self.prefix_tokens):
            raise ValueError("Prefix KV layout differs from compaction plan")
        positions=self.indices[:,:,None,None].expand(-1,-1,tensor.shape[2],tensor.shape[3])
        result=tensor.gather(1,positions)
        # Invalid row padding never becomes an attended key.
        return result.masked_fill(~self.valid[:,:,None,None],0)

    def cached_mask(self,mask):
        import torch

        if mask is None:
            if self.dense:return None
            return torch.cat([self.valid,torch.ones(self.batch,self.target_tokens,
                device=self.valid.device,dtype=torch.bool)],dim=1)[:,None,None,:]
        if (mask.ndim!=4 or mask.shape[:3]!=(self.batch,1,1)
                or mask.shape[-1]!=self.prefix_tokens+self.target_tokens or mask.dtype!=torch.bool):
            raise ValueError("Cached attention requires the original joint padding mask")
        prefix=mask[:,0,0,:self.prefix_tokens].gather(1,self.indices) & self.valid
        return torch.cat([prefix,mask[:,0,0,self.prefix_tokens:]],dim=1)[:,None,None,:]

    def extracting(self,cache,write_slice):
        if (cache is None or write_slice!=slice(0,self.prefix_tokens)):
            raise ValueError("Compaction requires the exact constant prefix extraction slice")
        plan=self
        class Extractor:
            def store(self,key,value):
                cache.store(plan.pack(key),plan.pack(value))
            def get(self):
                return cache.get()
        return Extractor()
