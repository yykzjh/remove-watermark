"""Independent verifier loading; quantization never replaces the vision encoder."""
import json
from pathlib import Path


def quality_architecture(path):
    architecture=json.loads((Path(path)/"config.json").read_text())["model_type"]
    if architecture not in {"qwen3_vl","qwen3_5"}:
        raise ValueError(f"Unsupported verifier architecture: {architecture}")
    return architecture


class _NonThinkingProcessor:
    """Keep short structured inspections outside the candidate's reasoning mode."""
    def __init__(self,processor):
        self._processor=processor

    def __getattr__(self,name):
        return getattr(self._processor,name)

    def apply_chat_template(self,*args,**kwargs):
        kwargs["enable_thinking"]=False
        return self._processor.apply_chat_template(*args,**kwargs)


def load_quality_processor(path):
    from transformers import AutoProcessor

    processor=AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    return _NonThinkingProcessor(processor) if quality_architecture(path)=="qwen3_5" else processor

def load_quality_encoder(path, config, device):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    architecture=quality_architecture(path)
    factory=Qwen3VLForConditionalGeneration
    if architecture=="qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        factory=Qwen3_5ForConditionalGeneration
        if config.quality_quantization!="none":
            raise ValueError("Qwen3.5 verifier candidate currently requires unquantized inference")


    kwargs={"dtype":getattr(torch,config.dtype),"local_files_only":True}
    if config.quality_quantization!="none":
        try:
            import bitsandbytes  # noqa: F401 - fail early with a useful install message
        except ImportError as error:
            raise RuntimeError("Install uv extra quality-quantization before quantized verifier deployment") from error
        settings={"llm_int8_skip_modules":["model.visual","visual","lm_head"]}
        if config.quality_quantization=="nf4":
            settings.update(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=getattr(torch,config.dtype))
        else:
            settings["load_in_8bit"]=True
        kwargs.update(quantization_config=BitsAndBytesConfig(**settings),
                      device_map={"":str(device)})
    encoder=factory.from_pretrained(path,**kwargs)
    if config.quality_quantization=="none":
        encoder=encoder.to(device)
    for name,module in encoder.named_modules():
        if "visual" in name and type(module).__name__ in {"Linear4bit","Linear8bitLt"}:
            raise RuntimeError("Verifier vision encoder must retain full inference precision")
    return encoder.eval()
