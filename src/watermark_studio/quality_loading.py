"""Independent verifier loading; quantization never replaces the vision encoder."""
def load_quality_encoder(path, config, device):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

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
    encoder=Qwen3VLForConditionalGeneration.from_pretrained(path,**kwargs)
    if config.quality_quantization=="none":
        encoder=encoder.to(device)
    for name,module in encoder.named_modules():
        if "visual" in name and type(module).__name__ in {"Linear4bit","Linear8bitLt"}:
            raise RuntimeError("Verifier vision encoder must retain full inference precision")
    return encoder.eval()
