from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["panel", "logo", "custom"] = "custom"
    box: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_box(self):
        x0, y0, x1, y1 = self.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("区域格式必须是 0..1 范围的 x0,y0,x1,y1，且起点小于终点")
        return self


PRESETS = {
    "gray": [
        Region(kind="panel", box=(0.012, 0.727, 0.45, 0.988)),
        Region(kind="logo", box=(0.864, 0.89, 1.0, 1.0)),
    ],
    "blue": [
        Region(kind="panel", box=(0.01, 0.781, 0.412, 0.987)),
        Region(kind="logo", box=(0.895, 0.922, 1.0, 1.0)),
    ],
}


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["full_image", "regions"] = "regions"
    output_width: int = Field(default=0, ge=0, le=16384)
    output_height: int = Field(default=0, ge=0, le=16384)
    concurrency: int = Field(default=32, ge=1, le=1024)
    preset: Literal["gray", "blue", "auto", "custom"] = "auto"
    regions: list[Region] = Field(default_factory=list, max_length=8)
    detection_resolution: int = Field(default=1280, ge=512, le=2048, multiple_of=32)
    preserve_background: bool = False
    first_pass_mode: Literal["whole", "focused", "thorough", "complete"] = "whole"
    whole_prompt_profile: Literal["baseline","fidelity","scene","located","concise","neutral","clean-photo","clean-photo-detailed"] = "neutral"
    repair_blending: Literal["feather", "harmonic", "harmonic-fine"] = "harmonic"
    quality_check: bool = False
    quality_cross_review: bool = False
    quality_auto_retry: bool = False
    quality_retry_localized: bool = False
    quality_max_attempts: int = Field(default=3, ge=0, le=20)
    quality_retry_resolution: Literal[512, 768, 1024, 1280, 1536, 2048] = 1024
    quality_resolution: int = Field(default=768, ge=384, le=1280, multiple_of=32)
    quality_batch_size: int = Field(default=1, ge=1, le=4)
    quality_max_tokens: int = Field(default=128, ge=64, le=256)
    quality_detail_patches: int = Field(default=4, ge=0, le=8)
    remove_logo: bool = False
    resolution: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    roi_grouping: Literal["separate", "adaptive"] = "separate"
    context: float = Field(default=0.6, ge=0.1, le=2)
    feather: int = Field(default=12, ge=0, le=64)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    overwrite: bool = False
    extra_prompt: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_regions(self):
        if self.first_pass_mode != "whole" and not self.preserve_background:
            raise ValueError("首轮精细策略需要启用原图保护")
        if self.quality_cross_review and not self.quality_check:
            raise ValueError("编辑侧争议复核需要启用质量检验")
        if self.quality_retry_localized and not self.quality_auto_retry:
            raise ValueError("有位置依据的返工策略需要启用自动返工")
        if self.quality_auto_retry and not self.quality_check:
            raise ValueError("自动返工需要启用质量检验")
        if bool(self.output_width) != bool(self.output_height):
            raise ValueError("输出宽高必须同时填写，或同时为 0 保留原图尺寸")
        if self.output_width * self.output_height > 64_000_000:
            raise ValueError("输出最多支持 6400 万像素")
        if self.scope == "regions" and self.preset == "custom" and not self.regions:
            raise ValueError("自定义模式至少需要一个水印区域")
        return self


def processing_route(mode):
    """Frontend route names map to the existing explicit API processing fields."""
    if mode == "regions":
        return "regions", "whole"  # Basic automatic crop path; no boolean is implied.
    if mode in ("whole", "focused", "thorough", "complete"):
        return "full_image", mode
    raise ValueError("未知去水印处理方式")


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["batch", "single"] = "batch"
    preview_token: str = Field(default="", max_length=100)
    input_file: str = Field(default="", max_length=4096)
    input_dir: str = Field(default="", max_length=4096)
    output_dir: str = Field(min_length=1, max_length=4096)
    options: ProcessingConfig = Field(default_factory=ProcessingConfig)

    @model_validator(mode="after")
    def validate_source(self):
        if self.mode == "single" and not self.input_file.strip():
            raise ValueError("单张任务必须指定 input_file")
        if self.mode == "batch" and not self.input_dir.strip():
            raise ValueError("批量任务必须指定 input_dir")
        return self


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: Literal["127.0.0.1", "::1"] = "127.0.0.1"
    port: int = Field(default=8765, ge=1024, le=65535)
    model: str = "Qwen/Qwen-Image-2.1-Turbo"
    revision: str = "main"
    model_source: Literal["huggingface", "modelscope"] = "huggingface"
    uniform_steps: Literal[0,12,16,20,24,40] = 0
    roi_steps: Literal[0,12,16,20,24,40] = 0
    roi_step_min_area: float = Field(default=.14,ge=0,le=1,allow_inf_nan=False)
    cache_dir: str = "~/.cache/huggingface/hub"
    gpus: list[int] = Field(default_factory=lambda: list(range(8)), min_length=1, max_length=16)
    parallelism: Literal["single", "data"] = "data"
    batch_size: int = Field(default=2, ge=1, le=16)
    large_batch_size: int = Field(default=0, ge=0, le=16)
    small_batch_size: int = Field(default=0, ge=0, le=16)
    prefill_overlays: bool = False
    prefill_style: Literal["harmonic","gray"] = "harmonic"
    prefill_original_semantics: bool = False
    protect_denoising_context: bool = False
    protect_source_rims: bool = False
    translucent_body_hint: bool = False
    translucent_transmission: float = Field(default=.55, ge=.35, le=.85, allow_inf_nan=False)
    transformer_phase_offload: bool = False
    lora_path: str = Field(default="", max_length=4096)
    lora_scale: float = Field(default=1., ge=0, le=2, allow_inf_nan=False)
    lora_alpha: float = Field(default=0., ge=0, le=512, allow_inf_nan=False)
    lora_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    batch_window: int = Field(default=4, ge=1, le=8)
    batch_bucketing: bool = False
    adaptive_dispatch: bool = False
    detection_batch_size: int = Field(default=1, ge=1, le=4)
    oom_split_batch: bool = False
    replicas_per_gpu: int = Field(default=1, ge=1, le=4)
    quality_verifier: bool = False
    quality_physical_text_review: bool = False
    protected_document_resolution: Literal[0,1024,1280,1536,2048] = 0
    retain_rejected_first: bool = False
    source_vet_batch_size: int = Field(default=1,ge=1,le=4)
    protection_physical_vet: bool = False
    source_carrier_recheck: bool = False
    protection_detail_scan: bool = False
    protection_core_feather: bool = False
    separate_vae_decode: bool = False
    quality_phase_offload: bool = False
    quality_phase_offload_above: int = Field(default=0,ge=0,le=2048,multiple_of=32)
    quality_cross_full_context: bool = False
    quality_rectangle_guard: bool = False
    quality_backend: Literal["resident", "independent"] = "resident"
    quality_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    quality_revision: str = "main"
    quality_quantization: Literal["none","int8","nf4"] = "none"
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    attention: Literal["sdpa", "flex"] = "sdpa"
    attention_backend: Literal["native", "native_flash", "native_cudnn", "native_efficient", "flash_hub", "sage_hub"] = "native"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = "default"
    compile_scope: Literal["full", "blocks", "projections"] = "full"
    compile_dynamic: bool = False
    prompt_encoder: Literal["stock", "lean"] = "stock"
    encoder_batch_size: int = Field(default=1, ge=1, le=16)
    fuse_qkv: bool = False
    quantization: Literal["none", "float8", "int8"] = "none"
    float8_granularity: Literal["tensor", "row"] = "tensor"
    float8_accurate_accum: bool = False
    float8_amax_floor: float = Field(default=0, ge=0, le=1)
    cudnn_benchmark: bool = False
    vae_channels_last: bool = False
    vae_tile_size: int = Field(default=256, ge=128, le=1024, multiple_of=32)
    vae_tile_stride: int = Field(default=192, ge=64, le=1024, multiple_of=32)
    expandable_segments: bool = False
    shape_sort: bool = False
    scan_workers: int = Field(default=1, ge=1, le=16)
    report_interval: int = Field(default=1, ge=1, le=1024)
    offload: Literal["none", "model", "sequential"] = "none"
    vae_tiling: bool = False
    vae_slicing: bool = False
    kv_cache: bool = False
    tf32: bool = False
    cpu_threads: int = Field(default=1, ge=1, le=64)
    load_parallelism: int = Field(default=8, ge=1, le=16)
    allowed_roots: list[str] = Field(default_factory=lambda: ["~", "/mnt"])
    state_dir: str = "~/.local/state/watermark-studio"
    backend: Literal["qwen", "mock"] = "qwen"

    @model_validator(mode="after")
    def compatible(self):
        if self.quality_physical_text_review and not (self.quality_verifier and self.quality_backend=="independent" and self.protection_physical_vet):
            raise ValueError("Physical text review requires an independent verifier and source physical vetting")
        if self.quantization=="float8" and self.float8_granularity=="row" and not self.float8_amax_floor:
            raise ValueError("Per-row FP8 requires a positive float8-amax-floor (e.g. 1e-6) to prevent zero-row NaNs")
        if self.float8_accurate_accum and self.quantization!="float8":
            raise ValueError("Accurate FP8 accumulation requires float8 quantization")
        if self.float8_amax_floor and self.quantization!="float8":
            raise ValueError("An FP8 activation amax floor requires float8 quantization")
        if self.uniform_steps and self.roi_steps:
            raise ValueError("Select global uniform-steps or source-area ROI steps, not both")
        if self.prefill_style=="gray" and (not self.prefill_overlays or not self.protect_source_rims or self.prefill_original_semantics):
            raise ValueError("Gray whole conditioning requires prefill-overlays and protect-source-rims; original-semantics is incompatible")
        if self.translucent_body_hint and (not self.protect_source_rims or self.prefill_overlays):
            raise ValueError("半透明背景输入实验需要protect-source-rims且不能同时使用prefill-overlays")
        if self.transformer_phase_offload and (self.quantization!="none" or self.offload!="none"):
            raise ValueError("源图阶段停放扩散模型目前仅支持quantization=none、offload=none")
        if self.source_carrier_recheck and (not self.protection_physical_vet
                or not self.quality_verifier or self.quality_backend!="independent"):
            raise ValueError("Source carrier recheck requires physical vetting and independent verifier")
        if self.protect_source_rims and (not self.quality_verifier or self.quality_backend!="independent"):
            raise ValueError("原图边沿保护需要quality-verifier和独立检验模型")
        if self.protect_denoising_context and self.prefill_overlays:
            raise ValueError("生成上下文保护需使用未初始化的完整原图，不能同时启用prefill-overlays")
        if self.quality_quantization!="none" and (not self.quality_verifier or self.quality_backend!="independent"):
            raise ValueError("检验量化需要启用独立检验模型")
        if len(set(self.gpus)) != len(self.gpus) or any(g < 0 for g in self.gpus):
            raise ValueError("GPU 编号必须非负且不能重复")
        if self.parallelism == "single" and self.worker_count != 1:
            raise ValueError("single 模式要求一个 GPU 且 replicas_per_gpu=1")
        if self.attention == "flex" and not self.compile:
            raise ValueError("flex attention 必须启用 compile，否则可能造成显存溢出")
        if self.vae_tile_stride > self.vae_tile_size:
            raise ValueError("VAE stride 不得大于 tile size")
        if self.compile and self.compile_mode not in {"default", "max-autotune-no-cudagraphs"}:
            raise ValueError("当前固定依赖的 reduce-overhead/max-autotune 实测 CUDA Graph 错误，请使用 default")
        if self.attention == "flex" and self.compile:
            raise ValueError("当前固定依赖的 Flex 实测 Inductor 编译失败，请使用 sdpa")
        if self.separate_vae_decode and self.offload!="none":
            raise ValueError("分离VAE解码目前需要offload=none")
        if (self.quality_phase_offload or self.quality_phase_offload_above or self.transformer_phase_offload) and self.quality_backend=="independent" and self.quality_quantization!="none":
            raise ValueError("阶段权重卸载只支持未量化的独立检验器")
        if self.offload != "none" and (self.quantization != "none" or self.prompt_encoder == "lean" or self.fuse_qkv):
            raise ValueError("CPU offload 尚未验证量化、lean 编码或 QKV 融合组合，请使用 stock/none 且关闭 fuse-qkv")
        if self.compile and self.offload != "none":
            raise ValueError("当前实现不组合 compile 与 CPU offload，请二选一")
        return self

    @property
    def worker_count(self):
        return len(self.gpus) * self.replicas_per_gpu

    @property
    def capacity(self):
        return self.worker_count * self.batch_size * self.batch_window


def load_deployment(path: str | Path) -> DeploymentConfig:
    with Path(path).open("rb") as f:
        return DeploymentConfig.model_validate(tomllib.load(f))
