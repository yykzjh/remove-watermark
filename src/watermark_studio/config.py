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
    remove_logo: bool = False
    resolution: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    context: float = Field(default=0.6, ge=0.1, le=2)
    feather: int = Field(default=12, ge=0, le=64)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    overwrite: bool = False
    extra_prompt: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_regions(self):
        if bool(self.output_width) != bool(self.output_height):
            raise ValueError("输出宽高必须同时填写，或同时为 0 保留原图尺寸")
        if self.output_width * self.output_height > 64_000_000:
            raise ValueError("输出最多支持 6400 万像素")
        if self.scope == "regions" and self.preset == "custom" and not self.regions:
            raise ValueError("自定义模式至少需要一个水印区域")
        return self


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
    cache_dir: str = "~/.cache/huggingface/hub"
    gpus: list[int] = Field(default_factory=lambda: list(range(8)), min_length=1, max_length=16)
    parallelism: Literal["single", "data"] = "data"
    batch_size: int = Field(default=2, ge=1, le=16)
    batch_window: int = Field(default=4, ge=1, le=8)
    batch_bucketing: bool = False
    oom_split_batch: bool = False
    replicas_per_gpu: int = Field(default=1, ge=1, le=4)
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    attention: Literal["sdpa", "flex"] = "sdpa"
    attention_backend: Literal["native", "native_flash", "native_cudnn", "native_efficient", "flash_hub", "sage_hub"] = "native"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = "default"
    compile_scope: Literal["full", "blocks"] = "full"
    compile_dynamic: bool = False
    prompt_encoder: Literal["stock", "lean"] = "stock"
    encoder_batch_size: int = Field(default=1, ge=1, le=16)
    fuse_qkv: bool = False
    quantization: Literal["none", "float8", "int8"] = "none"
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
