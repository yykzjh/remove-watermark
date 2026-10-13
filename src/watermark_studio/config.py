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
        if self.preset == "custom" and not self.regions:
            raise ValueError("自定义模式至少需要一个水印区域")
        return self


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_dir: str = Field(min_length=1, max_length=4096)
    output_dir: str = Field(min_length=1, max_length=4096)
    options: ProcessingConfig = Field(default_factory=ProcessingConfig)


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
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
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
        if self.compile and self.compile_mode != "default":
            raise ValueError("当前固定依赖的 reduce-overhead/max-autotune 实测 CUDA Graph 错误，请使用 default")
        if self.attention == "flex" and self.compile:
            raise ValueError("当前固定依赖的 Flex 实测 Inductor 编译失败，请使用 sdpa")
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
