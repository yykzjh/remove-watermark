"""One CLI option for every validated deployment field; command line wins over TOML."""
import argparse
import json
import tomllib
from typing import Literal, get_args, get_origin

from .config import DeploymentConfig

HELP = {
    "host": "Loopback bind address: 127.0.0.1 or ::1",
    "port": "API port, 1024..65535",
    "model": "Model repository ID or a local model directory",
    "revision": "Model repository revision (main on HF, master on ModelScope, or pinned commit)",
    "model_source": "Model download provider; ignored for a local model directory",
    "cache_dir": "Model download cache directory",
    "parallelism": "single: exactly one worker; data: independent image workers on selected GPUs",
    "gpus": "Comma-separated physical GPU indices, e.g. 1,2,3",
    "replicas_per_gpu": "Full model copies per GPU, 1..4; multiplies VRAM use",
    "dtype": "Model weight and inference dtype",
    "attention": "sdpa or flex (flex requires --compile)",
    "compile": "Compile transformer; first inference and new shapes can take longer",
    "compile_mode": "PyTorch compile mode; only active with --compile",
    "offload": "CPU offload: none, model, sequential; incompatible with compile",
    "vae_tiling": "Tile VAE to reduce activation memory",
    "vae_slicing": "Enable VAE slicing; limited benefit when decoding one image at a time",
    "kv_cache": "Reuse prefix KV within the current image's denoising steps",
    "tf32": "Allow TF32 for eligible FP32 CUDA matrix multiplications",
    "cpu_threads": "PyTorch CPU threads per worker, 1..64",
    "load_parallelism": "Concurrent model-loading workers, 1..16; does not set job concurrency",
    "allowed_roots": "Replace allowed directory roots; space-separated paths",
    "state_dir": "Directory for job summaries and per-image result logs",
    "backend": "qwen: real model; mock: tests only, copies images without removal",
}


def gpu_list(value):
    try:
        return [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU 列表格式应为 1,2,3") from error


def make_parser():
    parser = argparse.ArgumentParser(
        description="Deploy watermark service. Priority: CLI > TOML > built-in defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/deployment.toml", help="Base TOML configuration")
    parser.add_argument("--print-config", action="store_true",
                        help="Validate and print effective JSON, then exit without loading models")
    for name, field in DeploymentConfig.model_fields.items():
        options = {"dest": name, "default": argparse.SUPPRESS, "help": HELP[name]}
        annotation = field.annotation
        if name == "gpus":
            options["type"] = gpu_list
        elif annotation is bool:
            options["action"] = argparse.BooleanOptionalAction
        elif get_origin(annotation) is Literal:
            options["choices"] = get_args(annotation)
        elif get_origin(annotation) is list:
            options.update(nargs="+", type=str)
        else:
            options["type"] = annotation
        parser.add_argument("--" + name.replace("_", "-"), **options)
    parser.add_argument("--mock", dest="backend", action="store_const", const="mock",
                        default=argparse.SUPPRESS, help="Alias for --backend mock")
    return parser


def parse_deployment(argv=None):
    parser = make_parser()
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    print_config = args.pop("print_config")
    try:
        with open(config_path, "rb") as stream:
            config = DeploymentConfig.model_validate(tomllib.load(stream) | args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return config, print_config


def render_config(config):
    return json.dumps(config.model_dump(), ensure_ascii=False, indent=2)
