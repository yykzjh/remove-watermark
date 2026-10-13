"""Download on the development machine directly, independently of GPU installation."""
import argparse
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from watermark_studio.download import download_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/deployment.toml")
args = parser.parse_args()
with open(args.config, "rb") as stream:
    config = tomllib.load(stream)
path = download_model(config.get("model", "Qwen/Qwen-Image-2.1-Turbo"),
                      config.get("model_source", "huggingface"),
                      config.get("revision", "main"),
                      config.get("cache_dir", "~/.cache/huggingface/hub"))
print("Model snapshot ready:", path)
