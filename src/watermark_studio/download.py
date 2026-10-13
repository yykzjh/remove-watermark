from pathlib import Path


def download_model(model: str, source: str, revision: str, cache_dir: str) -> str:
    if Path(model).expanduser().is_dir():
        return str(Path(model).expanduser().resolve())
    cache = str(Path(cache_dir).expanduser())
    ignored = ["assets/*", "*.md", "*.png", "*.jpg"]
    if source == "modelscope":
        from modelscope_hub.compat import snapshot_download

        return snapshot_download(
            model, revision=revision, cache_dir=cache, ignore_file_pattern=ignored
        )
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision, cache_dir=cache, ignore_patterns=ignored)
