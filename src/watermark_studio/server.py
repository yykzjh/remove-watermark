from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .config import DeploymentConfig, JobRequest
from .deployment_cli import parse_deployment, render_config
from .runtime import Runtime


def create_app(config: DeploymentConfig):
    runtime = Runtime(config)

    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        yield
        runtime.close()

    app = FastAPI(title="Watermark Studio", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.runtime = runtime

    @app.get("/health")
    def health():
        return runtime.status()

    @app.get("/directories")
    def directories(
        path: str = Query(default="", max_length=4096), offset: int = Query(default=0, ge=0),
        files: bool = False
    ):
        try:
            root = runtime.safe_path(path) if path else runtime.roots[0]
            if not root.is_dir():
                raise ValueError("目录不存在")
            from .images import EXTENSIONS
            children = sorted(
                p.name + ("/" if p.is_dir() else "") for p in root.iterdir()
                if not p.is_symlink() and (p.is_dir() or (files and p.suffix.lower() in EXTENSIONS))
            )
            try:
                parent = str(runtime.safe_path(str(root.parent)))
            except ValueError:
                parent = None
            return {
                "path": str(root),
                "parent": parent,
                "dirs": children[offset : offset + 100],
                "next_offset": offset + 100 if offset + 100 < len(children) else None,
            }
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/preview")
    def preview(path: str = Query(min_length=1, max_length=4096)):
        # Explicit single-image requests only; never prefetch images or full originals.
        import base64
        import io

        from PIL import Image, ImageOps

        from .images import EXTENSIONS
        try:
            source = runtime.safe_path(path)
            if source.suffix.lower() not in EXTENSIONS:
                raise ValueError("不支持该图像格式")
            with Image.open(source) as raw:
                if getattr(raw, "n_frames", 1) != 1:
                    raise ValueError("不支持多帧图像")
                image = ImageOps.exif_transpose(raw).convert("RGB")
            original_size = image.size
            image.thumbnail((1024, 1024))
            while True:
                stream = io.BytesIO()
                image.save(stream, format="JPEG", quality=65, optimize=True)
                data = stream.getvalue()
                if len(data) <= 64 * 1024:
                    break
                image.thumbnail((max(1, image.width * 3 // 4), max(1, image.height * 3 // 4)))
            stat = source.stat()
            return {"path": str(source), "size": original_size,
                    "token": f"{stat.st_size}:{stat.st_mtime_ns}",
                    "jpeg": base64.b64encode(data).decode("ascii"), "bytes": len(data)}
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/jobs", status_code=202)
    def submit(request: JobRequest):
        try:
            return {"id": runtime.submit(request)}
        except (ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/jobs/{action}")
    def control(action: str):
        try:
            runtime.control(action)
            return {"ok": True}
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    return app


def main():
    config, print_config = parse_deployment()
    if print_config:
        print(render_config(config))
        return
    import uvicorn

    uvicorn.run(create_app(config), host=config.host, port=config.port, access_log=False)


if __name__ == "__main__":
    main()
