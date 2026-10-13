from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .config import DeploymentConfig, JobRequest, load_deployment
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
        path: str = Query(default="", max_length=4096), offset: int = Query(default=0, ge=0)
    ):
        try:
            root = runtime.safe_path(path) if path else runtime.roots[0]
            if not root.is_dir():
                raise ValueError("目录不存在")
            children = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink())
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
    parser = argparse.ArgumentParser(
        description="Start API, download weights and deploy GPU workers"
    )
    parser.add_argument("--config", default="config/deployment.toml")
    parser.add_argument("--gpus", help="Override GPU indices, e.g. 1,2,3")
    parser.add_argument("--mock", action="store_true", help="Test only; copies images, no removal")
    args = parser.parse_args()
    config = load_deployment(args.config)
    overrides = {}
    if args.gpus:
        overrides["gpus"] = [int(value) for value in args.gpus.split(",")]
    if args.mock:
        overrides["backend"] = "mock"
    config = DeploymentConfig.model_validate(config.model_dump() | overrides)
    import uvicorn

    uvicorn.run(create_app(config), host=config.host, port=config.port, access_log=False)


if __name__ == "__main__":
    main()
