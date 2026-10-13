from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path

from .config import DeploymentConfig, JobRequest
from .images import scan_images, validate_directories
from .worker import worker_main


class Runtime:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.cancel_scan = threading.Event()
        self.stage = "starting"
        self.error = ""
        self.workers = []
        self.job = None
        self.pending = deque()
        self.ctx = mp.get_context("spawn")
        self.events = self.ctx.Queue()
        self.threads = []
        self.state_dir = Path(config.state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.roots = [Path(r).expanduser().resolve() for r in config.allowed_roots]
        self.model_revision = config.revision

    def start(self):
        thread = threading.Thread(target=self._bootstrap, daemon=True)
        self.threads.append(thread)
        thread.start()

    def _bootstrap(self):
        try:
            with self.lock:
                self.stage = "downloading"
            model_path = self.config.model
            if self.config.backend != "mock":
                from .download import download_model

                model_path = download_model(
                    self.config.model,
                    self.config.model_source,
                    self.config.revision,
                    self.config.cache_dir,
                )
                snapshot_name = Path(model_path).name
                if len(snapshot_name) == 40 and all(c in "0123456789abcdef" for c in snapshot_name):
                    self.model_revision = snapshot_name
            if self.stop.is_set():
                return
            gate = self.ctx.Semaphore(self.config.load_parallelism)
            with self.lock:
                self.stage = "loading"
                for gpu in self.config.gpus:
                    for _ in range(self.config.replicas_per_gpu):
                        inbox = self.ctx.Queue(maxsize=1)
                        index = len(self.workers)
                        process = self.ctx.Process(
                            target=worker_main,
                            args=(
                                index,
                                gpu,
                                self.config.model_dump(),
                                model_path,
                                inbox,
                                self.events,
                                gate,
                            ),
                            daemon=True,
                        )
                        process.start()
                        self.workers.append(
                            {
                                "process": process,
                                "queue": inbox,
                                "gpu": gpu,
                                "state": "loading",
                                "task": None,
                            }
                        )
            self._event_loop()
        except Exception as error:  # noqa: BLE001 - isolate worker/service failures
            with self.lock:
                self.stage, self.error = "failed", f"{type(error).__name__}: {error}"[:1000]

    def _event_loop(self):
        while not self.stop.is_set():
            try:
                event = self.events.get(timeout=0.2)
                with self.lock:
                    worker = self.workers[event["worker"]]
                    if event["type"] == "ready":
                        worker["state"] = "idle"
                    elif event["type"] == "fatal":
                        worker["state"] = "failed"
                        self.error = event["error"]
                        self.stage = "failed"
                    elif event["type"] == "result":
                        worker["state"], worker["task"] = "idle", None
                        if self.job and event["job"] == self.job["id"]:
                            self._record(event)
                    if all(w["state"] == "idle" for w in self.workers) and self.stage == "loading":
                        self.stage = "ready"
            except queue.Empty:
                pass
            with self.lock:
                for worker in self.workers:
                    if worker["state"] != "failed" and not worker["process"].is_alive():
                        if worker["task"] and self.job:
                            self._record(
                                {
                                    "file": worker["task"]["relative"],
                                    "ok": False,
                                    "error": "GPU 工作进程意外退出",
                                }
                            )
                        worker["state"], worker["task"] = "failed", None
                        self.error = "GPU 工作进程意外退出，请查看服务日志并重启服务"
                        self.stage = "failed"
                self._dispatch()

    def _record(self, result):
        job = self.job
        job["completed"] += 1
        job["succeeded" if result["ok"] else "failed"] += 1
        job["current"] = result["file"]
        if not result["ok"]:
            job["errors"] = (job["errors"] + [{"file": result["file"], "error": result["error"]}])[
                -20:
            ]
        with (self.state_dir / job["id"] / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        self._persist()

    def _dispatch(self):
        job = self.job
        if not job or job["state"] not in {"running", "paused", "cancelling"}:
            return
        busy = sum(w["state"] == "busy" for w in self.workers)
        if job["state"] == "running":
            if self.stage == "failed":
                job["state"] = "cancelling"
                job["service_failure"] = self.error
            else:
                for worker in self.workers:
                    if not self.pending or busy >= job["options"]["concurrency"]:
                        break
                    if worker["state"] == "idle":
                        task = self.pending.popleft()
                        worker["state"], worker["task"] = "busy", task
                        worker["queue"].put(
                            {"job": job["id"], "task": task, "options": job["options"]}
                        )
                        busy += 1
        if job["state"] == "cancelling" and busy == 0:
            job["state"] = "failed" if job.get("service_failure") else "cancelled"
            job["unprocessed"] = len(self.pending)
            self.pending.clear()
            self._finish()
        elif not self.pending and busy == 0 and job["state"] in {"running", "paused"}:
            job["state"] = "completed"
            self._finish()

    def _persist(self):
        if self.job:
            root = self.state_dir / self.job["id"]
            root.mkdir(exist_ok=True)
            temp = root / "summary.tmp"
            temp.write_text(json.dumps(self.job, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(root / "summary.json")

    def _finish(self):
        self.job["finished_at"] = time.time()
        self._persist()

    def safe_path(self, value: str):
        path = Path(value).expanduser().resolve()
        if not any(path == root or root in path.parents for root in self.roots):
            raise ValueError("目录不在 deployment.toml 的 allowed_roots 内")
        return path

    def submit(self, request: JobRequest):
        with self.lock:
            if self.stage != "ready":
                raise ValueError("模型尚未就绪")
            if self.job and self.job["state"] in {"scanning", "running", "paused", "cancelling"}:
                raise ValueError("已有任务正在处理")
            if request.options.concurrency > self.config.capacity:
                raise ValueError(f"并发度不能超过已部署工作进程数量 {self.config.capacity}")
            source, target = validate_directories(
                str(self.safe_path(request.input_dir)), str(self.safe_path(request.output_dir))
            )
            self.cancel_scan.clear()
            self.job = {
                "id": uuid.uuid4().hex,
                "state": "scanning",
                "total": 0,
                "completed": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
                "invalid": 0,
                "errors": [],
                "current": "",
                "started_at": time.time(),
                "input_dir": str(source),
                "output_dir": str(target),
                "options": request.options.model_dump(),
                "backend": self.config.backend,
                "model": self.config.model,
                "model_revision": self.model_revision,
                "model_source": self.config.model_source,
            }
            self._persist()
            thread = threading.Thread(target=self._scan, args=(source, target), daemon=True)
            self.threads.append(thread)
            thread.start()
            return self.job["id"]

    def _scan(self, source, target):
        try:
            tasks, skipped, invalid = scan_images(
                source, target, self.job["options"]["overwrite"], self.cancel_scan
            )
            with self.lock:
                self.pending = deque(asdict(t) for t in tasks)
                self.job.update(
                    total=len(tasks), skipped=skipped, invalid=len(invalid), errors=invalid[-20:]
                )
                self.job["state"] = "cancelling" if self.cancel_scan.is_set() else "running"
                (self.state_dir / self.job["id"] / "invalid.json").write_text(
                    json.dumps(invalid, ensure_ascii=False), encoding="utf-8"
                )
                self._persist()
        except Exception as error:  # noqa: BLE001 - isolate worker/service failures
            with self.lock:
                self.job["state"] = "failed"
                self.job["errors"].append({"file": "", "error": str(error)[:500]})
                self._finish()

    def control(self, action):
        with self.lock:
            if not self.job:
                raise ValueError("没有任务")
            state = self.job["state"]
            if action == "pause" and state == "running":
                self.job["state"] = "paused"
            elif action == "resume" and state == "paused":
                self.job["state"] = "running"
            elif action == "cancel" and state in {"scanning", "running", "paused", "cancelling"}:
                self.cancel_scan.set()
                if state != "scanning":
                    self.job["state"] = "cancelling"
            else:
                raise ValueError("当前状态不支持此操作")
            self._persist()

    def status(self):
        with self.lock:
            return {
                "stage": self.stage,
                "error": self.error,
                "backend": self.config.backend,
                "workers": [{"gpu": w["gpu"], "state": w["state"]} for w in self.workers],
                "capacity": self.config.capacity,
                "model": self.config.model,
                "revision": self.model_revision,
                "roots": list(map(str, self.roots)),
                "job": (
                    {
                        k: (v[-3:] if k == "errors" else v)
                        for k, v in self.job.items()
                        if k
                        not in {"options", "input_dir", "output_dir", "model_revision", "model"}
                    }
                    if self.job
                    else None
                ),
            }

    def close(self):
        self.stop.set()
        self.cancel_scan.set()
        for worker in self.workers:
            if worker["process"].is_alive():
                worker["process"].terminate()
        for worker in self.workers:
            worker["process"].join(timeout=3)
        with self.lock:
            if self.job and self.job["state"] in {"scanning", "running", "paused", "cancelling"}:
                self.job["state"] = "interrupted"
                self._persist()
