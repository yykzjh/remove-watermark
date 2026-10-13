from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import httpx
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import ProcessingConfig, Region

STYLE = """
QWidget { background: #f4f6f5; color: #182d2a; font-size: 14px; font-family: "Noto Sans CJK SC", sans-serif; }
QMainWindow { background: #f4f6f5; }
QGroupBox { border: 1px solid #d7e1dc; border-radius: 9px; margin-top: 18px;
            padding: 18px 12px 12px; font-weight: 600; background: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QListWidget {
    background: #ffffff; border: 1px solid #cad6d0; border-radius: 5px; padding: 6px;
}
QPushButton { background: #e5eee9; border: 1px solid #c6d7cc; border-radius: 6px;
              padding: 8px 18px; }
QPushButton:hover { background: #d4e6dc; }
QPushButton:disabled { color: #89958f; background: #edf0ee; }
QPushButton#primary { background: #176f55; color: white; border: none; font-weight: 600; }
QProgressBar { border: none; background: #dde6e0; border-radius: 6px;
               text-align: center; min-height: 24px; }
QProgressBar::chunk { background: #35a37b; border-radius: 6px; }
QLabel#muted { color: #61766d; }
QLabel#title { font-size: 27px; font-weight: 700; }
"""


class Api(QObject):
    done = Signal(str, object, str)

    def __init__(self, url):
        super().__init__()
        self.url = url.rstrip("/")
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.closed = False

    def send(self, name, method, path, payload=None):
        def work():
            try:
                with httpx.Client(timeout=10, trust_env=False) as client:
                    response = client.request(method, self.url + path, json=payload)
                    response.raise_for_status()
                    data, error = response.json(), ""
            except Exception as exc:  # noqa: BLE001 - report errors across GUI thread boundary
                data = None
                error = str(exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        error = str(exc.response.json().get("detail", error))
                    except ValueError:
                        pass
            if not self.closed:
                self.done.emit(name, data, error[:1000])

        self.pool.submit(work)

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)


class DirectoryDialog(QDialog):
    def __init__(self, api, initial, roots, parent=None):
        super().__init__(parent)
        self.api, self.selected, self.parent_path = api, "", None
        self.request_name = "browse-" + str(id(self))
        self.next_offset = None
        self.setWindowTitle("选择开发机目录")
        self.resize(600, 480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("可输入新输出目录；输入目录必须已经存在。"))
        self.path = QLineEdit(initial or roots[0])
        layout.addWidget(self.path)
        row = QHBoxLayout()
        for title, action in [("打开路径", self.load), ("上一级", self.up)]:
            button = QPushButton(title)
            button.clicked.connect(action)
            row.addWidget(button)
        self.roots = QComboBox()
        self.roots.addItems(roots)
        self.roots.activated.connect(lambda: self.load(self.roots.currentText()))
        row.addWidget(self.roots)
        layout.addLayout(row)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(
            lambda item: self.load(self.path.text().rstrip("/") + "/" + item.text())
        )
        layout.addWidget(self.list)
        self.status = QLabel()
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.more = QPushButton("更多目录")
        self.more.clicked.connect(self.load_more)
        buttons.addWidget(self.more)
        choose = QPushButton("选择此路径")
        choose.setObjectName("primary")
        choose.clicked.connect(self.choose)
        buttons.addWidget(choose)
        layout.addLayout(buttons)
        self.api.done.connect(self.result)
        self.finished.connect(lambda: self.api.done.disconnect(self.result))
        self.load()

    def load(self, path=None):
        if isinstance(path, str):
            self.path.setText(path)
        self.list.clear()
        self.fetch(0)

    def fetch(self, offset):
        from urllib.parse import urlencode

        self.status.setText("正在读取目录…")
        self.api.send(
            self.request_name,
            "GET",
            "/directories?" + urlencode({"path": self.path.text(), "offset": offset}),
        )

    def load_more(self):
        if self.next_offset is not None:
            self.fetch(self.next_offset)

    def up(self):
        if self.parent_path:
            self.load(self.parent_path)

    def result(self, name, data, error):
        if name != self.request_name:
            return
        if error:
            self.status.setText(error)
            return
        self.path.setText(data["path"])
        self.parent_path = data["parent"]
        self.next_offset = data["next_offset"]
        self.more.setEnabled(self.next_offset is not None)
        self.list.addItems(data["dirs"])
        self.status.setText(f"已显示 {self.list.count()} 个子目录")

    def choose(self):
        self.selected = self.path.text().strip()
        if self.selected:
            self.accept()


class MainWindow(QMainWindow):
    def __init__(self, api, state):
        super().__init__()
        self.api, self.capacity, self.roots = api, state["capacity"], state["roots"]
        self.job_state = ""
        self.setWindowTitle("批量图像去水印 · Qwen Image")
        self.resize(880, 870)
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel("批量图像去水印")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel(f"Qwen-Image-2.1-Turbo  ·  {self.capacity} 个 GPU 工作进程已就绪")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        if state["backend"] == "mock":
            layout.addWidget(QLabel("测试模式：只复制图片，不执行模型去水印。"))
        folders = QGroupBox("开发机目录")
        form = QFormLayout(folders)
        self.input_dir, self.output_dir = QLineEdit(), QLineEdit()
        for label, edit in [("输入目录", self.input_dir), ("输出目录", self.output_dir)]:
            edit.setPlaceholderText(
                "例如 /home/zjh/images"
                if edit is self.input_dir
                else "例如 /home/zjh/images-restored"
            )
            row = QHBoxLayout()
            row.addWidget(edit)
            button = QPushButton("浏览")
            button.clicked.connect(lambda checked=False, field=edit: self.browse(field))
            row.addWidget(button)
            form.addRow(label, row)
        layout.addWidget(folders)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.metrics = QLabel("选择目录后开始处理。输出保留原有子目录和文件名。")
        layout.addWidget(self.metrics)
        self.current = QLabel("")
        self.current.setObjectName("muted")
        self.current.setWordWrap(True)
        layout.addWidget(self.current)

        controls = QGroupBox("批处理设置")
        form = QFormLayout(controls)
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, self.capacity)
        self.concurrency.setValue(self.capacity)
        form.addRow("并发图片数", self.concurrency)
        self.resolution = QComboBox()
        for size in [512, 768, 1024, 1280, 1536, 2048]:
            self.resolution.addItem(str(size), size)
        self.resolution.setCurrentIndex(2)
        form.addRow("修复区域推理分辨率", self.resolution)
        self.preset = QComboBox()
        for title, value in [
            ("自动识别水印位置（逐张检测，无需框选）", "auto"),
            ("灰底黑字（样例 1）", "gray"),
            ("蓝底白字 + 灰底正文（样例 2）", "blue"),
            ("自定义区域", "custom"),
        ]:
            self.preset.addItem(title, value)
        form.addRow("水印定位方式", self.preset)
        self.boxes = QLineEdit("0.012,0.727,0.45,0.988; 0.864,0.89,1,1")
        self.boxes.setToolTip("每个区域为 x0,y0,x1,y1，范围 0..1；分号分隔多个区域")
        self.boxes.setEnabled(False)
        self.preset.currentIndexChanged.connect(
            lambda: self.boxes.setEnabled(self.preset.currentData() == "custom")
        )
        form.addRow("自定义区域", self.boxes)
        note = QLabel("默认逐图识别任意位置的数字水印；未检测到则原样复制并单独计数。固定模板仅供已知布局使用。")
        note.setObjectName("muted")
        note.setWordWrap(True)
        form.addRow(note)
        self.detection_resolution = QComboBox()
        for size in [768, 1024, 1280, 1536, 2048]:
            self.detection_resolution.addItem(str(size), size)
        self.detection_resolution.setCurrentIndex(2)
        form.addRow("自动检测分辨率（长边）", self.detection_resolution)
        self.logo = QCheckBox("同时去除相机标识及附属小字（任意位置）")
        self.logo.setChecked(True)
        self.overwrite = QCheckBox("覆盖输出目录中已有同名文件")
        row = QHBoxLayout()
        row.addWidget(self.logo)
        row.addWidget(self.overwrite)
        form.addRow(row)
        self.context = QDoubleSpinBox()
        self.context.setRange(0.1, 2.0)
        self.context.setSingleStep(0.1)
        self.context.setValue(0.6)
        self.feather = QSpinBox()
        self.feather.setRange(0, 64)
        self.feather.setValue(12)
        row = QHBoxLayout()
        row.addWidget(QLabel("上下文扩展比例"))
        row.addWidget(self.context)
        row.addWidget(QLabel("边缘过渡像素"))
        row.addWidget(self.feather)
        form.addRow(row)
        self.extra_prompt = QLineEdit()
        self.extra_prompt.setPlaceholderText("可选：针对特殊地面、建筑线条等补充说明")
        form.addRow("补充提示词", self.extra_prompt)
        layout.addWidget(controls)
        self.controls = controls
        self.folders = folders
        row = QHBoxLayout()
        self.start_button = QPushButton("开始批处理")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_job)
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause_job)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(
            lambda: self.api.send("control", "POST", "/jobs/cancel", {})
        )
        row.addWidget(self.start_button)
        row.addWidget(self.pause_button)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.errors = QTextEdit()
        self.errors.setReadOnly(True)
        self.errors.setMaximumHeight(85)
        self.errors.setPlaceholderText("异常摘要将显示在这里；详细逐图报告保存在开发机。")
        layout.addWidget(self.errors)
        self.api.done.connect(self.result)
        self.update_state(state)

    def browse(self, field):
        dialog = DirectoryDialog(self.api, field.text(), self.roots, self)
        if dialog.exec():
            field.setText(dialog.selected)

    def start_job(self):
        try:
            regions = []
            if self.preset.currentData() == "custom":
                regions = [
                    Region(box=tuple(float(v.strip()) for v in box.split(",")))
                    for box in self.boxes.text().split(";")
                    if box.strip()
                ]
            options = ProcessingConfig(
                concurrency=self.concurrency.value(),
                resolution=self.resolution.currentData(),
                preset=self.preset.currentData(),
                regions=regions,
                detection_resolution=self.detection_resolution.currentData(),
                remove_logo=self.logo.isChecked(),
                overwrite=self.overwrite.isChecked(),
                context=self.context.value(),
                feather=self.feather.value(),
                extra_prompt=self.extra_prompt.text(),
            )
            if not self.input_dir.text().strip() or not self.output_dir.text().strip():
                raise ValueError("请选择输入和输出目录")
            self.start_button.setEnabled(False)
            self.api.send(
                "submit",
                "POST",
                "/jobs",
                {
                    "input_dir": self.input_dir.text().strip(),
                    "output_dir": self.output_dir.text().strip(),
                    "options": options.model_dump(),
                },
            )
        except ValueError as error:
            QMessageBox.warning(self, "参数有误", str(error))

    def pause_job(self):
        self.api.send(
            "control", "POST", "/jobs/" + ("resume" if self.job_state == "paused" else "pause"), {}
        )

    def result(self, name, data, error):
        if name in {"submit", "control"} and error:
            QMessageBox.warning(self, "操作失败", error)
            self.start_button.setEnabled(True)
        elif name == "poll":
            if error:
                self.current.setText("服务连接中断，正在重试。任务继续在开发机执行。")
            else:
                self.update_state(data)

    def update_state(self, state):
        job = state.get("job")
        if state["stage"] == "failed":
            self.errors.setPlainText(state.get("error", "服务异常"))
        self.job_state = job["state"] if job else ""
        active = self.job_state in {"scanning", "running", "paused", "cancelling"}
        self.start_button.setEnabled(not active and state["stage"] == "ready")
        self.folders.setEnabled(not active)
        self.controls.setEnabled(not active)
        self.pause_button.setEnabled(self.job_state in {"running", "paused"})
        self.pause_button.setText("继续" if self.job_state == "paused" else "暂停")
        self.cancel_button.setEnabled(self.job_state in {"scanning", "running", "paused"})
        if not job:
            return
        names = {
            "scanning": "扫描目录",
            "running": "处理中",
            "paused": "已暂停",
            "cancelling": "等待当前图片完成后取消",
            "cancelled": "已取消",
            "completed": "处理结束",
            "failed": "失败",
            "interrupted": "已中断",
        }
        total = job["total"]
        self.progress.setRange(0, 0 if self.job_state == "scanning" else max(1, total))
        self.progress.setValue(job["completed"])
        self.metrics.setText(
            f"{names.get(self.job_state, self.job_state)}  ·  {job['completed']}/{total}  "
            f"成功 {job['succeeded']}（未检出 {job.get('no_watermark', 0)}）  失败 {job['failed']}  "
            f"已有文件跳过 {job['skipped']}  无效图片 {job['invalid']}"
        )
        self.current.setText(job.get("current", ""))
        self.errors.setPlainText(
            "\n".join(f"{e['file']}: {e['error']}" for e in job.get("errors", []))
        )


class Startup(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Watermark Studio · 模型部署")
        self.resize(620, 210)
        layout = QVBoxLayout(self)
        title = QLabel("正在准备图像修复服务")
        title.setObjectName("title")
        layout.addWidget(title)
        self.label = QLabel("正在连接服务…")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("模型下载及加载完成后，批处理窗口将自动打开。"))


def main():
    parser = argparse.ArgumentParser(
        description="Linux desktop; launch a local service or connect remotely"
    )
    parser.add_argument(
        "--url", help="Connect to an existing API instead of starting a local service"
    )
    parser.add_argument("--config", default="config/deployment.toml")
    parser.add_argument("--ssh-host", help="Open metadata-only tunnel and start the remote service")
    parser.add_argument("--remote-dir", default="~/project/inference/remove-watermark")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--remote-port", type=int, default=8765)
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    splash = Startup()
    splash.show()
    processes = []
    log = tempfile.NamedTemporaryFile(prefix="watermark-studio-", suffix=".log", delete=False)  # noqa: SIM115 - closed in finally
    local_backend = None
    if args.ssh_host:
        if not 1024 <= args.port <= 65535:
            parser.error("port must be 1024..65535")
        remote = args.remote_dir
        remote = (
            '"$HOME"/' + shlex.quote(remote[2:]) if remote.startswith("~/") else shlex.quote(remote)
        )
        command = f"cd {remote} && bash scripts/start-server.sh {shlex.quote(args.config)}"
        processes.append(
            subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", args.ssh_host, command], stdout=log, stderr=log
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-N",
                    "-L",
                    f"{args.port}:127.0.0.1:{args.remote_port}",
                    args.ssh_host,
                ],
                stdout=log,
                stderr=log,
            )
        )
        url = f"http://127.0.0.1:{args.port}"
    elif args.url:
        url = args.url
    else:
        from .config import load_deployment

        deployment = load_deployment(args.config)
        url = f"http://127.0.0.1:{deployment.port}"
        local_backend = subprocess.Popen(
            [sys.executable, "-m", "watermark_studio.server", "--config", args.config],
            stdout=log,
            stderr=log,
        )
    api = Api(url)
    window = None
    pending = False
    failures = 0

    def poll():
        nonlocal pending
        if not pending:
            pending = True
            api.send("poll", "GET", "/health")

    def receive(name, state, error):
        nonlocal window, pending, failures
        if name != "poll":
            return
        pending = False
        if error:
            failures += 1
            splash.label.setText(f"等待服务启动（第 {failures} 次检查）\n本地连接日志：{log.name}")
            if (local_backend and local_backend.poll() is not None) or any(
                p.poll() not in (None, 0) for p in processes
            ):
                splash.label.setText(f"服务启动失败。请查看日志：{log.name}")
                splash.progress.setRange(0, 1)
                timer.stop()
            return
        failures = 0
        stage = state["stage"]
        if stage == "ready" and window is None:
            window = MainWindow(api, state)
            window.show()
            splash.close()
        elif stage == "failed":
            splash.label.setText("模型部署失败：\n" + state["error"])
            splash.progress.setRange(0, 1)
        else:
            names = {
                "starting": "启动服务",
                "downloading": "下载模型权重",
                "loading": "加载 GPU 工作进程",
            }
            ready = sum(w["state"] == "idle" for w in state["workers"])
            splash.label.setText(f"{names.get(stage, stage)} · {ready}/{state['capacity']} 已就绪")

    api.done.connect(receive)
    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(max(3, args.poll_seconds) * 1000)
    poll()
    try:
        app.exec()
    finally:
        timer.stop()
        api.close()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        if local_backend and local_backend.poll() is None:
            local_backend.terminate()
        log.close()


if __name__ == "__main__":
    main()
