from __future__ import annotations

import argparse
import base64
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PySide6.QtCore import QObject, QSize, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
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
    def __init__(self, api, initial, roots, parent=None, files=False):
        super().__init__(parent)
        self.files = files
        self.api, self.selected, self.parent_path = api, "", None
        self.request_name = "browse-" + str(id(self))
        self.next_offset = None
        self.setWindowTitle("选择开发机图像" if files else "选择开发机目录")
        self.resize(600, 480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("双击图像打开；双击目录进入。" if files else "可输入新输出目录；输入目录必须已经存在。"))
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
            lambda item: self.activate_item(item.text())
        )
        layout.addWidget(self.list)
        self.status = QLabel()
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.more = QPushButton("更多目录")
        self.more.clicked.connect(self.load_more)
        buttons.addWidget(self.more)
        choose = QPushButton("选择图像路径" if files else "选择此路径")
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
            "/directories?" + urlencode({"path": self.path.text(), "offset": offset, "files": str(self.files).lower()}),
        )

    def activate_item(self, name):
        path = self.path.text().rstrip("/") + "/" + name.rstrip("/")
        if self.files and not name.endswith("/"):
            self.selected = path
            self.accept()
        else:
            self.load(path)

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
        item = self.list.currentItem()
        if self.files and item and not item.text().endswith("/"):
            self.activate_item(item.text())
            return
        self.selected = self.path.text().strip()
        if self.selected:
            self.accept()


class WorkflowTabs(QTabWidget):
    """Size the form for the active workflow, without reserving a hidden preview."""

    def sizeHint(self):
        page = self.currentWidget()
        if page is None:
            return super().sizeHint()
        content = page.layout().sizeHint().expandedTo(page.layout().minimumSize())
        navigation = self.tabBar().sizeHint()
        return QSize(max(content.width(), navigation.width()) + 8,
                     content.height() + navigation.height() + 8)

    def minimumSizeHint(self):
        return self.sizeHint()


class MainWindow(QMainWindow):
    def __init__(self, api, state):
        super().__init__()
        from .canvas import RegionCanvas
        self.api, self.capacity, self.roots = api, state["capacity"], state["roots"]
        self.job_state = ""
        self.preview_path = self.preview_token = ""
        self.preview_request = ""
        self.setWindowTitle("图像去水印 · Qwen Image")
        self.resize(1000, 940)
        body = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        shell = QWidget()
        outer_layout = QVBoxLayout(shell)
        outer_layout.addWidget(scroll)
        self.setCentralWidget(shell)
        layout = QVBoxLayout(body)
        title = QLabel("图像去水印")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(QLabel(f"开发机服务 · {state.get('worker_count', 8)} 个 GPU 工作进程"))
        optimization = state.get("optimization", {})
        if optimization.get("quantization", "none") != "none":
            precision = {"float8": "FP8", "int8": "INT8"}.get(optimization["quantization"], optimization["quantization"])
            layout.addWidget(QLabel(f"当前采用 {precision} 量化推理：吞吐优先，生成细节可能与 BF16 不同。"))
        if optimization.get("compile"):
            layout.addWidget(QLabel("已启用模型编译；首次处理及新输入形状可能需要额外准备时间。"))
        if state["backend"] == "mock":
            layout.addWidget(QLabel("测试模式：不执行模型去水印。"))
        self.tabs = WorkflowTabs()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.tabs)
        batch = QWidget()
        self.tabs.addTab(batch, "批量去除水印")
        batch_form = QFormLayout(batch)
        batch_form.addRow(QLabel("逐张处理完整图像，自动去除任意位置的水印。递归遍历目录并保留输出目录结构。"))
        self.input_dir, self.output_dir = QLineEdit(), QLineEdit()
        for label, edit in [("输入目录", self.input_dir), ("输出目录", self.output_dir)]:
            row = QHBoxLayout()
            row.addWidget(edit)
            button = QPushButton("浏览开发机")
            button.clicked.connect(lambda checked=False, field=edit: self.browse(field))
            row.addWidget(button)
            batch_form.addRow(label, row)
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, self.capacity)
        self.concurrency.setValue(min(self.capacity, state.get("recommended_concurrency", 32)))
        self.concurrency.setToolTip("全服务在途原图上限，包含等待凑批、推理和保存。每卡 batch 上限由服务启动参数设置。")
        batch_form.addRow("全局在途图片数", self.concurrency)
        single = QWidget()
        self.tabs.addTab(single, "单张去除水印")
        single_form = QFormLayout(single)
        self.input_file = QLineEdit()
        self.input_file.setPlaceholderText("开发机上的图像文件路径")
        row = QHBoxLayout()
        row.addWidget(self.input_file)
        browse = QPushButton("浏览图像")
        browse.clicked.connect(self.browse_image)
        row.addWidget(browse)
        self.open_button = QPushButton("打开预览")
        self.open_button.clicked.connect(self.open_preview)
        row.addWidget(self.open_button)
        single_form.addRow("输入图像", row)
        self.single_output = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(self.single_output)
        browse_out = QPushButton("浏览开发机")
        browse_out.clicked.connect(lambda: self.browse(self.single_output))
        row.addWidget(browse_out)
        single_form.addRow("输出目录", row)
        self.preset = QComboBox()
        self.preset.addItem("自动识别水印位置", "auto")
        self.preset.addItem("自定义区域", "custom")
        single_form.addRow("处理模式", self.preset)
        self.mode_help = QLabel()
        self.mode_help.setWordWrap(True)
        single_form.addRow(self.mode_help)
        self.canvas = RegionCanvas()
        single_form.addRow(self.canvas)
        self.preview_status = QLabel("仅按需传输压缩预览；处理始终使用开发机原图。")
        single_form.addRow(self.preview_status)
        self.region_controls = QWidget()
        region_form = QFormLayout(self.region_controls)
        self.boxes = QLineEdit()
        self.boxes.setPlaceholderText("x0,y0,x1,y1; x0,y0,x1,y1（0..1，最多 8 个）")
        self.boxes.setToolTip("相对于完整图像的归一化坐标；可拖动鼠标添加多个矩形，也可手动输入。")
        self.boxes.editingFinished.connect(self.apply_boxes)
        region_form.addRow("区域坐标", self.boxes)
        buttons = QHBoxLayout()
        undo = QPushButton("删除最后一个区域")
        undo.clicked.connect(self.undo_region)
        clear = QPushButton("清空区域")
        clear.clicked.connect(self.canvas.clear)
        buttons.addWidget(undo)
        buttons.addWidget(clear)
        region_form.addRow(buttons)
        single_form.addRow(self.region_controls)
        self.canvas.changed.connect(self.boxes_changed)
        self.preset.currentIndexChanged.connect(self.mode_changed)
        self.input_file.textChanged.connect(self.invalidate_preview)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.folders = self.tabs

        controls = QGroupBox("分辨率与输出")
        form = QFormLayout(controls)
        self.resolution = QComboBox()
        for size in [512, 768, 1024, 1280, 1536, 2048]:
            self.resolution.addItem(f"{size} 档（约 {size} × {size} 像素预算）", size)
        self.resolution.setCurrentIndex(2)
        self.resolution.setToolTip("保留送入模型的图像宽高比，并按模型网格对齐。较低档位省显存；更大 batch 需调整服务参数。")
        form.addRow("模型输入分辨率", self.resolution)
        input_note = QLabel("输入档位影响效果、显存和可用 batch 大小；实际宽高保持输入比例并对齐模型网格，不决定最终整图尺寸。")
        input_note.setWordWrap(True)
        input_note.setObjectName("muted")
        form.addRow(input_note)
        self.output_mode = QComboBox()
        self.output_mode.addItems(["每张图像的原始分辨率", "自定义完整图像分辨率"])
        form.addRow("输出图像分辨率", self.output_mode)
        self.output_dimensions = QWidget()
        row = QHBoxLayout(self.output_dimensions)
        self.output_width, self.output_height = QSpinBox(), QSpinBox()
        for widget, value in [(self.output_width, 1920), (self.output_height, 1080)]:
            widget.setRange(1, 16384)
            widget.setValue(value)
        row.addWidget(QLabel("宽"))
        row.addWidget(self.output_width)
        row.addWidget(QLabel("高"))
        row.addWidget(self.output_height)
        row.addWidget(QLabel("像素 · 完整图像"))
        form.addRow(self.output_dimensions)
        self.output_dimensions.setEnabled(False)
        self.output_mode.currentIndexChanged.connect(lambda index: self.output_dimensions.setEnabled(index == 1))
        note = QLabel("先完成去水印与整图合成，再以 Lanczos 高质量重采样缩放。自定义宽高会精确匹配尺寸；比例不同会拉伸。此方式不是 AI 超分辨率。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        form.addRow(note)
        self.overwrite = QCheckBox("覆盖输出目录中的同名文件")
        form.addRow(self.overwrite)
        self.extra_prompt = QLineEdit()
        self.extra_prompt.setPlaceholderText("可选：补充地面、人物衣着、建筑线条等场景说明")
        form.addRow("补充提示词", self.extra_prompt)
        layout.addWidget(controls)
        layout.addStretch()
        self.controls = controls
        layout = outer_layout
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.metrics = QLabel("请选择开发机上的输入与输出路径。")
        layout.addWidget(self.metrics)
        self.current = QLabel()
        self.current.setWordWrap(True)
        layout.addWidget(self.current)
        row = QHBoxLayout()
        self.start_button = QPushButton("开始批量去除水印")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_job)
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause_job)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(lambda: self.api.send("control", "POST", "/jobs/cancel", {}))
        for button in [self.start_button, self.pause_button, self.cancel_button]:
            row.addWidget(button)
        layout.addLayout(row)
        self.errors = QTextEdit()
        self.errors.setReadOnly(True)
        self.errors.setMaximumHeight(65)
        layout.addWidget(self.errors)
        self.api.done.connect(self.result)
        self.mode_changed()
        self.update_state(state)

    def tab_changed(self):
        self.tabs.updateGeometry()
        self.start_button.setText("开始单张去除水印" if self.tabs.currentIndex() else "开始批量去除水印")

    def mode_changed(self):
        custom = self.preset.currentData() == "custom"
        self.mode_help.setText(
            "拖动鼠标添加多个水印框，或输入区域坐标；处理后合并回整图，再调整输出尺寸。"
            if custom else "自动处理整张图像中的水印；手动框选、区域坐标和区域编辑按钮均已禁用。"
        )
        self.region_controls.setEnabled(custom)
        self.canvas.set_selectable(custom and bool(self.preview_path))

    def boxes_changed(self, boxes):
        self.boxes.setText("; ".join(",".join(f"{v:.6f}" for v in box) for box in boxes))

    def parsed_regions(self):
        return [Region(box=tuple(float(v.strip()) for v in box.split(",")))
                for box in self.boxes.text().split(";") if box.strip()]

    def apply_boxes(self):
        try:
            regions = self.parsed_regions()
            if len(regions) > 8:
                raise ValueError("最多选择 8 个区域")
            self.canvas.boxes = [r.box for r in regions]
            self.canvas.update()
        except ValueError as error:
            self.preview_status.setText("区域坐标无效：" + str(error)[:160])

    def undo_region(self):
        self.canvas.boxes = self.canvas.boxes[:-1]
        self.boxes_changed(self.canvas.boxes)
        self.canvas.update()

    def invalidate_preview(self):
        self.preview_request = ""
        self.open_button.setEnabled(True)
        self.preview_path = self.preview_token = ""
        self.canvas.pixmap = QPixmap()
        self.canvas.clear()
        self.mode_changed()

    def browse_image(self):
        dialog = DirectoryDialog(self.api, "", self.roots, self, files=True)
        if dialog.exec():
            self.input_file.setText(dialog.selected)
            self.open_preview()

    def open_preview(self):
        from urllib.parse import urlencode
        if not self.input_file.text().strip():
            return
        self.invalidate_preview()
        self.preview_request = "preview-" + str(time.monotonic_ns())
        self.open_button.setEnabled(False)
        self.preview_status.setText("正在读取开发机压缩预览…")
        self.api.send(self.preview_request, "GET", "/preview?" + urlencode({"path": self.input_file.text().strip()}))

    def browse(self, field):
        dialog = DirectoryDialog(self.api, field.text(), self.roots, self)
        if dialog.exec():
            field.setText(dialog.selected)

    def start_job(self):
        try:
            single = self.tabs.currentIndex() == 1
            custom = single and self.preset.currentData() == "custom"
            if custom and not self.preview_path:
                raise ValueError("请先打开图像预览，再选择或指定区域")
            options = ProcessingConfig(
                scope="regions" if custom else "full_image",
                concurrency=1 if single else self.concurrency.value(),
                resolution=self.resolution.currentData(),
                preset=self.preset.currentData() if single else "auto",
                regions=self.parsed_regions() if custom else [],
                remove_logo=True,
                overwrite=self.overwrite.isChecked(),
                output_width=self.output_width.value() if self.output_mode.currentIndex() else 0,
                output_height=self.output_height.value() if self.output_mode.currentIndex() else 0,
                extra_prompt=self.extra_prompt.text(),
            )
            source = self.input_file.text().strip() if single else self.input_dir.text().strip()
            destination = self.single_output.text().strip() if single else self.output_dir.text().strip()
            if not source or not destination:
                raise ValueError("请选择输入和输出路径")
            self.start_button.setEnabled(False)
            self.api.send("submit", "POST", "/jobs", {
                "mode": "single" if single else "batch",
                "input_dir": source,
                "input_file": self.preview_path or source if single else "",
                "preview_token": self.preview_token if single else "",
                "output_dir": destination,
                "options": options.model_dump(),
            })
        except ValueError as error:
            QMessageBox.warning(self, "参数有误", str(error))

    def pause_job(self):
        self.api.send(
            "control", "POST", "/jobs/" + ("resume" if self.job_state == "paused" else "pause"), {}
        )

    def result(self, name, data, error):
        if name == self.preview_request:
            self.open_button.setEnabled(True)
            if error:
                self.preview_status.setText(error)
            else:
                self.preview_path, self.preview_token = data["path"], data["token"]
                self.canvas.set_image(base64.b64decode(data["jpeg"]))
                self.preview_status.setText(
                    f"原图 {data['size'][0]} × {data['size'][1]} · 预览 {data['bytes'] // 1024} KiB · 最多 8 个区域"
                )
                self.mode_changed()
            return
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
            "cancelling": "等待当前凑批窗口完成后取消",
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
            f"已有文件跳过 {job['skipped']}  无效图片 {job['invalid']}  "
            f"在途 {state.get('inflight_images', 0)}"
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
        title = QLabel("正在准备图像去水印服务")
        title.setObjectName("title")
        layout.addWidget(title)
        self.label = QLabel("正在连接服务…")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("模型下载及加载完成后，批处理窗口将自动打开。"))


def default_ssh_program():
    # Use the host SSH transport in WSL; model, GUI and dependencies remain Linux.
    native = Path("/mnt/c/Windows/System32/OpenSSH/ssh.exe")
    release = Path("/proc/sys/kernel/osrelease")
    if native.is_file() and release.exists() and "microsoft" in release.read_text().lower():
        return str(native)
    return "ssh"


def main():
    parser = argparse.ArgumentParser(
        description="Linux desktop; launch a local service or connect remotely"
    )
    parser.add_argument(
        "--url", help="Connect to an existing API instead of starting a local service"
    )
    parser.add_argument("--config", default="config/deployment.toml")
    parser.add_argument("--ssh-program", default=default_ssh_program(), help="SSH executable; WSL prefers host OpenSSH when available")
    parser.add_argument("--ssh-host", help="Open status and bounded-preview tunnel and start the remote service")
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
        command = (
            f"cd {remote} && (.venv/bin/python scripts/check-health.py "
            f"--url http://127.0.0.1:{args.remote_port} >/dev/null 2>&1 || "
            f"bash scripts/launch-server.sh --config {shlex.quote(args.config)} "
            f"--port {args.remote_port})"
        )
        processes.append(
            subprocess.Popen(
                [args.ssh_program, "-o", "BatchMode=yes", args.ssh_host, command], stdout=log, stderr=log
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    args.ssh_program,
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-N",
                    "-L",
                    f"127.0.0.1:{args.port}:127.0.0.1:{args.remote_port}",
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
            splash.label.setText(f"{names.get(stage, stage)} · {ready}/{state.get('worker_count', len(state.get('workers', [])))} 已就绪")

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
