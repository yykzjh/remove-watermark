import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from watermark_studio.desktop import STYLE, MainWindow


class FakeApi(QObject):
    done = Signal(str, object, str)

    def __init__(self):
        super().__init__()
        self.calls = []

    def send(self, *args):
        self.calls.append(args)


def test_gui_builds_valid_remote_job_and_tracks_progress(tmp_path):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    api = FakeApi()
    state = {
        "capacity": 2,
        "roots": ["/home/zjh"],
        "backend": "mock",
        "stage": "ready",
        "job": None,
    }
    window = MainWindow(api, state)
    window.input_dir.setText("/home/zjh/input")
    window.output_dir.setText("/home/zjh/output")
    window.start_job()
    assert api.calls[-1][3]["options"]["concurrency"] == 2
    state["job"] = {
        "state": "running",
        "total": 10,
        "completed": 3,
        "succeeded": 3,
        "failed": 0,
        "skipped": 0,
        "invalid": 0,
        "errors": [],
        "current": "deep/photo.png",
    }
    window.update_state(state)
    assert window.progress.value() == 3
    assert not window.start_button.isEnabled()
    assert window.pause_button.isEnabled()
    window.show()
    app.processEvents()
    assert window.grab().save(str(tmp_path / "desktop.png"))
    window.close()
