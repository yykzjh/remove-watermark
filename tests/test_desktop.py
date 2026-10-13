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

def test_navigation_and_multi_region_selection(tmp_path):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    api = FakeApi()
    window = MainWindow(api, {"capacity": 64, "roots": ["/home/zjh"], "backend": "mock",
                              "stage": "ready", "job": None})
    window.show()
    window.tabs.setCurrentIndex(1)
    app.processEvents()
    assert window.preset.count() == 2
    assert not window.region_controls.isEnabled()
    assert not window.canvas.selectable
    window.input_file.setText("/home/zjh/image.png")
    window.preview_path = "/home/zjh/image.png"
    window.preview_token = "token"
    pixmap = QPixmap(800, 400)
    pixmap.fill(Qt.GlobalColor.white)
    window.canvas.pixmap = pixmap
    window.preset.setCurrentIndex(1)
    app.processEvents()
    assert window.canvas.selectable
    rect = window.canvas.image_rect()
    for left in [.1, .6]:
        start = QPoint(int(rect.x() + left * rect.width()), int(rect.y() + .2 * rect.height()))
        end = QPoint(int(rect.x() + (left + .2) * rect.width()), int(rect.y() + .5 * rect.height()))
        QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(window.canvas, end)
        QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=end)
    assert len(window.parsed_regions()) == 2
    assert abs(window.parsed_regions()[0].box[0] - .1) < .01
    window.single_output.setText("/home/zjh/out")
    window.output_mode.setCurrentIndex(1)
    window.start_job()
    payload = api.calls[-1][3]
    assert payload["mode"] == "single"
    assert len(payload["options"]["regions"]) == 2
    assert payload["options"]["output_width"] == 1920
    window.preset.setCurrentIndex(0)
    assert not window.canvas.selectable
    assert not window.boxes.isEnabled()
    window.start_job()
    assert api.calls[-1][3]["options"]["regions"] == []
    assert api.calls[-1][3]["options"]["scope"] == "full_image"
    window.tabs.setCurrentIndex(0)
    window.input_dir.setText("/home/zjh/batch")
    window.output_dir.setText("/home/zjh/batch-out")
    window.start_job()
    assert api.calls[-1][3]["options"]["scope"] == "full_image"
    window.close()
