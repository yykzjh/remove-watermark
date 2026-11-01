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


def test_batch_layout_does_not_reserve_single_preview_space():
    from PySide6.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    window = MainWindow(FakeApi(), {
        "capacity": 64, "roots": ["/home/zjh"], "backend": "qwen",
        "stage": "ready", "job": None,
    })
    window.show()
    QTest.qWait(50)
    batch_height = window.tabs.height()
    assert not window.region_controls.isVisible()
    assert window.controls.isVisible()
    window.tabs.setCurrentIndex(1)
    QTest.qWait(50)
    assert window.tabs.height() > batch_height + 200
    assert "已禁用" in window.mode_help.text()
    assert not window.boxes.isEnabled()
    window.preset.setCurrentIndex(1)
    assert window.boxes.isEnabled()
    window.tabs.setCurrentIndex(0)
    QTest.qWait(50)
    assert window.tabs.height() == batch_height
    assert not window.region_controls.isVisible()
    window.close()

def test_explicit_quality_launch_defaults_do_not_submit_job():
    from types import SimpleNamespace

    from watermark_studio.desktop import apply_processing_defaults

    app = QApplication.instance() or QApplication([])
    api = FakeApi()
    window = MainWindow(api, {"capacity": 20, "roots": ["/mnt"], "backend": "qwen",
                              "stage": "ready", "job": None, "quality_verifier": True,
                              "cross_review_supported": True})
    apply_processing_defaults(window, SimpleNamespace(
        preserve_background=True, quality_check=True, quality_auto_retry=True, quality_cross_review=True,
        quality_batch_size=2, input_dir="/mnt/input", output_dir="/mnt/output",
    ))
    assert window.preserve_background.isChecked()
    assert window.quality_check.isChecked() and window.quality_retry.isChecked()
    assert window.quality_cross_review.isChecked()
    assert window.quality_batch.value() == 2
    assert window.input_dir.text() == "/mnt/input" and window.output_dir.text() == "/mnt/output"
    assert not any(call[1:3] == ("POST", "/jobs") for call in api.calls)
    app.processEvents()
    window.close()


def test_same_base_verifier_visibly_disables_cross_review_even_after_quality_toggle():
    app = QApplication.instance() or QApplication([])
    state={"capacity":20,"roots":["/mnt"],"backend":"qwen","stage":"ready","job":None,
           "quality_verifier":True,"cross_review_supported":False}
    window=MainWindow(FakeApi(),state)
    window.quality_check.setChecked(True)
    window.quality_cross_review.setChecked(True)
    window.update_state(state)
    assert not window.quality_cross_review.isEnabled()
    assert not window.quality_cross_review.isChecked()
    window.quality_check.setChecked(False)
    window.quality_check.setChecked(True)
    assert not window.quality_cross_review.isEnabled()
    state["cross_review_supported"]=True
    window.update_state(state)
    assert window.quality_cross_review.isEnabled()
    app.processEvents()
    window.close()


def test_tuned_launch_context_reaches_submission_without_adding_batch_regions():
    from types import SimpleNamespace

    from watermark_studio.desktop import apply_processing_defaults
    _app=QApplication.instance() or QApplication([])
    api=FakeApi()
    window=MainWindow(api,{"capacity":20,"roots":["/home/zjh"],"backend":"mock","stage":"ready","job":None})
    args=SimpleNamespace(context=.45,first_pass_mode="complete",input_resolution=1024,
        repair_blending="harmonic",preserve_background=True,quality_check=False,
        quality_auto_retry=False,quality_batch_size=2,input_dir="/home/zjh/input",
        output_dir="/home/zjh/output")
    apply_processing_defaults(window,args)
    assert not any(c[0]=="submit" for c in api.calls)
    window.start_job()
    payload=api.calls[-1][3]
    assert payload["options"]["context"]==.45
    assert payload["options"]["scope"]=="full_image" and payload["options"]["regions"]==[]
    window.close()
