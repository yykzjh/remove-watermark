"""Display-coordinate region selection on bounded remote thumbnails."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class RegionCanvas(QWidget):
    changed = Signal(list)

    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap()
        self.boxes = []
        self.selectable = False
        self.origin = None
        self.tip = None
        self.setMinimumSize(400, 240)

    def image_rect(self):
        if self.pixmap.isNull():
            return QRectF()
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        w, h = self.pixmap.width() * scale, self.pixmap.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def normalized(self, point):
        rect = self.image_rect()
        return QPointF(
            min(1, max(0, (point.x() - rect.x()) / rect.width())),
            min(1, max(0, (point.y() - rect.y()) / rect.height())),
        )

    def set_image(self, data):
        self.pixmap.loadFromData(data)
        self.clear()

    def clear(self):
        self.boxes = []
        self.origin = self.tip = None
        self.changed.emit([])
        self.update()

    def set_selectable(self, enabled):
        self.selectable = enabled
        self.origin = self.tip = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def mousePressEvent(self, event):
        if (self.selectable and event.button() == Qt.MouseButton.LeftButton
                and len(self.boxes) < 8 and self.image_rect().contains(event.position())):
            self.origin = self.tip = self.normalized(event.position())

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            self.tip = self.normalized(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.origin is None:
            return
        tip = self.normalized(event.position())
        box = (min(self.origin.x(), tip.x()), min(self.origin.y(), tip.y()),
               max(self.origin.x(), tip.x()), max(self.origin.y(), tip.y()))
        if box[2] - box[0] > .002 and box[3] - box[1] > .002:
            self.boxes.append(box)
            self.changed.emit(self.boxes)
        self.origin = self.tip = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#e1e8e4"))
        rect = self.image_rect()
        if self.pixmap.isNull():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开开发机上的图像以显示预览")
            return
        painter.drawPixmap(rect.toRect(), self.pixmap)
        if not self.selectable:
            painter.fillRect(0, 0, self.width(), 28, QColor(245, 245, 245, 220))
            painter.drawText(10, 20, "自动识别水印位置 · 手动框选已禁用")
            return
        boxes = list(self.boxes)
        if self.origin is not None:
            boxes.append((self.origin.x(), self.origin.y(), self.tip.x(), self.tip.y()))
        painter.setPen(QPen(QColor("#ff6633"), 2))
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            selection = QRectF(rect.x() + x0 * rect.width(), rect.y() + y0 * rect.height(),
                               (x1 - x0) * rect.width(), (y1 - y0) * rect.height()).normalized()
            painter.drawRect(selection)
            painter.drawText(selection.topLeft() + QPointF(4, 18), str(index + 1))
