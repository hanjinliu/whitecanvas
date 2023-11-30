from __future__ import annotations

from qtpy import QtWidgets as QtW, QtGui, QtCore
from whitecanvas.canvas import CanvasGrid


class QtCanvas(QtW.QWidget):
    def __init__(self, canvas: CanvasGrid):
        super().__init__()
        self._canvas = canvas
        self._canvas_qimage = QtGui.QImage()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        rect = event.rect()
        painter = QtGui.QPainter(self)

        painter.eraseRect(rect)  # clear the widget canvas
        origin = QtCore.QPoint(rect.left(), rect.top())
        painter.drawImage(origin, self._canvas_qimage)

    def sizeHint(self) -> QtCore.QSize:
        w, h = self._canvas.size
        return int(w), int(h)

    def _update_qimage(self):
        buf = self._canvas.screenshot()
        qimage = QtGui.QImage(
            buf.tobytes(),
            buf.shape[1],
            buf.shape[0],
            QtGui.QImage.Format.Format_RGBA8888,
        )

        qimage.setDevicePixelRatio(self.devicePixelRatioF())
        self._canvas_qimage = qimage

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        qsize = event.size()

        self._canvas.size = qsize.width(), qsize.height()
        self._update_qimage()
        self.update()


class QtMainWindow(QtW.QMainWindow):
    def __init__(self, canvas: CanvasGrid):
        super().__init__()
        self.setWindowTitle("whitecanvas")
        self._widget = QtCanvas(canvas)
        self.setCentralWidget(self._widget)
