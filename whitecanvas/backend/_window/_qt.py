from __future__ import annotations

from typing import Callable, cast

from qtpy import QtCore, QtGui
from qtpy import QtWidgets as QtW

from whitecanvas._axis import CategoricalAxis, DimAxis, RangeAxis
from whitecanvas.canvas import CanvasBase, CanvasGrid


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
        return QtCore.QSize(int(w), int(h))

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

    def _update_widget_state(self):
        self._update_qimage()
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        qsize = event.size()

        self._canvas.size = qsize.width(), qsize.height()
        self._update_widget_state()


class QtMainWindow(QtW.QMainWindow):
    _instance = None

    def __init__(self, canvas: CanvasGrid):
        super().__init__()
        self.setWindowTitle("whitecanvas")
        self._widget = QtCanvas(canvas)
        self.setCentralWidget(self._widget)
        sl = QtDimSliders.from_canvas(canvas[0, 0], self)
        dock = QtW.QDockWidget("Dimensions", self)
        dock.setWidget(sl)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        canvas.events.drawn.connect(self._widget._update_widget_state)
        self.__class__._instance = self


class QtDimSliders(QtW.QWidget):
    changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtW.QFormLayout()
        self.setLayout(layout)
        self._layout = layout
        self._widgets: dict[str, QtW.QWidget] = {}

    def set_axes(self, axes: list[DimAxis]):
        for _ in range(len(self._widgets)):
            self._layout.removeRow(0)
        self._widgets.clear()
        for ax in axes:
            if isinstance(ax, RangeAxis):
                widget = QtW.QSlider(QtCore.Qt.Orientation.Horizontal)
                widget.setRange(0, ax.size() - 1)
                widget.valueChanged.connect(self._emit_changed)
            elif isinstance(ax, CategoricalAxis):
                widget = QtW.QComboBox()
                widget.addItems(ax.categories())
                widget.currentIndexChanged.connect(self._emit_changed)
            else:
                raise NotImplementedError
            self._layout.addRow(ax.name, widget)
            self._widgets[ax.name] = widget

    def connect_changed(self, callback: Callable[[dict[str, object]], None]):
        self.changed.connect(callback)

    def _emit_changed(self):
        values = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QtW.QSlider):
                values[name] = cast(QtW.QSlider, widget).value()
            elif isinstance(widget, QtW.QComboBox):
                cbox = cast(QtW.QComboBox, widget)
                idx = cbox.currentIndex()
                values[name] = cbox.itemText(idx)
            else:
                raise NotImplementedError
        self.changed.emit(values)

    @classmethod
    def from_canvas(cls, canvas: CanvasBase, parent=None):
        self = cls(parent=parent)

        @canvas.dims.events.axis_names.connect
        def _update_axes():
            self.set_axes(canvas.dims._axes)

        self.connect_changed(canvas.dims.set_indices)
        _update_axes()
        return self
