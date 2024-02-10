from __future__ import annotations

from functools import reduce

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent as pgHoverEvent
from qtpy import QtCore, QtGui

from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._qt_utils import (
    array_to_qcolor,
    from_qt_line_style,
    to_qt_line_style,
)
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.utils.type_check import is_real_number


@check_protocol(LineProtocol)
class MonoLine(pg.PlotCurveItem, PyQtLayer):
    def __init__(self, xdata, ydata):
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        super().__init__(xdata, ydata, pen=pen, antialias=False, clickable=True)
        self._hover_texts: list[str] | None = None
        self._toolTipCleared = True

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.getData()

    def _plt_set_data(self, xdata, ydata):
        self.setData(xdata, ydata)

    ##### HasEdges #####
    def _get_pen(self) -> QtGui.QPen:
        return self.opts["pen"]

    def _plt_get_edge_width(self) -> float:
        return self._get_pen().widthF()

    def _plt_set_edge_width(self, width: float):
        pen = self._get_pen()
        pen.setWidthF(width)
        self.setPen(pen)

    def _plt_get_edge_style(self) -> LineStyle:
        return from_qt_line_style(self._get_pen().style())

    def _plt_set_edge_style(self, style: LineStyle):
        pen = self._get_pen()
        pen.setStyle(to_qt_line_style(style))
        self.setPen(pen)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(self._get_pen().color().getRgbF())

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        pen = self._get_pen()
        pen.setColor(array_to_qcolor(color))
        self.setPen(pen)

    def _plt_get_antialias(self) -> bool:
        return self.opts["antialias"]

    def _plt_set_antialias(self, antialias: bool):
        self.opts["antialias"] = antialias
        self.update()

    def closestPoint(self, pos: QtCore.QPointF) -> int:
        # TODO: pick the closest point for now, but ideally, we should pick the
        # closest edge first, and then the closest point on the edge
        x, y = self.getData()
        dist2 = (x - pos.x()) ** 2 + (y - pos.y()) ** 2
        idx = np.argmin(dist2)
        if dist2[idx] < 10:
            return idx
        return idx

    def _plt_connect_pick_event(self, callback):
        def cb(ins, ev: QtGui.QMouseEvent):
            callback([self.closestPoint(ev.pos())])

        self.sigClicked.connect(cb)

    def _plt_set_hover_text(self, text: list[str]):
        self._hover_texts = text

    def hoverEvent(self, ev: pgHoverEvent):
        vb = self.getViewBox()
        if vb is not None and self._hover_texts is not None:
            if self.mouseShape().contains(ev.pos()):
                i = self.closestPoint(ev.pos())
                vb.setToolTip(self._hover_texts[i])
                self._toolTipCleared = False
            else:
                if not self._toolTipCleared:
                    vb.setToolTip("")
                    self._toolTipCleared = True


@check_protocol(MultiLineProtocol)
class MultiLine(pg.ItemGroup, PyQtLayer):
    clicked = QtCore.Signal(list)

    def __init__(self, data: list[NDArray[np.floating]]):
        super().__init__()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setCosmetic(True)
        self._lines: list[pg.PlotCurveItem] = []
        self._qpens: list[QtGui.QPen] = []
        for i, seg in enumerate(data):
            item = pg.PlotCurveItem(
                seg[:, 0], seg[:, 1], pen=pen, antialias=True, clickable=True
            )
            item.sigClicked.connect(self._clicked_cb(i))
            self.addItem(item)
            self._lines.append(item)
            self._qpens.append(pen)
        self._data = data
        self._bounding_rect_cache = None
        self._antialias = True
        self._hover_texts: list[str] = None
        self._toolTipCleared = True

    def _clicked_cb(self, idx: int):
        return lambda _ins, _: self.clicked.emit([idx])

    def boundingRect(self):
        if self._bounding_rect_cache is not None:
            return self._bounding_rect_cache
        if len(self._lines) == 0:
            return QtCore.QRectF()
        rect = reduce(lambda a, b: a | b, (item.boundingRect() for item in self._lines))
        self._bounding_rect_cache = rect
        return rect

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self._data

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        ndata = len(data)
        nitem = len(self._lines)
        scene = self.scene()
        if ndata < nitem:
            for item in self._lines[ndata:]:
                scene.removeItem(item)
                item.sigClicked.disconnect()
            self._lines = self._lines[:ndata]
            self._qpens = self._qpens[:ndata]
        else:
            # pick last pen
            if nitem > 0:
                pen = self._qpens[-1]
            else:
                pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
            for i in range(nitem, ndata):
                item = pg.PlotCurveItem(pen=pen, antialias=True, clickable=True)
                item.sigClicked.connect(self._clicked_cb(i))
                self.addItem(item)
                self._lines.append(item)
                self._qpens.append(pen)
        for item, seg in zip(self._lines, data):
            item.setData(seg[:, 0], seg[:, 1])
        self._data = data
        self._bounding_rect_cache = None

    ##### HasEdges #####
    def _set_pen_to_curves(self):
        for item, pen in zip(self._lines, self._qpens):
            item.setPen(pen)

    def _plt_get_edge_width(self) -> NDArray[np.float32]:
        return np.array([pen.widthF() for pen in self._qpens], dtype=np.float32)

    def _plt_set_edge_width(self, width: float | NDArray[np.float32]):
        if is_real_number(width):
            widths: list[float] = [width] * len(self._qpens)
        else:
            widths = width
        for pen, w in zip(self._qpens, widths):
            pen.setWidthF(w)
        self._set_pen_to_curves()

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_qt_line_style(pen.style()) for pen in self._qpens]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            styles: list[LineStyle] = [style] * len(self._qpens)
        else:
            styles = style
        for pen, s in zip(self._qpens, styles):
            pen.setStyle(to_qt_line_style(s))
        self._set_pen_to_curves()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(
            [pen.color().getRgbF() for pen in self._qpens], dtype=np.float32
        )

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            colors = [color] * len(self._qpens)
        else:
            colors = color
        for pen, c in zip(self._qpens, colors):
            pen.setColor(array_to_qcolor(c))
        self._set_pen_to_curves()

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        for item in self._lines:
            item.opts["antialias"] = antialias
            item.update()
        self._antialias = antialias

    def hoverEvent(self, ev: pgHoverEvent):
        vb = self.getViewBox()
        if vb is not None and self._hover_texts is not None:
            for i, line in enumerate(self._lines):
                if line.mouseShape().contains(ev.pos()):
                    vb.setToolTip(self._hover_texts[i])
                    self._toolTipCleared = False
                    break
            else:
                if not self._toolTipCleared:
                    vb.setToolTip("")
                    self._toolTipCleared = True

    def _plt_set_hover_text(self, texts: list[str]):
        self._hover_texts = texts

    def _plt_connect_pick_event(self, callback):
        self.clicked.connect(callback)
