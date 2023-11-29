from __future__ import annotations

from qtpy import QtGui
import pyqtgraph as pg

import numpy as np
from numpy.typing import NDArray
from whitecanvas.protocols import MarkersProtocol, HeteroMarkersProtocol, check_protocol
from whitecanvas.types import Symbol, LineStyle, FacePattern
from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.utils.normalize import as_color_array
from ._qt_utils import (
    array_to_qcolor,
    from_qt_line_style,
    to_qt_line_style,
    from_qt_symbol,
    to_qt_symbol,
    from_qt_brush_style,
    to_qt_brush_style,
)


@check_protocol(MarkersProtocol)
class Markers(pg.ScatterPlotItem, PyQtLayer):
    def __init__(self, xdata, ydata):
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setCosmetic(True)
        super().__init__(
            xdata,
            ydata,
            pen=pen,
            brush=QtGui.QBrush(QtGui.QColor(255, 0, 0)),
            antialias=True,
            useCache=False,  # NOTE: should be True eventually, but pyqtgraph has
            # a bug in caching
        )

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.getData()

    def _plt_set_data(self, xdata, ydata):
        self.setData(xdata, ydata)

    ##### HasSymbol protocol #####
    def _get_brush(self) -> QtGui.QBrush:
        return self.opts["brush"]

    def _plt_get_symbol(self) -> Symbol:
        return from_qt_symbol(self.opts["symbol"])

    def _plt_set_symbol(self, symbol: Symbol):
        self.setSymbol(to_qt_symbol(symbol))

    def _plt_get_symbol_size(self) -> float:
        return self.opts["size"]

    def _plt_set_symbol_size(self, size: float):
        self.setSize(size)

    ##### HasFace protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        rgba = self._get_brush().color().getRgbF()
        return np.array(rgba)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self.setBrush(array_to_qcolor(color))

    def _plt_get_face_pattern(self) -> FacePattern:
        return from_qt_brush_style(self._get_brush().style())

    def _plt_set_face_pattern(self, pattern: FacePattern):
        brush = self._get_brush()
        brush.setStyle(to_qt_brush_style(pattern))
        self.setBrush(brush)

    ##### HasEdges protocol #####
    def _get_pen(self) -> QtGui.QPen:
        return self.opts["pen"]

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        rgba = self._get_pen().color().getRgbF()
        return np.array(rgba)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.setPen(array_to_qcolor(color))

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


@check_protocol(HeteroMarkersProtocol)
class HeteroMarkers(pg.ScatterPlotItem, PyQtLayer):
    def __init__(self, xdata, ydata):
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        ndata = len(xdata)
        super().__init__(
            xdata,
            ydata,
            pen=[QtGui.QPen(pen) for _ in range(ndata)],
            brush=[QtGui.QBrush(QtGui.QColor(0, 0, 0)) for _ in range(ndata)],
            antialias=False,
            useCache=False,  # NOTE: should be True eventually, but pyqtgraph has
            # a bug in caching
        )

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.getData()

    def _plt_set_data(self, xdata, ydata):
        self.setData(xdata, ydata)

    ##### HasSymbol protocol #####
    def _plt_get_symbol(self) -> Symbol:
        return from_qt_symbol(self.opts["symbol"])

    def _plt_set_symbol(self, symbol: Symbol):
        self.setSymbol([to_qt_symbol(symbol)] * len(self.data["x"]))

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return self.data["size"]

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if isinstance(size, (int, float, np.number)):
            size = np.full(len(self.data["x"]), size)
        self.setSize(size)

    ##### HasFace protocol #####
    def _get_brush(self) -> list[QtGui.QBrush]:
        brushes = self.data["brush"]
        return brushes

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return np.array(
            [brush.color().getRgbF() for brush in self._get_brush()], dtype=np.float32
        )

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.data["x"]))
        brushes = self._get_brush()
        for brush, c in zip(brushes, color):
            brush.setColor(array_to_qcolor(c))
        self.setBrush(brushes)

    def _plt_get_face_pattern(self) -> list[FacePattern]:
        return [from_qt_brush_style(brush.style()) for brush in self._get_brush()]

    def _plt_set_face_pattern(self, pattern: FacePattern | list[FacePattern]):
        brushes = self._get_brush()
        if isinstance(pattern, FacePattern):
            ptn = to_qt_brush_style(pattern)
            for brush in brushes:
                brush.setStyle(ptn)
        else:
            for brush, ptn in zip(brushes, pattern):
                brush.setStyle(to_qt_brush_style(ptn))
        self.setBrush(brushes)

    ##### HasEdges protocol #####
    def _get_pen(self) -> list[QtGui.QPen]:
        pens = self.data["pen"]
        return pens

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(
            [pen.color().getRgbF() for pen in self._get_pen()], dtype=np.float32
        )

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.data["x"]))
        pens = self._get_pen()
        for pen, c in zip(pens, color):
            pen.setColor(array_to_qcolor(c))
        self.setPen(pens)

    def _plt_get_edge_width(self) -> float:
        return np.array([pen.widthF() for pen in self._get_pen()], dtype=np.float32)

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if isinstance(width, (int, float, np.number)):
            width = np.full(len(self.data["x"]), width)
        pens = self._get_pen()
        for pen, w in zip(pens, width):
            pen.setWidthF(w)
        self.setPen(pens)

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_qt_line_style(pen.style()) for pen in self._get_pen()]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        pens = self._get_pen()
        if isinstance(style, LineStyle):
            s = to_qt_line_style(style)
            for pen in pens:
                pen.setStyle(s)
        else:
            for pen, s in zip(pens, style):
                pen.setStyle(to_qt_line_style(s))
        self.setPen(pens)
