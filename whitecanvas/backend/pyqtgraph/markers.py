from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy import QtGui

from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._qt_utils import (
    array_to_qcolor,
    from_qt_brush_style,
    from_qt_line_style,
    from_qt_symbol,
    to_qt_brush_style,
    to_qt_line_style,
    to_qt_symbol,
)
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Hatch, LineStyle, Symbol
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number


@check_protocol(MarkersProtocol)
class Markers(pg.ScatterPlotItem, PyQtLayer):
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
        self.opts["tip"] = "{data}".format

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.getData()

    def _plt_set_data(self, xdata: np.ndarray, ydata: np.ndarray):
        pens = self._get_pen()
        brushes = self._get_brush()
        ndata = xdata.size
        if ndata <= len(brushes):
            self.setData(
                xdata,
                ydata,
                pen=pens[:ndata],
                brush=brushes[:ndata],
                size=self.data["size"][:ndata],
            )
        else:
            new_pen = [QtGui.QPen(pens[-1])] * (ndata - len(pens))
            new_brush = [QtGui.QBrush(brushes[-1])] * (ndata - len(brushes))
            new_size = np.full(ndata - len(self.data["size"]), self.data["size"][-1])
            self.setData(
                xdata,
                ydata,
                pen=np.concatenate([pens, new_pen]),
                brush=np.concatenate([brushes, new_brush]),
                size=np.concatenate([self.data["size"], new_size]),
            )

    ##### HasSymbol protocol #####
    def _plt_get_symbol(self) -> Symbol:
        return from_qt_symbol(self.opts["symbol"])

    def _plt_set_symbol(self, symbol: Symbol):
        self.setSymbol(to_qt_symbol(symbol))

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return self.data["size"]

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if is_real_number(size):
            size = np.full(len(self.data["x"]), size)
        self.setSize(size)

    ##### HasFace protocol #####
    def _get_brush(self) -> list[QtGui.QBrush]:
        brushes = self.data["brush"]
        return brushes

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        brushes = self._get_brush()
        if len(brushes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(
            [brush.color().getRgbF() for brush in brushes], dtype=np.float32
        )

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.data["x"]))
        brushes = self._get_brush()
        for brush, c in zip(brushes, color):
            brush.setColor(array_to_qcolor(c))
        self.setBrush(brushes)

    def _plt_get_face_hatch(self) -> list[Hatch]:
        return [from_qt_brush_style(brush.style()) for brush in self._get_brush()]

    def _plt_set_face_hatch(self, pattern: Hatch | list[Hatch]):
        brushes = self._get_brush()
        if isinstance(pattern, Hatch):
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
        pens = self._get_pen()
        if len(pens) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array([pen.color().getRgbF() for pen in pens], dtype=np.float32)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.data["x"]))
        pens = self._get_pen()
        for pen, c in zip(pens, color):
            pen.setColor(array_to_qcolor(c))
        self.setPen(pens)

    def _plt_get_edge_width(self) -> float:
        return np.array([pen.widthF() for pen in self._get_pen()], dtype=np.float32)

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if is_real_number(width):
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

    def _plt_connect_pick_event(self, callback):
        def cb(ins, points, ev):
            callback([p.index() for p in points])

        self.sigClicked.connect(cb)

    def _plt_set_hover_text(self, text: list[str]):
        self.data["data"] = text
        self.opts["hoverable"] = True
