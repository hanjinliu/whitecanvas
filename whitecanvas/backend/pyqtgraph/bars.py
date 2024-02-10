from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent as pgHoverEvent
from qtpy import QtCore, QtGui

from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._qt_utils import (
    array_to_qcolor,
    from_qt_brush_style,
    from_qt_line_style,
    to_qt_brush_style,
    to_qt_line_style,
)
from whitecanvas.protocols import BarProtocol, check_protocol
from whitecanvas.types import Hatch, LineStyle
from whitecanvas.utils.normalize import as_color_array


@check_protocol(BarProtocol)
class Bars(pg.BarGraphItem, PyQtLayer):
    clicked = QtCore.Signal(object)

    def __init__(self, xlow, xhigh, ylow, yhigh):
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        ndata = len(xlow)
        super().__init__(
            x0=xlow, x1=xhigh, y0=ylow, y1=yhigh,
            pens=[QtGui.QPen(pen) for _ in range(ndata)],
            brushes=[QtGui.QBrush(QtGui.QColor(0, 0, 0)) for _ in range(ndata)],
        )  # fmt: skip
        self._hover_texts: list[str] = None
        self._toolTipCleared = True

    ##### XYDataProtocol #####
    def _plt_get_data(self) -> Sequence[NDArray[np.floating]]:
        return self.opts["x0"], self.opts["x1"], self.opts["y0"], self.opts["y1"]

    def _plt_set_data(self, xlow, xhigh, ylow, yhigh):
        self.setOpts(x0=xlow, x1=xhigh, y0=ylow, y1=yhigh)

    ##### HasFace protocol #####

    def _get_brushes(self) -> list[QtGui.QBrush]:
        return self.opts["brushes"]

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        colors = []
        for brush in self._get_brushes():
            colors.append(brush.color().getRgbF())
        return np.array(colors, dtype=np.float32)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.opts["x0"]))
        brushes = self._get_brushes()
        for brush, c in zip(brushes, color):
            brush.setColor(array_to_qcolor(c))
        self.setOpts(brushes=brushes)

    def _plt_get_face_hatch(self) -> list[Hatch]:
        return [from_qt_brush_style(brush.style()) for brush in self._get_brushes()]

    def _plt_set_face_hatch(self, pattern: Hatch | list[Hatch]):
        brushes = self._get_brushes()
        if isinstance(pattern, Hatch):
            ptn = to_qt_brush_style(pattern)
            for brush in brushes:
                brush.setStyle(ptn)
        else:
            for brush, ptn in zip(brushes, pattern):
                brush.setStyle(to_qt_brush_style(ptn))
        self.setOpts(brushes=brushes)

    ##### HasEdges protocol #####

    def _get_pens(self) -> list[QtGui.QPen]:
        return self.opts["pens"]

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        colors = []
        for pen in self._get_pens():
            colors.append(pen.color().getRgbF())
        return np.array(colors, dtype=np.float32)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.opts["x0"]))
        pens = self._get_pens()
        for pen, c in zip(pens, color):
            pen.setColor(array_to_qcolor(c))
        self.setOpts(pens=pens)

    def _plt_get_edge_width(self) -> float:
        return np.array([pen.widthF() for pen in self._get_pens()])

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if np.isscalar(width):
            width = np.full(len(self.opts["x0"]), width)
        pens = self._get_pens()
        for pen, w in zip(pens, width):
            pen.setWidthF(w)
        self.setOpts(pens=pens)

    def _plt_get_edge_style(self) -> LineStyle:
        return [from_qt_line_style(pen.style()) for pen in self._get_pens()]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        pens = self._get_pens()
        if isinstance(style, LineStyle):
            style = to_qt_line_style(style)
            for pen in pens:
                pen.setStyle(style)
        else:
            for pen, s in zip(pens, style):
                pen.setStyle(to_qt_line_style(s))
        self.setOpts(pens=pens)

    def _plt_set_hover_text(self, texts: list[str]):
        self._hover_texts = texts

    def _plt_connect_pick_event(self, callback):
        def cb(ev: QtGui.QMouseEvent):
            idx = self.barUnderCursor(ev.pos())
            if idx >= 0:
                callback(idx)

        self.clicked.connect(cb)

    def barUnderCursor(self, pos: QtCore.QPointF) -> int:
        rect: QtCore.QRectF = self.boundingRect()
        if not rect.contains(pos):
            return -1
        x0, x1, y0, y1 = self._plt_get_data()
        px, py = pos.x(), pos.y()
        indices = np.where((x0 <= px) & (px <= x1) & (y0 <= py) & (py <= y1))[0]
        if indices.size > 0:
            return indices[0]
        return -1

    def hoverEvent(self, ev: pgHoverEvent):
        vb = self.getViewBox()
        if vb is not None and self._hover_texts is not None:
            idx = self.barUnderCursor(ev.pos())
            if idx >= 0:
                self._toolTipCleared = False
                vb.setToolTip(self._hover_texts[idx])
            else:
                if not self._toolTipCleared:
                    vb.setToolTip("")
                    self._toolTipCleared = True

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        super().mousePressEvent(ev)
        self.clicked.emit(ev)
