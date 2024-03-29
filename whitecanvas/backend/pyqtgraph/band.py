from __future__ import annotations

import weakref

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent as pgHoverEvent
from qtpy import QtGui

from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._qt_utils import (
    array_to_qcolor,
    from_qt_brush_style,
    from_qt_line_style,
    to_qt_brush_style,
    to_qt_line_style,
)
from whitecanvas.protocols import BandProtocol, check_protocol
from whitecanvas.types import Hatch, LineStyle, Orientation


@check_protocol(BandProtocol)
class Band(pg.FillBetweenItem, PyQtLayer):
    def __init__(self, t, ydata0, ydata1, orient: Orientation):
        if orient.is_vertical:
            c0 = pg.PlotCurveItem(t, ydata0)
            c1 = pg.PlotCurveItem(t, ydata1)
        else:
            c0 = pg.PlotCurveItem(ydata0, t)
            c1 = pg.PlotCurveItem(ydata1, t)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        super().__init__(c0, c1, pen=pen, brush=QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        self._viewBox = None
        self._hover_texts: list[str] = None
        self._callbacks = []

    def getViewBox(self):
        if self._viewBox is None:
            p = self
            while True:
                try:
                    p = p.parentItem()
                except RuntimeError:
                    return None
                if p is None:
                    vb = self.getViewWidget()
                    if vb is None:
                        return None
                    else:
                        self._viewBox = weakref.ref(vb)
                        break
                if hasattr(p, "implements") and p.implements("ViewBox"):
                    self._viewBox = weakref.ref(p)
                    break
        return self._viewBox()

    ##### XYDataProtocol #####
    def _plt_get_vertical_data(self):
        c0: pg.PlotCurveItem = self.curves[0]
        c1: pg.PlotCurveItem = self.curves[1]
        return c0.xData, c0.yData, c1.yData

    def _plt_set_vertical_data(self, t, ydata0, ydata1):
        c0: pg.PlotCurveItem = self.curves[0]
        c1: pg.PlotCurveItem = self.curves[1]
        c0.setData(t, ydata0)
        c1.setData(t, ydata1)

    def _plt_get_horizontal_data(self):
        c0: pg.PlotCurveItem = self.curves[0]
        c1: pg.PlotCurveItem = self.curves[1]
        return c0.yData, c0.xData, c1.xData

    def _plt_set_horizontal_data(self, y, xdata0, xdata1):
        c0: pg.PlotCurveItem = self.curves[0]
        c1: pg.PlotCurveItem = self.curves[1]
        c0.setData(xdata0, y)
        c1.setData(xdata1, y)

    ##### HasFace protocol #####
    def _get_brush(self) -> QtGui.QBrush:
        return self.brush()

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        rgba = self._get_brush().color().getRgbF()
        return np.array(rgba)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        brush = self._get_brush()
        brush.setColor(array_to_qcolor(color))
        self.setBrush(brush)

    def _plt_get_face_hatch(self) -> Hatch:
        return from_qt_brush_style(self._get_brush().style())

    def _plt_set_face_hatch(self, pattern: Hatch):
        brush = self._get_brush()
        brush.setStyle(to_qt_brush_style(pattern))
        self.setBrush(brush)

    ##### HasEdges protocol #####
    def _get_pen(self) -> QtGui.QPen:
        return self.pen()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        rgba = self._get_pen().color().getRgbF()
        return np.array(rgba)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        pen = self._get_pen()
        pen.setColor(array_to_qcolor(color))
        self.setPen(pen)

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

    def hoverEvent(self, ev: pgHoverEvent):
        vb = self.getViewBox()
        if vb is not None and self._hover_texts is not None:
            if self.contains(ev.pos()):
                self._toolTipCleared = False
                vb.setToolTip(self._hover_texts[0])
            else:
                if not self._toolTipCleared:
                    vb.setToolTip("")
                    self._toolTipCleared = True

    def _plt_set_hover_text(self, text: str):
        self._hover_texts = [text]

    def _plt_connect_pick_event(self, callback):
        def cb(ev: QtGui.QMouseEvent):
            if self.contains(ev.pos()):
                callback()

        self._callbacks.append(cb)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        for cb in self._callbacks:
            cb(ev)
        return super().mousePressEvent(ev)
