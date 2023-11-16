from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from qtpy import QtGui
import pyqtgraph as pg
from whitecanvas.protocols import FillBetweenProtocol, check_protocol
from whitecanvas.types import LineStyle, FacePattern
from ._qt_utils import (
    array_to_qcolor,
    from_qt_line_style,
    to_qt_line_style,
    from_qt_brush_style,
    to_qt_brush_style,
)


@check_protocol(FillBetweenProtocol)
class FillBetween(pg.FillBetweenItem):
    def __init__(self, xdata, ydata0, ydata1):
        c0 = pg.PlotCurveItem(xdata, ydata0)
        c1 = pg.PlotCurveItem(xdata, ydata1)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        super().__init__(c0, c1, pen=pen, brush=QtGui.QBrush(QtGui.QColor(0, 0, 0)))

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        self.setVisible(visible)

    def _plt_get_zorder(self) -> int:
        return self.zValue()

    def _plt_set_zorder(self, zorder: int):
        self.setZValue(zorder)

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.getData()

    def _plt_set_data(self, xdata, ydata):
        self.setData(xdata, ydata)

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

    def _plt_get_face_pattern(self) -> FacePattern:
        return from_qt_brush_style(self._get_brush().style())

    def _plt_set_face_pattern(self, pattern: FacePattern):
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
