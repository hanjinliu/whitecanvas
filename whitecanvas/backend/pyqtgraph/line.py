from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from qtpy.QtCore import Qt
from qtpy import QtGui
import pyqtgraph as pg
from whitecanvas.protocols import LineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.backend._const import DEFAULT_COLOR
from ._qt_utils import array_to_qcolor, from_qt_line_style, to_qt_line_style


@check_protocol(LineProtocol)
class Line(pg.PlotCurveItem):
    def __init__(self, xdata, ydata):
        pen = pg.mkPen(DEFAULT_COLOR, width=1, style=Qt.PenStyle.SolidLine)
        super().__init__(xdata, ydata, pen=pen, antialias=False)

    ##### BaseProtocol #####
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
