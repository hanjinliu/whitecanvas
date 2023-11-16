from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy import QtGui
import pyqtgraph as pg

import numpy as np
from numpy.typing import NDArray
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Symbol, LineStyle
from ._qt_utils import array_to_qcolor, from_qt_line_style, to_qt_line_style


@check_protocol(MarkersProtocol)
class Markers(pg.ScatterPlotItem):
    def __init__(self, xdata, ydata):
        pen = pg.mkPen("white", width=1, style=Qt.PenStyle.SolidLine)
        brush = pg.mkBrush("white")
        super().__init__(
            xdata,
            ydata,
            pen=pen,
            brush=brush,
            symbol="o",
            size=6,
            antialias=False,
        )

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

    ##### HasSymbol protocol #####
    def _get_brush(self) -> QtGui.QBrush:
        return self.opts["brush"]

    def _plt_get_symbol(self) -> Symbol:
        return self.opts["symbol"]

    def _plt_set_symbol(self, symbol: Symbol):
        self.setSymbol(symbol.value)  # TODO

    def _plt_get_symbol_size(self) -> float:
        return self.opts["size"]

    def _plt_set_symbol_size(self, size: float):
        self.setSize(size)

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        rgba = self._get_brush().color().getRgbF()
        return np.array(rgba)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        brush = self._get_brush()
        brush.setColor(array_to_qcolor(color))
        self.setBrush(brush)

    def _get_pen(self) -> QtGui.QPen:
        return self.opts["pen"]

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
