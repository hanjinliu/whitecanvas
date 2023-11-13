from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy import QtGui
import pyqtgraph as pg

import numpy as np
from numpy.typing import NDArray
from neoplot.protocols import ScatterProtocol
from neoplot.types import Symbol
from cmap import Color


class Scatter(pg.ScatterPlotItem):
    def __init__(self, xdata, ydata):
        pen = pg.mkPen("white", width=1, style=Qt.PenStyle.SolidLine)
        brush = pg.mkBrush("white")
        super().__init__(
            xdata,
            ydata,
            pen=pen,
            brush=brush,
            symbol=None,
            symbolSize=7,
            antialias=False,
            symbolPen=pen,
            symbolBrush=brush,
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
    def _plt_get_data(self) -> np.ndarray:
        return self.getData()
    
    def _plt_set_data(self, data: np.ndarray):
        xdata = data[:, 0]
        ydata = data[:, 1]
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
    
    def _plt_get_symbol_face_color(self) -> NDArray[np.float32]:
        rgba = self._get_brush().color().getRgbF()
        return np.array(rgba)
    
    def _plt_set_symbol_face_color(self, color: NDArray[np.float32]):
        brush = self._get_brush()
        brush.setColor(QtGui.QColor(Color(color).hex))
        self.setBrush(brush)
    
    def _get_symbol_pen(self) -> QtGui.QPen:
        return self.opts["pen"]

    def _plt_get_symbol_edge_color(self) -> NDArray[np.float32]:
        rgba = self._get_symbol_pen().color().getRgbF()
        return np.array(rgba)
    
    def _plt_set_symbol_edge_color(self, color: NDArray[np.float32]):
        pen = self._get_symbol_pen()
        pen.setColor(QtGui.QColor(Color(color).hex))
        self.setPen(pen)

assert isinstance(Scatter, ScatterProtocol)
