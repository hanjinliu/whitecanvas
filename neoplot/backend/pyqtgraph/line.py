from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from qtpy.QtCore import Qt
from qtpy import QtGui
import pyqtgraph as pg
from neoplot.protocols import LineProtocol
from neoplot.types import Symbol, LineStyle
from cmap import Color
from neoplot.backend._const import DEFAULT_COLOR


class Line(pg.PlotDataItem):
    
    def __init__(self, xdata, ydata):
        pen = pg.mkPen(DEFAULT_COLOR, width=1, style=Qt.PenStyle.SolidLine)
        brush = pg.mkBrush(DEFAULT_COLOR)
        symbol = None
        super().__init__(
            xdata,
            ydata,
            pen=pen,
            brush=brush,
            symbol=symbol,
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

    ##### HasLine #####
    def _get_pen(self) -> QtGui.QPen:
        return self.opts["pen"]
    
    def _plt_get_line_width(self) -> float:
        return self._get_pen().widthF()
    
    def _plt_set_line_width(self, width: float):
        pen = self._get_pen()
        pen.setWidthF(width)
        self.setPen(pen)

    def _plt_get_line_style(self) -> LineStyle:
        return _LINE_STYLE_INV[self._get_pen().style()]
    
    def _plt_set_line_style(self, style: LineStyle):
        _ls = _LINE_STYLE[style]
        pen = self._get_pen()
        pen.setStyle(_ls)
        self.setPen(pen)
    
    def _plt_get_line_color(self) -> NDArray[np.float32]:
        return np.array(self._get_pen().color().getRgbF())
    
    def _plt_set_line_color(self, color: NDArray[np.float32]):
        pen = self._get_pen()
        pen.setColor(QtGui.QColor(Color(color).hex))
        self.setPen(pen)
    
    def _plt_get_antialias(self) -> bool:
        return self.opts["antialias"]

    def _plt_set_antialias(self, antialias: bool):
        self.opts["antialias"] = antialias
        self.updateItems(True)
        
    ##### HasSymbol protocol #####
    def _get_brush(self) -> QtGui.QBrush:
        return self.opts["symbolBrush"]

    def _plt_get_symbol(self) -> Symbol:
        return self.opts["symbol"]
    
    def _plt_set_symbol(self, symbol: Symbol):
        self.setSymbol(symbol.value)  # TODO
    
    def _plt_get_symbol_size(self) -> float:
        return self.opts["symbolSize"]
    
    def _plt_set_symbol_size(self, size: float):
        self.setSymbolSize(size)
    
    def _plt_get_symbol_face_color(self) -> NDArray[np.float32]:
        rgba = self._get_brush().color().getRgbF()
        return np.array(rgba)
    
    def _plt_set_symbol_face_color(self, color: NDArray[np.float32]):
        brush = self._get_brush()
        brush.setColor(QtGui.QColor(Color(color).hex))
        self.setSymbolBrush(brush)
    
    def _get_symbol_pen(self) -> QtGui.QPen:
        return self.opts["symbolPen"]

    def _plt_get_symbol_edge_color(self) -> NDArray[np.float32]:
        rgba = self._get_symbol_pen().color().getRgbF()
        return np.array(rgba)
    
    def _plt_set_symbol_edge_color(self, color: NDArray[np.float32]):
        pen = self._get_symbol_pen()
        pen.setColor(QtGui.QColor(Color(color).hex))
        self.setSymbolPen(pen)


_LINE_STYLE = {
    LineStyle.SOLID: Qt.PenStyle.SolidLine,
    LineStyle.DASH: Qt.PenStyle.DashLine,
    LineStyle.DOT: Qt.PenStyle.DotLine,
    LineStyle.DASH_DOT: Qt.PenStyle.DashDotLine,
}

_LINE_STYLE_INV = {v: k for k, v in _LINE_STYLE.items()}

assert isinstance(Line, LineProtocol)
