from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from neoplot.protocols import LineProtocol
from neoplot.types import Symbol, LineStyle
from cmap import Color

from neoplot.backend._const import DEFAULT_COLOR

class Line(Line2D):
    
    def __init__(self, xdata, ydata):
        super().__init__(
            xdata,
            ydata,
            linewidth=1,
            linestyle="-",
            color=DEFAULT_COLOR,
            markersize=7,
        )
    
    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.get_visible()
    
    def _plt_set_visible(self, visible: bool):
        self.set_visible(visible)
    
    def _plt_get_zorder(self) -> int:
        return self.get_zorder()
    
    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

    ##### XYDataProtocol #####
    def _plt_get_data(self) -> np.ndarray:
        return self.get_data()
    
    def _plt_set_data(self, data: np.ndarray):
        xdata = data[:, 0]
        ydata = data[:, 1]
        self.set_data(xdata, ydata)

    ##### HasLine #####
    def _plt_get_line_width(self) -> float:
        return self.get_linewidth()
    
    def _plt_set_line_width(self, width: float):
        self.set_linewidth(width)

    def _plt_get_line_style(self) -> LineStyle:
        return LineStyle(self.get_linestyle())
    
    def _plt_set_line_style(self, style: LineStyle):
        self.set_linestyle(style.value)
    
    def _plt_get_line_color(self) -> NDArray[np.float32]:
        return self.get_color()
    
    def _plt_set_line_color(self, color: NDArray[np.float32]):
        self.set_color(color)
    
    def _plt_get_antialias(self) -> bool:
        return self.get_antialiased()

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)
        
    ##### HasSymbol protocol #####
    def _plt_get_symbol(self) -> Symbol:
        return Symbol(self.get_marker())
    
    def _plt_set_symbol(self, symbol: Symbol):
        self.set_marker(symbol.value)
    
    def _plt_get_symbol_size(self) -> float:
        return self.get_markersize()
    
    def _plt_set_symbol_size(self, size: float):
        self.set_markersize(size)
    
    def _plt_get_symbol_face_color(self) -> NDArray[np.float32]:
        return self.get_markerfacecolor()
    
    def _plt_set_symbol_face_color(self, color: NDArray[np.float32]):
        self.set_markerfacecolor(color)
    
    def _plt_get_symbol_edge_color(self) -> NDArray[np.float32]:
        return self.get_markeredgecolor()
    
    def _plt_set_symbol_edge_color(self, color: NDArray[np.float32]):
        self.set_markeredgecolor(color)

assert isinstance(Line, LineProtocol)
