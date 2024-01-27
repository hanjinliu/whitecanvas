from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import as_color_array


@check_protocol(LineProtocol)
class MonoLine(Line2D, MplLayer):
    def __init__(self, xdata, ydata):
        super().__init__(
            xdata,
            ydata,
            linewidth=1,
            linestyle="-",
            color="blue",
            markersize=0,
        )

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.get_data()

    def _plt_set_data(self, xdata, ydata):
        self.set_data(xdata, ydata)

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> float:
        return self.get_linewidth()

    def _plt_set_edge_width(self, width: float):
        self.set_linewidth(width)

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle(self.get_linestyle())

    def _plt_set_edge_style(self, style: LineStyle):
        self.set_linestyle(style.value)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_color()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_color(color)

    def _plt_get_antialias(self) -> bool:
        return self.get_antialiased()

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)


@check_protocol(MultiLineProtocol)
class MultiLine(LineCollection, MplLayer):
    def __init__(self, data: list[NDArray[np.floating]]):
        # data: list of (N, 2)
        super().__init__(data, linewidths=1)
        self._ndata = len(data)
        self._linestyle = [LineStyle.SOLID] * self._ndata

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.get_segments()

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        self.set_segments(data)
        self._ndata = len(data)

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self.get_linewidth()

    def _plt_set_edge_width(self, width: float):
        if isinstance(width, (int, float, np.number)):
            width = np.full(self._ndata, width)
        else:
            width = np.asarray(width)
        self.set_linewidth(width)

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return self._linestyle

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            _style = [style.value] * self._ndata
            _style_enum = [style] * self._ndata
        else:
            _style = [s.value for s in style]
            _style_enum = style
        self.set_linestyle(_style)
        self._linestyle = _style_enum

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_color()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_color(as_color_array(color, self._ndata))

    def _plt_get_antialias(self) -> bool:
        return self.get_antialiased()[0]

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)
