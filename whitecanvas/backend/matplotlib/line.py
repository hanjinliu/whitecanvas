from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection, Collection
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

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.get_segments()

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        self.set_segments(data)
        self._ndata = len(data)

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> float:
        return self.get_linewidth()[0]

    def _plt_set_edge_width(self, width: float):
        self.set_linewidth(width)

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle(self.get_linestyle()[0])

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        self.set_linestyle(style.value)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_color()[0]

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_color(color)

    def _plt_get_antialias(self) -> bool:
        return self.get_antialiased()[0]

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)
