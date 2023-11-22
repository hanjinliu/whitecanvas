from __future__ import annotations
from matplotlib.collections import LineCollection
import numpy as np
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle


@check_protocol(LineProtocol)
class MonoLine(Line2D):
    def __init__(self, xdata, ydata):
        super().__init__(
            xdata,
            ydata,
            linewidth=1,
            linestyle="-",
            color="blue",
            markersize=0,
        )

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.get_visible()

    def _plt_set_visible(self, visible: bool):
        self.set_visible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

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
class MultiLine(LineCollection):
    def __init__(self, data: list[NDArray[np.floating]]):
        # data: list of (N, 2)
        super().__init__(data, linewidths=1)

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.get_visible()

    def _plt_set_visible(self, visible: bool):
        self.set_visible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.get_segments()

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        self.set_segments(data)

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self.get_linewidth()

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        self.set_linewidth(width)

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [LineStyle(ls) for ls in self.get_linestyle()]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            self.set_linestyle(style.value)
        else:
            self.set_linestyle([s.value for s in style])

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_color()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_color(color)

    def _plt_get_antialias(self) -> bool:
        return self.get_antialiased()[0]

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)
