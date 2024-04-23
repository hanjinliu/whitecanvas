from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import art3d
from numpy.typing import NDArray

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.protocols import LineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number


class MonoLine3D(art3d.Line3D, MplLayer):
    def __init__(self, xdata, ydata, zdata):
        super().__init__(
            xdata,
            ydata,
            zdata,
            linewidth=1,
            linestyle="-",
            color="blue",
            markersize=0,
            # picker=True,
            # pickradius=5,
        )

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.get_data_3d()

    def _plt_set_data(self, xdata, ydata, zdata):
        self.set_data_3d(xdata, ydata, zdata)

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
