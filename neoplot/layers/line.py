from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from neoplot.protocols import LineProtocol
from neoplot.layers._base import Layer, XYData
from neoplot.backend import Backend
from neoplot.types import LineStyle
from neoplot.utils.normalize import as_array_1d, norm_color, normalize_xy


class Line(Layer[LineProtocol]):
    def __init__(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        color=None,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.name = name if name is not None else "Line"
        self.color = color
        self.width = width
        self.style = style
        self.antialias = antialias

    @property
    def data(self) -> XYData:
        """Current data of the layer."""
        return XYData(*self._backend._plt_get_data())

    def set_data(
        self,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
    ):
        x0, y0 = self.data
        if xdata is not None:
            x0 = as_array_1d(xdata)
        if ydata is not None:
            y0 = as_array_1d(ydata)
        if x0.size != y0.size:
            raise ValueError("Expected xdata and ydata to have the same size, " f"got {x0.size} and {y0.size}")
        self._backend._plt_set_data(x0, y0)

    @property
    def width(self):
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @width.setter
    def width(self, width):
        self._backend._plt_set_edge_width(width)

    @property
    def style(self) -> LineStyle:
        """Style of the line."""
        return self._backend._plt_get_edge_style()

    @style.setter
    def style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))

    @property
    def color(self):
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def antialias(self) -> bool:
        """Whether to use antialiasing."""
        return self._backend._plt_get_antialias()

    @antialias.setter
    def antialias(self, antialias: bool):
        self._backend._plt_set_antialias(antialias)
