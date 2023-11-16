from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import LineProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, Symbol, ColorType, _Void
from whitecanvas.utils.normalize import as_array_1d, norm_color, normalize_xy

if TYPE_CHECKING:
    from whitecanvas.layers.group import LineMarkers

_void = _Void()


class Line(PrimitiveLayer[LineProtocol]):
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

    def with_markers(
        self,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float = 0,
        edge_style: str | LineStyle = LineStyle.SOLID,
    ) -> LineMarkers:
        from whitecanvas.layers.group import LineMarkers
        from whitecanvas.layers.primitive import Markers

        if face_color is _void:
            face_color = self.color
        if edge_color is _void:
            edge_color = self.color

        markers = Markers(
            *self.data, symbol=symbol, size=size, face_color=face_color,
            edge_color=edge_color, edge_width=edge_width, edge_style=edge_style,
            backend=self._backend_name,
        )  # fmt: skip
        return LineMarkers(self, markers, name=self.name)
