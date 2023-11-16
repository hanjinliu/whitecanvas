from __future__ import annotations

from numpy.typing import ArrayLike

from whitecanvas.protocols import MarkersProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.backend import Backend
from whitecanvas.types import Symbol, LineStyle
from whitecanvas.utils.normalize import as_array_1d, norm_color, normalize_xy


class Markers(PrimitiveLayer[MarkersProtocol]):
    def __init__(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        symbol=Symbol.CIRCLE,
        size=6,
        face_color="blue",
        edge_color="black",
        edge_width=0,
        edge_style=LineStyle.SOLID,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.name = name if name is not None else "Line"
        self.symbol = symbol
        self.size = size
        self.face_color = face_color
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.edge_style = edge_style

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
    def symbol(self) -> Symbol:
        return self._backend._plt_get_symbol()

    @symbol.setter
    def symbol(self, symbol: str | Symbol):
        self._backend._plt_set_symbol(Symbol(symbol))

    @property
    def size(self) -> float:
        return self._backend._plt_get_symbol_size()

    @size.setter
    def size(self, size: float):
        self._backend._plt_set_symbol_size(size)

    @property
    def face_color(self):
        """Face color of the marker symbol."""
        return self._backend._plt_get_face_color()

    @face_color.setter
    def face_color(self, color):
        self._backend._plt_set_face_color(norm_color(color))

    @property
    def edge_color(self):
        """Edge color of the marker symbol."""
        return self._backend._plt_get_edge_color()

    @edge_color.setter
    def edge_color(self, color):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def edge_width(self) -> float:
        return self._backend._plt_get_edge_width()

    @edge_width.setter
    def edge_width(self, width: float):
        self._backend._plt_set_edge_width(width)

    @property
    def edge_style(self) -> LineStyle:
        return self._backend._plt_get_edge_style()

    @edge_style.setter
    def edge_style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))
