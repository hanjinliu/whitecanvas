from __future__ import annotations

from numpy.typing import ArrayLike

from neoplot.protocols import MarkersProtocol
from neoplot.layers._base import Layer, XYData
from neoplot.backend import Backend
from neoplot.types import Symbol, LineStyle
from neoplot.utils.normalize import normalize_xy, norm_color


class Scatter(Layer[MarkersProtocol]):
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
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.name = name
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

    def set_data(self, xdata: ArrayLike, ydata: ArrayLike):
        x, y = normalize_xy(xdata, ydata)
        self._backend._plt_set_data(x, y)

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
