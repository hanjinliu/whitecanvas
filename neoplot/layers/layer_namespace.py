from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar
from weakref import WeakValueDictionary
from cmap import Color
import numpy as np
from neoplot import protocols
from neoplot.types import LineStyle, Symbol

if TYPE_CHECKING:
    from typing_extensions import Self
    from neoplot.layers._base import Layer

_Protocol = TypeVar("_Protocol", bound=protocols.BaseProtocol)
_Layer = TypeVar("_Layer", bound="Layer")

_void = object()


def norm_color(color) -> np.ndarray:
    return np.array(Color(color).rgba, dtype=np.float32)


class Namespace(Generic[_Protocol]):
    _attrs: tuple[str, ...] = ()

    def __init__(self, layer: Layer[_Protocol] | None = None):
        if layer is not None:
            self._layer = layer
        else:
            self._layer = None
        self._instances: WeakValueDictionary[int, Self] = WeakValueDictionary()

    def __get__(self, layer, owner) -> Self:
        if layer is None:
            return self
        _id = id(layer)
        if (ns := self._instances.get(_id)) is None:
            ns = self._instances[_id] = type(self)(layer)
        return ns

    def _get_layer(self) -> Layer[_Protocol]:
        return self._layer

    def __repr__(self) -> str:
        cname = type(self).__name__
        props = [f"layer={self._get_layer()!r}"]
        for k in self._attrs:
            v = getattr(self, k)
            props.append(f"{k}={v!r}")
        return f"{cname}({', '.join(props)})"

    def _update(self, d: dict[str, Any] = {}):
        values = dict(d)
        invalid_args = set(values) - set(self._attrs)
        if invalid_args:
            raise TypeError(f"Cannot set {invalid_args!r} on {type(self).__name__}")
        for k, v in values.items():
            setattr(self, k, v)


class MarkerNamespace(Namespace[protocols.MarkersProtocol], Generic[_Layer]):
    _attrs = ("symbol", "size", "face_color", "edge_color")

    @property
    def symbol(self) -> Symbol:
        return self._get_layer()._backend._plt_get_symbol()

    @symbol.setter
    def symbol(self, symbol: str | Symbol):
        self._get_layer()._backend._plt_set_symbol(Symbol(symbol))

    @property
    def size(self) -> float:
        return self._get_layer()._backend._plt_get_symbol_size()

    @size.setter
    def size(self, size: float):
        self._get_layer()._backend._plt_set_symbol_size(size)

    @property
    def face_color(self):
        """Face color of the marker symbol."""
        return self._get_layer()._backend._plt_get_face_color()

    @face_color.setter
    def face_color(self, color):
        self._get_layer()._backend._plt_set_face_color(norm_color(color))

    @property
    def edge_color(self):
        """Edge color of the marker symbol."""
        return self._get_layer()._backend._plt_get_edge_color()

    @edge_color.setter
    def edge_color(self, color):
        self._get_layer()._backend._plt_set_edge_color(norm_color(color))

    def set_color(self, color):
        """Set both face and edge color."""
        self.face_color = color
        self.edge_color = color

    def __call__(
        self,
        symbol: str | Symbol = _void,
        size: float = _void,
        face_color: Any = _void,
        edge_color: Any = _void,
        color: Any = _void,
    ) -> _Layer:
        if symbol is not _void:
            self.symbol = symbol
        if size is not _void:
            self.size = size
        if face_color is not _void:
            self.face_color = face_color
        if edge_color is not _void:
            self.edge_color = edge_color
        if color is not _void:
            self.set_color(color)
        return self._get_layer()


class LineNamespace(Namespace[protocols.LineProtocol], Generic[_Layer]):
    _attrs = ("width", "style", "color")

    @property
    def width(self):
        """Width of the line."""
        return self._get_layer()._backend._plt_get_edge_width()

    @width.setter
    def width(self, width):
        self._get_layer()._backend._plt_set_edge_width(width)

    @property
    def style(self) -> LineStyle:
        """Style of the line."""
        return self._get_layer()._backend._plt_get_edge_style()

    @style.setter
    def style(self, style: str | LineStyle):
        self._get_layer()._backend._plt_set_edge_style(LineStyle(style))

    @property
    def color(self):
        """Color of the line."""
        return self._get_layer()._backend._plt_get_edge_color()

    @color.setter
    def color(self, color):
        self._get_layer()._backend._plt_set_edge_color(norm_color(color))

    def __call__(
        self,
        width: float = _void,
        style: str | LineStyle = _void,
        color: Any = _void,
    ) -> _Layer:
        if width is not _void:
            self.width = width
        if style is not _void:
            self.style = style
        if color is not _void:
            self.color = color
        return self._get_layer()
