from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar
import weakref
from neoplot import protocols
from neoplot.types import LineStyle, Symbol

if TYPE_CHECKING:
    from typing_extensions import Self
    from neoplot.layers._base import Layer

_Protocol = TypeVar("_Protocol", bound=protocols.BaseProtocol)

class ReferenceDeletedError(RuntimeError):
    """Raised when a weakref.ref() has been deleted."""

class Namespace(Generic[_Protocol]):
    _attrs: tuple[str, ...] = ()

    def __init__(self, layer: Layer[_Protocol] | None = None):
        if layer is not None:
            self._layer_ref = weakref.ref(layer)
        else:
            self._layer_ref = lambda: None
        self._instances: dict[int, Layer[_Protocol]] = {}
    
    def __get__(self, layer, owner) -> Self:
        if layer is None:
            return self
        _id = id(layer)
        if (ns := self._instances.get(_id)) is None:
            ns = self._instances[_id] = type(self)(layer)
        return ns
    
    def _get_layer(self) -> Layer[_Protocol]:
        l = self._layer_ref()
        if l is None:
            raise ReferenceDeletedError("Layer has been deleted.")
        return l

    def __repr__(self) -> str:
        cname = type(self).__name__
        try:
            props = [f"layer={self._get_layer()!r}"]
            for k in self._attrs:
                v = getattr(self, k)
                props.append(f"{k}={v!r}")
            return f"{cname}({', '.join(props)})"
        
        except ReferenceDeletedError:
            return f"<{cname} of deleted layer>"
    
    def update(self, d: dict[str, Any] = {}, **kwargs):
        values = dict(d, **kwargs)
        invalid_args = set(values) - set(self._attrs)
        if invalid_args:
            raise TypeError(
                f"Cannot set {invalid_args!r} on {type(self).__name__}"
            )
        for k, v in values.items():
            setattr(self, k, v)

class MarkerNamespace(Namespace[protocols.HasSymbol]):
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
        return self._get_layer()._backend._plt_get_symbol_face_color()
    
    @face_color.setter
    def face_color(self, color):
        self._get_layer()._backend._plt_set_symbol_face_color(color)
    
    @property
    def edge_color(self):
        return self._get_layer()._backend._plt_get_symbol_edge_color()

    @edge_color.setter
    def edge_color(self, color):
        self._get_layer()._backend._plt_set_symbol_edge_color(color)
    
    color = property()
    @color.setter
    def color(self, color):
        self.face_color = color
        self.edge_color = color

class LineNamespace(Namespace[protocols.HasLine]):
    _attrs = ("width", "style", "color")

    @property
    def width(self):
        return self._get_layer()._backend._plt_get_line_width()
    
    @width.setter
    def width(self, width):
        self._get_layer()._backend._plt_set_line_width(width)

    @property
    def style(self) -> LineStyle:
        return self._get_layer()._backend._plt_get_line_style()
    
    @style.setter
    def style(self, style: str | LineStyle):
        self._get_layer()._backend._plt_set_line_style(LineStyle(style))
    
    @property
    def color(self):
        return self._get_layer()._backend._plt_get_line_color()
    
    @color.setter
    def color(self, color):
        self._get_layer()._backend._plt_set_line_color(color)
