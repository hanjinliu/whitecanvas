from __future__ import annotations

from whitecanvas.types import ColorType, _Void, Symbol, LineStyle
from whitecanvas.layers.primitive import Line, Band, Bars, Errorbars
from whitecanvas.layers._base import LayerGroup, PrimitiveLayer, XYData


_void = _Void()


class LineBand(LayerGroup):
    def __init__(self, line: Line, band: Band, name: str | None = None):
        super().__init__([line, band], name=name)

    @property
    def line(self) -> Line:
        """The central line layer."""
        return self._children[0]

    @property
    def band(self) -> Band:
        """The band region layer."""
        return self._children[1]
