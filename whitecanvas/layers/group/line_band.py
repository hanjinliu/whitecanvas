from __future__ import annotations

from typing import Generic, TypeVar

from whitecanvas.layers._primitive import Band, Markers
from whitecanvas.layers._primitive.line import _SingleLine
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.types import ColorType, Hatch, Symbol, XYData, _Void

_void = _Void()
_L = TypeVar("_L", bound=_SingleLine)


class LineBand(LayerContainer, Generic[_L]):
    """
    Group of Line, Band and Markers.

    Properties:
        line: The central line layer
        band: The band region around the central line
        markers: The markers at the data points
    """

    def __init__(
        self,
        line: _L,
        band: Band,
        markers: Markers | None = None,
        name: str | None = None,
    ):
        if markers is None:
            markers = Markers([], [], name="markers")
        super().__init__([line, band, markers], name=name)

    @property
    def line(self) -> _L:
        """The central line layer."""
        return self._children[0]

    @property
    def band(self) -> Band:
        """The band region layer."""
        return self._children[1]

    @property
    def markers(self) -> Markers:
        """The markers layer."""
        return self._children[2]

    @property
    def data(self) -> XYData:
        """Current data of the central line."""
        return self.line.data

    def with_markers(
        self,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        color: ColorType | _Void = _void,
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> LineBand:
        """Add markers at the data points."""
        if color is _void:
            color = self.line.color
        self.markers.update(
            symbol=symbol, size=size, color=color, alpha=alpha, hatch=hatch
        )
        return self
