from __future__ import annotations

from whitecanvas.layers._primitive import Band, Line, Markers
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.types import ColorType, Hatch, Symbol, XYData, _Void

_void = _Void()


class LineBand(LayerContainer):
    """
    Group of Line, Band and Markers.

    Properties:
        line: The central line layer
        band: The band region around the central line
        markers: The markers at the data points
    """

    def __init__(
        self,
        line: Line,
        band: Band,
        markers: Markers | None = None,
        name: str | None = None,
    ):
        if markers is None:
            markers = Markers([], [], name="markers")
        super().__init__([line, band, markers], name=name)

    @property
    def line(self) -> Line:
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
