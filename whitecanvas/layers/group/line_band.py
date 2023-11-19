from __future__ import annotations

from whitecanvas.types import _Void
from whitecanvas.layers.primitive import Line, Band
from whitecanvas.layers._base import XYData
from whitecanvas.layers.group._collections import ListLayerGroup


class LineBand(ListLayerGroup):
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

    @property
    def data(self) -> XYData:
        """Current data of the central line."""
        return self.line.data
