from __future__ import annotations
from typing import Literal

from whitecanvas.types import ColorType, _Void
from whitecanvas.layers.primitive import Band
from whitecanvas.layers.group._base import ListLayerGroup


class ViolinPlot(ListLayerGroup):
    def __init__(self, bands: list[Band], name: str | None = None):
        super().__init__(bands, name=name)
        self._shape: Literal["both", "left", "right"] = "both"

    def nth(self, n: int) -> Band:
        return self._children[n]
