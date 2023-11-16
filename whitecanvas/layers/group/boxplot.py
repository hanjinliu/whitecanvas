from __future__ import annotations

from whitecanvas.types import ColorType, _Void
from whitecanvas.layers.primitive import Markers, Bars, Errorbars
from whitecanvas.layers._base import LayerGroup


_void = _Void()


class BoxPlot(LayerGroup):
    def __init__(self, bars: Bars, err: Errorbars, markers: Markers, name: str | None = None):
        super().__init__([bars, err, markers], name=name)

    @property
    def bars(self) -> Bars:
        return self._children[0]

    @property
    def errorbars(self) -> Errorbars:
        return self._children[1]

    @property
    def markers(self) -> Markers:
        return self._children[2]

    @property
    def edge_color(self) -> ColorType:
        return self.bars.edge_color

    @edge_color.setter
    def edge_color(self, color: ColorType):
        self.bars.edge_color = color
        self.errorbars.color = color
