from __future__ import annotations
from typing import Iterator, NamedTuple
from whitecanvas.protocols import BaseProtocol

from whitecanvas.types import ColorType, _Void
from whitecanvas.layers.primitive import Markers, Bars, Errorbars
from whitecanvas.layers._base import LayerGroup, PrimitiveLayer


_void = _Void()


class BoxTuple(NamedTuple):
    bars: Bars
    errorbars: Errorbars
    markers: Markers


class BoxPlot(LayerGroup):
    """Box-plot is a list of bars, errorbars and markers."""

    def __init__(
        self,
        components: list[tuple[Bars, Errorbars, Markers]],
        name: str | None = None,
    ):
        super().__init__([bars, err, markers], name=name)

    def _iter_children(self) -> Iterator[PrimitiveLayer[BaseProtocol]]:
        return super()._iter_children()

    @property
    def bars(self) -> Bars:
        return self._children[0]

    @property
    def errorbars(self) -> Errorbars:
        return self._children[1]

    @property
    def markers(self) -> Markers:
        return self._children[2]
