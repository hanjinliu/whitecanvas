from __future__ import annotations

import weakref
from typing import TypeVar, Generic, TYPE_CHECKING
import numpy as np

from whitecanvas.types import ArrayLike1D, ColorType, FacePattern
from whitecanvas.layers import Layer, Bars, Band
from whitecanvas.layers.group import LabeledBars
from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

_C = TypeVar("_C", bound="CanvasBase")
_L = TypeVar("_L", bound=Layer)


class StackPlotter(Generic[_C, _L]):
    def __init__(self, canvas: _C, over: _L):
        self._canvas_ref = weakref.ref(canvas)
        self._relative_to_ref = weakref.ref(over)

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    def _layer(self) -> _L:
        layer = self._relative_to_ref()
        if layer is None:
            raise ReferenceDeletedError("Layer has been deleted.")
        return layer

    def add(
        self,
        ydata: ArrayLike1D,
        *,
        color: ColorType | None = None,
        pattern: str | FacePattern = FacePattern.SOLID,
        alpha: float = 1.0,
        name: str | None = None,
    ) -> _L:
        canvas = self._canvas()
        layer = self._layer()
        xdata = layer.data.x
        y0 = np.maximum(layer.data.y0, layer.data.y1)
        color = canvas._generate_colors(color)
        if isinstance(layer, Bars):
            new_layer = Bars(
                xdata,
                ydata + y0,
                y0,
                orient=layer.orient,
                color=color,
                alpha=alpha,
                name=name,
                pattern=pattern,
                bar_width=layer.bar_width,
                backend=layer._backend_name,
            )
        elif isinstance(layer, Band):
            new_layer = Band(
                xdata,
                y0,
                ydata + y0,
                orient=layer.orient,
                color=color,
                alpha=alpha,
                name=name,
                pattern=pattern,
                backend=layer._backend_name,
            )
        elif isinstance(layer, LabeledBars):
            bars = layer.bars
            new_layer = Bars(
                xdata,
                ydata + y0,
                y0,
                orient=bars.orient,
                color=color,
                alpha=alpha,
                name=name,
                pattern=pattern,
                bar_width=bars.bar_width,
                backend=layer._backend_name,
            )
        else:
            raise TypeError("Only Bars and Band are supported.")
        canvas.add_layer(new_layer)
        return new_layer
