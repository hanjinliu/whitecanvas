from __future__ import annotations

import re
import weakref
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.layers import Band, Bars, Layer
from whitecanvas.layers.group import LabeledBars, StemPlot
from whitecanvas.types import ArrayLike1D, ColorType, Hatch
from whitecanvas.utils.normalize import as_array_1d

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

_C = TypeVar("_C", bound="CanvasBase")
_L = TypeVar("_L", bound=Layer)


class StackOverPlotter(Generic[_C, _L]):
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
        hatch: str | Hatch = Hatch.SOLID,
        alpha: float = 1.0,
        name: str | None = None,
    ) -> _L:
        canvas = self._canvas()
        layer = self._layer()
        color = canvas._generate_colors(color)

        if name is None:
            if re.match(r".+\+\d", layer.name):
                stem, num = layer.name.rsplit("+", 1)
                name = f"{stem}+{int(num) + 1}"
            else:
                name = f"{layer.name}+1"

        # unwrap nested layers in a group
        if isinstance(layer, LabeledBars):
            layer = layer.bars

        if isinstance(layer, Bars):
            new_layer = Bars(
                layer.data.x,
                ydata,
                bottom=layer.top,
                orient=layer.orient,
                color=color,
                alpha=alpha,
                name=name,
                hatch=hatch,
                extent=layer.bar_width,
                backend=layer._backend_name,
            )
        elif isinstance(layer, Band):
            y0 = np.maximum(layer.data.y0, layer.data.y1)
            new_layer = Band(
                layer.data.x,
                y0,
                ydata + y0,
                orient=layer.orient,
                color=color,
                alpha=alpha,
                name=name,
                hatch=hatch,
                backend=layer._backend_name,
            )
        elif isinstance(layer, StemPlot):
            new_layer = StemPlot.from_arrays(
                layer.data.x,
                ydata,
                bottom=layer.top,
                name=name,
                orient=layer.orient,
                backend=layer._backend_name,
            )
        else:
            raise TypeError("Only Bars and Band are supported.")
        canvas.add_layer(new_layer)
        return new_layer

    def add_hist(
        self,
        data: ArrayLike1D,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        hatch: str | Hatch = Hatch.SOLID,
        alpha: float = 1.0,
    ) -> _L:
        """
        Add data as histogram on top of the existing histogram.

        Parameters
        ----------
        data : array-like
            Data to be added as histogram.
        name : str, optional
            Name of the layer.
        color : color-like, optional
            Color of the layer face.
        alpha : float, default 1.0
            Alpha channel of the bars.
        hatch : str or FacePattern, default FacePattern.SOLID
            Hatch Pattern of the bar faces.

        Returns
        -------
        Bars
            The newly added histogram layer.
        """
        data = as_array_1d(data)
        canvas = self._canvas()
        layer = self._layer()
        color = canvas._generate_colors(color)
        if isinstance(layer, LabeledBars):
            layer = layer.bars

        if not isinstance(layer, Bars):
            raise TypeError(f"Can not stack histogram on {layer!r}.")
        if not layer._bar_type.startswith("histogram"):
            raise TypeError(f"{layer!r} is not histogram.")
        centers = layer.data.x
        density = layer._bar_type == "histogram-density"
        dx = layer.bar_width / 2
        bins = np.concatenate([[centers[0] - dx], centers + dx])
        counts, edges = np.histogram(
            data, bins, density=density, range=(bins.min(), bins.max())
        )
        new_layer = Bars(
            centers, counts, bottom=layer.top, extent=dx * 2, name=name,
            color=color, alpha=alpha, orient=layer.orient, hatch=hatch,
            backend=layer._backend_name,
        )  # fmt: skip
        new_layer._bar_type = layer._bar_type
        return canvas.add_layer(new_layer)
