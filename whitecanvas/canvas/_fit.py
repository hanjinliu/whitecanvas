"""X/Y fitting on layers."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from whitecanvas import theme
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.layers import InfCurve, InfLine, Layer
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.protocols.layer_protocols import XYDataProtocol
from whitecanvas.types import ColorType, LineStyle, XYData

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

_C = TypeVar("_C", bound="CanvasBase")
_L = TypeVar("_L", bound=Layer)
_P = TypeVar("_P", bound=XYDataProtocol)


class FitPlotter(Generic[_C, _P]):
    def __init__(self, canvas: _C, layer: DataBoundLayer[_P, XYData]):
        self._canvas_ref = weakref.ref(canvas)
        self._layer_ref = weakref.ref(layer)

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    def _layer(self) -> DataBoundLayer[_P, XYData]:
        layer = self._layer_ref()
        if layer is None:
            raise ReferenceDeletedError("Layer has been deleted.")
        return layer

    def linear(
        self,
        *,
        color: ColorType | None = None,
        width: float | None = None,
        style: str | LineStyle | None = None,
    ) -> InfLine:
        """
        Add a linear regression result of the X/Y data.

        >>> layer = canvas.add_markers(x, y)  # add markers
        >>> canvas.fit(layer).linear()  # linear regressions

        Parameters
        ----------
        color : color-like, optional
            Color of the output line.
        width : float, optional
            Width of the output line.
        style : str or LineStyle, optional
            Line style of the output line.

        Returns
        -------
        InfLine
            _description_
        """
        layer = self._layer()
        canvas = self._canvas()
        data = layer.data

        def _reg(x: NDArray[np.number], y: NDArray[np.number]):
            n = len(x)
            xs1 = x.sum()
            ys1 = y.sum()
            xs2 = (x**2).sum()
            xydot = np.dot(x, y)
            a = (xydot - ys1 * xs1() / n) / (xs2 - xs1() ** 2 / n)
            b = (ys1 - a * xs1()) / n
            return a, b

        a, b = _reg(data.x, data.y)
        color = canvas._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        new = InfLine(
            (0, b), angle=np.rad2deg(np.arctan(a)), color=color, width=width,
            style=style, backend=canvas._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(new)

    def polynomial(
        self,
        degree: int,
        *,
        color: ColorType | None = None,
        width: float | None = None,
        style: str | LineStyle | None = None,
    ) -> InfCurve:
        layer = self._layer()
        canvas = self._canvas()
        data = layer.data
        poly: np.polynomial.Polynomial = np.polynomial.Polynomial.fit(
            data.x, data.y, degree
        )
        color = canvas._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        new = InfCurve(
            poly, color=color, width=width, style=style, backend=canvas._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(new)
