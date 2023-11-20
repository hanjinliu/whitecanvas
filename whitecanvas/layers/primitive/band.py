from __future__ import annotations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import BandProtocol
from whitecanvas.layers._base import XYYData
from whitecanvas.layers._mixin import FaceEdgeMixin
from whitecanvas.backend import Backend
from whitecanvas.types import FacePattern, ColorType, _Void
from whitecanvas.utils.normalize import as_array_1d

_void = _Void()


class Band(FaceEdgeMixin[BandProtocol]):
    def __init__(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
        orient: Literal["vertical", "horizontal"] = "vertical",
        *,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: Backend | str | None = None,
    ):
        x = as_array_1d(t)
        y0 = as_array_1d(edge_low)
        y1 = as_array_1d(edge_high)
        if x.size != y0.size or x.size != y1.size:
            raise ValueError(
                "Expected xdata, ydata0, ydata1 to have the same size, "
                f"got {x.size}, {y0.size}, {y1.size}"
            )
        self._backend = self._create_backend(Backend(backend), x, y0, y1, orient)
        self.name = name if name is not None else "Band"
        self._orient = orient
        self.face.setup(color=color, alpha=alpha, pattern=pattern)

    @property
    def data(self) -> XYYData:
        """Current data of the layer."""
        if self._orient == "vertical":
            x, y0, y1 = self._backend._plt_get_vertical_data()
        elif self._orient == "horizontal":
            x, y0, y1 = self._backend._plt_get_horizontal_data()
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")
        return XYYData(x, y0, y1)

    @property
    def orient(self) -> Literal["vertical", "horizontal"]:
        """Orientation of the band."""
        return self._orient

    def set_data(
        self,
        t: ArrayLike | None = None,
        edge_low: ArrayLike | None = None,
        edge_high: ArrayLike | None = None,
    ):
        t0, y0, y1 = self.data
        if t is not None:
            t0 = t
        if edge_low is not None:
            y0 = as_array_1d(edge_low)
        if edge_high is not None:
            y1 = as_array_1d(edge_high)
        if t0.size != y0.size or t0.size != y1.size:
            raise ValueError(
                "Expected data to have the same size,"
                f"got {t0.size}, {y0.size}, {y1.size}"
            )
        if self._orient == "vertical":
            self._backend._plt_set_vertical_data(t0, y0, y1)
        elif self._orient == "horizontal":
            self._backend._plt_set_horizontal_data(t0, y0, y1)
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")
