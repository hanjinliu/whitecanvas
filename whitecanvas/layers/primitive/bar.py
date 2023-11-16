from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import BarProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, FacePattern
from whitecanvas.utils.normalize import as_array_1d, norm_color, normalize_xy


def _to_backend_arrays(xdata, height, width: float):
    xc, y1 = normalize_xy(xdata, height)
    x0 = xc - width / 2
    x1 = xc + width / 2
    y0 = np.zeros_like(y1)
    return x0, x1, y0, y1


class BarBase(PrimitiveLayer[BarProtocol]):
    @property
    def face_color(self):
        """Face color of the bar."""
        return self._backend._plt_get_face_color()

    @face_color.setter
    def face_color(self, color):
        self._backend._plt_set_face_color(norm_color(color))

    @property
    def face_pattern(self) -> FacePattern:
        """Face fill pattern of the bars."""
        return self._backend._plt_get_face_pattern()

    @face_pattern.setter
    def face_pattern(self, style: str | FacePattern):
        self._backend._plt_set_face_pattern(FacePattern(style))

    @property
    def edge_color(self):
        """Edge color of the bar."""
        return self._backend._plt_get_edge_color()

    @edge_color.setter
    def edge_color(self, color):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def edge_width(self) -> float:
        return self._backend._plt_get_edge_width()

    @edge_width.setter
    def edge_width(self, width: float):
        self._backend._plt_set_edge_width(width)

    @property
    def edge_style(self) -> LineStyle:
        return self._backend._plt_get_edge_style()

    @edge_style.setter
    def edge_style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))


class Bar(BarBase):
    def __init__(
        self,
        xdata: ArrayLike,
        height: ArrayLike,
        width: float = 0.8,
        *,
        name: str | None = None,
        face_color="blue",
        edge_color="black",
        edge_width=0,
        edge_style=LineStyle.SOLID,
        backend: Backend | str | None = None,
    ):
        if width <= 0:
            raise ValueError(f"Expected width > 0, got {width}")
        x0, x1, y0, y1 = _to_backend_arrays(xdata, height, width)
        self._backend = self._create_backend(Backend(backend), x0, x1, y0, y1)
        self._width = width
        self.name = name if name is not None else "Line"
        self.face_color = face_color
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.edge_style = edge_style

    @property
    def data(self) -> XYData:
        """Current data of the layer."""
        x0, x1, y0, y1 = self._backend._plt_get_data()
        return XYData((x0 + x1) / 2, y1)

    def set_data(
        self,
        xdata: ArrayLike | None = None,
        height: ArrayLike | None = None,
    ):
        xc, h = self.data
        if xdata is not None:
            xc = as_array_1d(xdata)
        if height is not None:
            h = as_array_1d(height)
        x0, x1, y0, y1 = _to_backend_arrays(xc, h, self.bar_width)
        self._backend._plt_set_data(x0, x1, y0, y1)

    @property
    def bar_width(self) -> float:
        """Width of the bars."""
        return self._width

    @bar_width.setter
    def bar_width(self, w: float):
        if w <= 0:
            raise ValueError(f"Expected width > 0, got {w}")
        x0, x1, y0, y1 = self._backend._plt_get_data()
        dx = (w - self._width) / 2
        x0 = x0 - dx
        x1 = x1 + dx
        self._backend._plt_set_data(x0, x1, y0, y1)
