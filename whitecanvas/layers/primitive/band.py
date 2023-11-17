from __future__ import annotations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import BandProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, FacePattern, ColorType
from whitecanvas.utils.normalize import as_array_1d, norm_color


class Filled(PrimitiveLayer[BandProtocol]):
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


class Band(Filled):
    def __init__(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
        orient: Literal["vertical", "horizontal"] = "vertical",
        *,
        name: str | None = None,
        face_color: ColorType = "blue",
        edge_color: ColorType = "black",
        edge_width: float = 0,
        edge_style: LineStyle | str = LineStyle.SOLID,
        backend: Backend | str | None = None,
    ):
        x = as_array_1d(t)
        y0 = as_array_1d(edge_low)
        y1 = as_array_1d(edge_high)
        if x.size != y0.size or x.size != y1.size:
            raise ValueError(
                "Expected xdata, ydata0, ydata1 to have the same size, " f"got {x.size}, {y0.size}, {y1.size}"
            )
        self._backend = self._create_backend(Backend(backend), x, y0, y1, orient)
        self.name = name if name is not None else "Band"
        self.face_color = face_color
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.edge_style = edge_style
        self._orient = orient

    @property
    def data(self) -> XYData:
        """Current data of the layer."""
        if self._orient == "vertical":
            x, y0, y1 = self._backend._plt_get_vertical_data()
        elif self._orient == "horizontal":
            x, y0, y1 = self._backend._plt_get_horizontal_data()
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")
        return x, y0, y1

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
            raise ValueError("Expected data to have the same size," f"got {t0.size}, {y0.size}, {y1.size}")
        if self._orient == "vertical":
            self._backend._plt_set_vertical_data(t0, y0, y1)
        elif self._orient == "horizontal":
            self._backend._plt_set_horizontal_data(t0, y0, y1)
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")
