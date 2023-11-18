from __future__ import annotations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import BandProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.layers._mixin import FaceMixin, EdgeMixin
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, FacePattern, ColorType, _Void
from whitecanvas.utils.normalize import as_array_1d, norm_color

_void = _Void()


class Band(FaceMixin[BandProtocol], EdgeMixin[BandProtocol], PrimitiveLayer[BandProtocol]):
    def __init__(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
        orient: Literal["vertical", "horizontal"] = "vertical",
        *,
        name: str | None = None,
        face_color: ColorType = "blue",
        face_pattern: str | FacePattern = FacePattern.SOLID,
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
        self._orient = orient
        self.setup(
            face_color=face_color,
            face_pattern=face_pattern,
            edge_color=edge_color,
            edge_width=edge_width,
            edge_style=edge_style,
        )  # type: ignore

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

    def setup(
        self,
        *,
        face_color: ColorType | _Void = _void,
        face_pattern: FacePattern | str | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | str | _Void = _void,
    ):
        if face_color is not _void:
            self.face_color = face_color
        if face_pattern is not _void:
            self.face_pattern = face_pattern
        if edge_color is not _void:
            self.edge_color = edge_color
        if edge_width is not _void:
            self.edge_width = edge_width
        if edge_style is not _void:
            self.edge_style = edge_style
        return self
