from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import ErrorbarProtocol
from whitecanvas.layers._base import PrimitiveLayer
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, ColorType, _Void
from whitecanvas.utils.normalize import as_array_1d, norm_color


_void = _Void()


class Errorbars(PrimitiveLayer[ErrorbarProtocol]):
    def __init__(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
        orient: Literal["vertical", "horizontal"] = "vertical",
        *,
        name: str | None = None,
        color=None,
        line_width: float = 1,
        line_style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        capsize: float = 0.0,
        backend: Backend | str | None = None,
    ):
        t0 = as_array_1d(t)
        y0 = as_array_1d(edge_low)
        y1 = as_array_1d(edge_high)
        if not (t0.size == y0.size == y1.size):
            raise ValueError("Expected all arrays to have the same size, " f"got {t0.size}, {y0.size}, {y1.size}")
        self._backend = self._create_backend(Backend(backend), t0, y0, y1, orient)
        self._orient = orient
        self.name = name if name is not None else "Errorbar"
        self.color = color
        self.line_width = line_width
        self.line_style = line_style
        # self.antialias = antialias  TODO
        self.capsize = capsize

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Current data of the layer."""
        if self._orient == "vertical":
            return self._backend._plt_get_vertical_data()
        else:
            return self._backend._plt_get_horizontal_data()

    def set_data(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
    ):
        x0, y0, y1 = self.data
        if t is not None:
            x0 = as_array_1d(t)
        if edge_low is not None:
            y0 = as_array_1d(edge_low)
        if edge_high is not None:
            y1 = as_array_1d(edge_high)
        if x0.size != y0.size or x0.size != y1.size:
            raise ValueError("Expected data to have the same size, " f"got {x0.size}, {y0.size}")
        if self._orient == "vertical":
            self._backend._plt_set_vertical_data(x0, y0, y1)
        else:
            self._backend._plt_set_horizontal_data(x0, y0, y1)

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data[0].size

    @property
    def line_width(self):
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @line_width.setter
    def line_width(self, width):
        self._backend._plt_set_edge_width(width)

    @property
    def line_style(self) -> LineStyle:
        """Style of the line."""
        return self._backend._plt_get_edge_style()

    @line_style.setter
    def line_style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))

    @property
    def color(self):
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def capsize(self) -> float:
        """Size of the cap."""
        return self._backend._plt_get_capsize()

    @capsize.setter
    def capsize(self, capsize: float):
        self._backend._plt_set_capsize(capsize, self._orient)

    # @property
    # def antialias(self) -> bool:
    #     """Whether to use antialiasing."""
    #     return self._backend._plt_get_antialias()

    # @antialias.setter
    # def antialias(self, antialias: bool):
    #     self._backend._plt_set_antialias(antialias)

    def setup(
        self,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if line_width is not _void:
            self.line_width = line_width
        if line_style is not _void:
            self.line_style = line_style
        if antialias is not _void:
            self.antialias = antialias
        if capsize is not _void:
            self.capsize = capsize
        return self
