from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import BarProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, FacePattern, ColorType, _Void
from whitecanvas.utils.normalize import as_array_1d, norm_color, normalize_xy

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg


def _to_backend_arrays(xdata, height, width: float):
    xc, y1 = normalize_xy(xdata, height)
    x0 = xc - width / 2
    x1 = xc + width / 2
    y0 = np.zeros_like(y1)
    return x0, x1, y0, y1


_void = _Void()


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


class Bars(BarBase):
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
        self.name = name if name is not None else "Bars"
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

    def setup(
        self,
        *,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | str | _Void = _void,
    ):
        if face_color is not _void:
            self.face_color = face_color
        if edge_color is not _void:
            self.edge_color = edge_color
        if edge_width is not _void:
            self.edge_width = edge_width
        if edge_style is not _void:
            self.edge_style = edge_style
        return self

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.LineErrorbars:
        from whitecanvas.layers.group import LineErrorbars
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge_color
        if line_width is _void:
            line_width = self.edge_width
        if line_style is _void:
            line_style = self.edge_style
        # if antialias is _void:
        #     antialias = self.antialias
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            line_width=line_width, line_style=line_style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars([], [], [], backend=self._backend_name)
        return LineErrorbars(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
    ) -> _lg.LineErrorbars:
        from whitecanvas.layers.group import LineErrorbars
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge_color
        if line_width is _void:
            line_width = self.edge_width
        if line_style is _void:
            line_style = self.edge_style
        # if antialias is _void:
        #     antialias = self.antialias
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            line_width=line_width, line_style=line_style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        xerr = Errorbars([], [], [], backend=self._backend_name)
        return LineErrorbars(self, xerr, yerr, name=self.name)
