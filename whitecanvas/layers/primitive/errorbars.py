from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.layers._base import XYYData
from whitecanvas.layers.primitive.line import MultiLine
from whitecanvas.layers._sizehint import xyy_size_hint
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, ColorType, _Void
from whitecanvas.utils.normalize import as_array_1d, norm_color


_void = _Void()


class Errorbars(MultiLine):
    """Errorbars layer (parallel lines with caps)."""

    def __init__(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
        orient: Literal["vertical", "horizontal"] = "vertical",
        *,
        name: str | None = None,
        color: ColorType = "black",
        alpha: float = 1,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        capsize: float = 0.0,
        backend: Backend | str | None = None,
    ):
        t0 = as_array_1d(t)
        y0 = as_array_1d(edge_low)
        y1 = as_array_1d(edge_high)
        if not (t0.size == y0.size == y1.size):
            raise ValueError(
                "Expected all arrays to have the same size, "
                f"got {t0.size}, {y0.size}, {y1.size}"
            )
        if capsize < 0:
            raise ValueError(f"Capsize must be non-negative, got {capsize!r}")
        if orient == "vertical":
            data = _xyy_to_segments(t0, y0, y1, capsize)
        elif orient == "horizontal":
            data = _yxx_to_segments(t0, y0, y1, capsize)
        else:
            raise ValueError(f"Unknown orientation {orient!r}")
        self._orient = orient
        self._capsize = capsize
        self._data = XYYData(t0, y0, y1)
        super().__init__(
            data, name=name, color=color, width=width, style=style,
            antialias=antialias, backend=backend,
        )  # fmt: skip
        self.update(
            color=color, width=width, style=style, alpha=alpha,
            antialias=antialias, capsize=capsize
        )  # fmt: skip
        self._x_hint, self._y_hint = xyy_size_hint(t0, y0, y1, orient, xpad=capsize / 2)

    @property
    def data(self) -> XYYData:
        """Current data of the layer."""
        return self._data

    def set_data(
        self,
        t: ArrayLike | None = None,
        edge_low: ArrayLike | None = None,
        edge_high: ArrayLike | None = None,
    ):
        x0, y0, y1 = self.data
        if t is not None:
            x0 = as_array_1d(t)
        if edge_low is not None:
            y0 = as_array_1d(edge_low)
        if edge_high is not None:
            y1 = as_array_1d(edge_high)
        if x0.size != y0.size or x0.size != y1.size:
            raise ValueError(
                "Expected data to have the same size, " f"got {x0.size}, {y0.size}"
            )
        if self._orient == "vertical":
            data = _xyy_to_segments(t, y0, y1, self.capsize)
        elif self._orient == "horizontal":
            data = _yxx_to_segments(t, y0, y1, self.capsize)
        else:
            raise ValueError(f"Unknown orientation {self._orient!r}")
        super().set_data(data)
        self._data = XYYData(x0, y0, y1)
        self._x_hint, self._y_hint = xyy_size_hint(x0, y0, y1, self.orient)

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data[0].size

    @property
    def orient(self) -> Literal["vertical", "horizontal"]:
        """Orientation of the error bars."""
        return self._orient

    @property
    def capsize(self) -> float:
        """Size of the cap."""
        return self._capsize

    @capsize.setter
    def capsize(self, capsize: float):
        if capsize < 0:
            raise ValueError(f"Capsize must be non-negative, got {capsize!r}")
        self._capsize = capsize
        self.set_data(*self._data)

    # @property
    # def antialias(self) -> bool:
    #     """Whether to use antialiasing."""
    #     return self._backend._plt_get_antialias()

    # @antialias.setter
    # def antialias(self, antialias: bool):
    #     self._backend._plt_set_antialias(antialias)

    def update(
        self,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | LineStyle | _Void = _void,
        alpha: float | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if width is not _void:
            self.width = width
        if style is not _void:
            self.style = style
        # if antialias is not _void:
        #     self.antialias = antialias
        if alpha is not _void:
            self.alpha = alpha
        if capsize is not _void:
            self.capsize = capsize
        return self

    @property
    def width(self) -> float:
        """Width of error bars."""
        widths = self._backend._plt_get_edge_width()
        if widths.size == 0:
            return 0
        return float(widths[0])

    @width.setter
    def width(self, width: float):
        width = float(width)
        if width < 0:
            raise ValueError(f"Width must be non-negative, got {width!r}")
        self._backend._plt_set_edge_width(width)

    @property
    def color(self) -> NDArray[np.floating]:
        """Color of error bars."""
        colors = self._backend._plt_get_edge_color()
        if colors.size == 0:
            return np.zeros(4)
        return colors[0]

    @color.setter
    def color(self, color: ColorType):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def style(self) -> LineStyle:
        """Style of error bars."""
        styles = self._backend._plt_get_edge_style()
        if len(styles) == 0:
            return LineStyle.SOLID
        return styles[0]

    @style.setter
    def style(self, style: LineStyle | str):
        self._backend._plt_set_edge_style(LineStyle(style))

    @property
    def alpha(self) -> float:
        """Alpha of error bars."""
        return float(self.color[3])

    @alpha.setter
    def alpha(self, alpha: float):
        self.color = np.array([*self.color[:3], alpha])


def _xyy_to_segments(
    x: ArrayLike,
    y0: ArrayLike,
    y1: ArrayLike,
    capsize: float,
):
    starts = np.stack([x, y0], axis=1)
    ends = np.stack([x, y1], axis=1)
    segments = [[start, end] for start, end in zip(starts, ends)]
    if capsize > 0:
        _c = np.array([capsize / 2, 0])
        cap0 = [[start - _c, start + _c] for start in starts]
        cap1 = [[end - _c, end + _c] for end in ends]
    else:
        cap0 = []
        cap1 = []
    return segments + cap0 + cap1


def _yxx_to_segments(
    y: ArrayLike,
    x0: ArrayLike,
    x1: ArrayLike,
    capsize: float,
):
    starts = np.stack([x0, y], axis=1)
    ends = np.stack([x1, y], axis=1)
    segments = [[start, end] for start, end in zip(starts, ends)]
    if capsize > 0:
        _c = np.array([0, capsize / 2])
        cap0 = [[start - _c, start + _c] for start in starts]
        cap1 = [[end - _c, end + _c] for end in ends]
    else:
        cap0 = []
        cap1 = []
    return segments + cap0 + cap1
