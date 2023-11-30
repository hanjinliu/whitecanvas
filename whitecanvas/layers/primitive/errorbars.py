from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.layers._base import XYYData
from whitecanvas.layers.primitive.line import MultiLine
from whitecanvas.layers._sizehint import xyy_size_hint
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, ColorType, _Void, Orientation
from whitecanvas.utils.normalize import as_array_1d


_void = _Void()


class Errorbars(MultiLine):
    """Errorbars layer (parallel lines with caps)."""

    def __init__(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
        orient: str | Orientation = Orientation.VERTICAL,
        *,
        name: str | None = None,
        color: ColorType = "black",
        alpha: float = 1,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
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
        ori = Orientation.parse(orient)
        if ori is Orientation.VERTICAL:
            data = _xyy_to_segments(t0, y0, y1, capsize)
        else:
            data = _yxx_to_segments(t0, y0, y1, capsize)
        self._orient = ori
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
        self._x_hint, self._y_hint = xyy_size_hint(t0, y0, y1, self.orient)

    @classmethod
    def empty(
        cls,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: Backend | str | None = None,
    ) -> Errorbars:
        """Return an Errorbars instance with no component."""
        return Errorbars([], [], [], orient=orient, backend=backend)

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
        if self._orient.is_vertical:
            data = _xyy_to_segments(t, y0, y1, self.capsize)
        else:
            data = _yxx_to_segments(t, y0, y1, self.capsize)
        super().set_data(data)
        self._data = XYYData(x0, y0, y1)
        self._x_hint, self._y_hint = xyy_size_hint(x0, y0, y1, self.orient)

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data[0].size

    @property
    def orient(self) -> Orientation:
        """Orientation of the error bars."""
        return self._orient

    @property
    def capsize(self) -> float:
        """Size of the cap of the line edges."""
        return self._capsize

    @capsize.setter
    def capsize(self, capsize: float):
        if capsize < 0:
            raise ValueError(f"Capsize must be non-negative, got {capsize!r}")
        self._capsize = capsize
        self.set_data(*self._data)

    @property
    def antialias(self) -> bool:
        """Whether to use antialiasing."""
        return self._backend._plt_get_antialias()

    @antialias.setter
    def antialias(self, antialias: bool):
        self._backend._plt_set_antialias(antialias)

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
        if antialias is not _void:
            self.antialias = antialias
        if alpha is not _void:
            self.alpha = alpha
        if capsize is not _void:
            self.capsize = capsize
        return self


def _xyy_to_segments(
    x: ArrayLike,
    y0: ArrayLike,
    y1: ArrayLike,
    capsize: float,
):
    """
    ──┬──  <-- y1
      │
      │
    ──┴──  <-- y0
      ↑
      x
    """
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
