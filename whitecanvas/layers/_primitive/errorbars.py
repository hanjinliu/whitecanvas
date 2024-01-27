from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.layers._primitive.line import LineLayerEvents, MultiLine
from whitecanvas.protocols.layer_protocols import MultiLineProtocol
from whitecanvas.types import ColorType, LineStyle, Orientation, XYYData, _Void
from whitecanvas.utils.normalize import as_array_1d

_void = _Void()


class ErrorbarsEvents(LineLayerEvents):
    capsize = Signal(float)


class Errorbars(MultiLine, DataBoundLayer[MultiLineProtocol, XYYData]):
    """Errorbars layer (parallel lines with caps)."""

    events: ErrorbarsEvents
    _events_class = ErrorbarsEvents

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

    @classmethod
    def empty(
        cls,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: Backend | str | None = None,
    ) -> Errorbars:
        """Return an Errorbars instance with no component."""
        return Errorbars([], [], [], orient=orient, backend=backend)

    def _get_layer_data(self) -> XYYData:
        """Current data of the layer."""
        return self._data

    def _norm_layer_data(self, data: Any) -> XYYData:
        x0, y0, y1 = self.data
        t, edge_low, edge_high = data
        if t is not None:
            x0 = as_array_1d(t)
        if edge_low is not None:
            y0 = as_array_1d(edge_low)
        if edge_high is not None:
            y1 = as_array_1d(edge_high)
        if x0.size != y0.size or x0.size != y1.size:
            raise ValueError(
                f"Expected data to have the same size, got {x0.size}, {y0.size}"
            )
        return XYYData(x0, y0, y1)

    def _set_layer_data(self, data: XYYData):
        t, y0, y1 = data
        if self._orient.is_vertical:
            segs = _xyy_to_segments(t, y0, y1, self.capsize)
        else:
            segs = _yxx_to_segments(t, y0, y1, self.capsize)
        super()._set_layer_data(segs)
        self._data = data

    def set_data(
        self,
        t: ArrayLike | None = None,
        edge_low: ArrayLike | None = None,
        edge_high: ArrayLike | None = None,
    ):
        self.data = t, edge_low, edge_high

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
        with self.events.data.blocked():
            self.set_data(*self._data)
        self.events.capsize.emit(capsize)

    def update(
        self,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | LineStyle | _Void = _void,
        alpha: float | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ):
        super().update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )
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
    return _to_segments(starts, ends, capsize)


def _yxx_to_segments(
    y: ArrayLike,
    x0: ArrayLike,
    x1: ArrayLike,
    capsize: float,
):
    starts = np.stack([x0, y], axis=1)
    ends = np.stack([x1, y], axis=1)
    return _to_segments(starts, ends, capsize)


def _to_segments(starts, ends, capsize: float):
    segments = np.stack([starts, ends], axis=1)
    if capsize > 0:
        _c = np.array([capsize / 2, 0])
        cap0 = np.stack([starts - _c, starts + _c], axis=1)
        cap1 = np.stack([ends - _c, ends + _c], axis=1)
    else:
        cap0 = []
        cap1 = []
    return list(segments) + list(cap0) + list(cap1)
