from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.layers.primitive.line import MultiLine
from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, _Void, Orientation
from whitecanvas.utils.normalize import as_array_1d


class Rug(MultiLine):
    """
    Rug plot (event plot) layer.

      │ ││  │   │
    ──┴─┴┴──┴───┴──>
    """

    def __init__(
        self,
        events: ArrayLike,
        *,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        color: ColorType = "black",
        alpha: float = 1.0,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ):
        events, segs, orient = _norm_input(events, low, high, orient)
        super().__init__(segs, name=name, color=color, alpha=alpha, backend=backend)
        self._orient = orient
        self._events = events
        self._low = low
        self._high = high

    @property
    def data(self) -> NDArray[np.number]:
        return self._events

    @property
    def low(self) -> float:
        return self._low

    @low.setter
    def low(self, val):
        _, segs, _ = _norm_input(self._events, val, self.high, self.orient)
        super().set_data(segs)
        self._low = val

    @property
    def high(self) -> float:
        return self._high

    @high.setter
    def high(self, val):
        _, segs, _ = _norm_input(self._events, self.low, val, self.orient)
        super().set_data(segs)
        self._high = val

    def set_data(self, events: ArrayLike):
        events, segs, orient = _norm_input(events, self.low, self.high, self.orient)
        super().set_data(segs)
        self._events = events

    @property
    def orient(self) -> Orientation:
        return self._orient


def _norm_input(
    events: ArrayLike,
    low: float,
    high: float,
    orient: str | Orientation,
) -> tuple[NDArray[np.number], NDArray[np.number], Orientation]:
    _t = as_array_1d(events)
    y0 = np.full_like(_t, low)
    y1 = np.full_like(_t, high)
    ori = Orientation.parse(orient)
    if ori.is_vertical:
        start = np.stack([_t, y0], axis=1)
        stop = np.stack([_t, y1], axis=1)
        segs = np.stack([start, stop], axis=0)
    else:
        start = np.stack([y0, _t], axis=1)
        stop = np.stack([y1, _t], axis=1)
        segs = np.stack([start, stop], axis=0)
    return events, np.moveaxis(segs, 1, 0), orient
