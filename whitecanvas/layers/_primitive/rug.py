from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from psygnal import Signal

from whitecanvas.layers._primitive.line import MultiLine, LineLayerEvents
from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, _Void, Orientation, LineStyle
from whitecanvas.utils.normalize import as_array_1d


class RugEvents(LineLayerEvents):
    low = Signal(float)
    high = Signal(float)


class Rug(MultiLine):
    """
    Rug plot (event plot) layer.

      │ ││  │   │
    ──┴─┴┴──┴───┴──>
    """

    events: RugEvents
    _events_class = RugEvents

    def __init__(
        self,
        events: ArrayLike,
        *,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        color: ColorType = "black",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
        antialias: bool = True,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ):
        events, segs, orient = _norm_input(events, low, high, orient)
        super().__init__(
            segs, name=name, color=color, alpha=alpha, width=width, style=style,
            antialias=antialias, backend=backend
        )  # fmt: skip
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
        with self.events.data.blocked():
            super().set_data(segs)
        self._low = val
        self.events.low.emit(val)

    @property
    def high(self) -> float:
        return self._high

    @high.setter
    def high(self, val):
        _, segs, _ = _norm_input(self._events, self.low, val, self.orient)
        with self.events.data.blocked():
            super().set_data(segs)
        self._high = val
        self.events.high.emit(val)

    def set_data(self, events: ArrayLike):
        events, segs, orient = _norm_input(events, self.low, self.high, self.orient)
        with self.events.data.blocked():
            super().set_data(segs)
        self._events = events
        self.events.data.emit(events)

    @property
    def orient(self) -> Orientation:
        """Orientation of the rug plot."""
        return self._orient


def _norm_input(
    events: ArrayLike,
    low: float,
    high: float,
    orient: str | Orientation,
) -> tuple[NDArray[np.number], NDArray[np.number], Orientation]:
    if low > high:
        raise ValueError(f"low must be less than high, got {low} and {high}")
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
