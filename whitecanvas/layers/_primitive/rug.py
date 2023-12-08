from __future__ import annotations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from psygnal import Signal

from whitecanvas.layers._primitive.line import (
    MultiLine,
    LineLayerEvents,
    MultiLineProtocol,
)
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, _Void, Orientation, LineStyle
from whitecanvas.utils.normalize import as_array_1d


class RugEvents(LineLayerEvents):
    low = Signal(float)
    high = Signal(float)


class Rug(MultiLine, DataBoundLayer[MultiLineProtocol, NDArray[np.number]]):
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

    def _get_layer_data(self) -> NDArray[np.number]:
        return self._events

    def _norm_layer_data(self, data: Any) -> NDArray[np.number]:
        return as_array_1d(data)

    def _set_layer_data(self, data: NDArray[np.number]):
        _, segs, _ = _norm_input(data, self.low, self.high, self.orient)
        return super()._set_layer_data(segs)

    @property
    def low(self) -> float:
        return self._low

    @low.setter
    def low(self, val):
        _, segs, _ = _norm_input(self._events, val, self.high, self.orient)
        super()._set_layer_data(segs)
        self._low = val
        self.events.low.emit(val)

    @property
    def high(self) -> float:
        return self._high

    @high.setter
    def high(self, val):
        _, segs, _ = _norm_input(self._events, self.low, val, self.orient)
        super()._set_layer_data(segs)
        self._high = val
        self.events.high.emit(val)

    def set_data(self, events: ArrayLike):
        self.data = events

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
