from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from whitecanvas.types import Orientation


def xy_size_hint(
    x: NDArray[np.number],
    y: NDArray[np.number],
):
    if x.size == 0:
        return None, None
    x_minmax = x.min(), x.max()
    y_minmax = y.min(), y.max()
    _x_hint, _y_hint = x_minmax, y_minmax
    return _x_hint, _y_hint


def xyy_size_hint(
    x: NDArray[np.number],
    y0: NDArray[np.number],
    y1: NDArray[np.number],
    orient: Orientation,
    xpad: float = 0,
):
    if x.size == 0:
        return None, None
    x_minmax = x.min() - xpad, x.max() + xpad
    y_minmax = min(y0.min(), y1.min()), max(y0.max(), y1.max())
    if orient is Orientation.VERTICAL:
        _x_hint, _y_hint = x_minmax, y_minmax
    else:
        _x_hint, _y_hint = y_minmax, x_minmax
    return _x_hint, _y_hint
