from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from cmap import Color


def as_array_1d(x: ArrayLike) -> NDArray[np.number]:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got {x.ndim}D array")
    if x.dtype.kind not in "iuf":
        raise ValueError(f"Input {x!r} did not return a numeric array")
    return x


def normalize_xy(*args) -> tuple[NDArray[np.number], NDArray[np.number]]:
    if len(args) == 1:
        ydata = as_array_1d(args[0])
        xdata = np.arange(ydata.size)
    elif len(args) == 2:
        xdata = as_array_1d(args[0])
        ydata = as_array_1d(args[1])
        if xdata.size != ydata.size:
            raise ValueError("Expected xdata and ydata to have the same size, " f"got {xdata.size} and {ydata.size}")
    else:
        raise TypeError(f"Expected 1 or 2 positional arguments, got {len(args)}")
    return xdata, ydata


def norm_color(color) -> np.ndarray:
    """Normalize a color input to a 4-element float array."""
    return np.array(Color(color).rgba, dtype=np.float32)
