from __future__ import annotations

from enum import Enum
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
            raise ValueError(
                "Expected xdata and ydata to have the same size, "
                f"got {xdata.size} and {ydata.size}"
            )
    else:
        raise TypeError(f"Expected 1 or 2 positional arguments, got {len(args)}")
    return xdata, ydata


def arr_color(color) -> np.ndarray:
    """Normalize a color input to a 4-element float array."""
    return np.array(Color(color).rgba, dtype=np.float32)


def hex_color(color) -> str:
    """Normalize a color input to a #RRGGBBAA string."""
    return Color(color).hex


def rgba_str_color(color) -> str:
    """Normalize a color input to a rgba(r, g, b, a) string."""
    return Color(color).rgba_string


def as_any_1d_array(x: float, size: int, dtype=None) -> np.ndarray:
    if np.isscalar(x) or isinstance(x, Enum):
        out = np.full((size,), x, dtype=dtype)
    else:
        out = np.asarray(x, dtype=dtype)
        if out.shape != (size,):
            raise ValueError(f"Expected shape ({size},), got {out.shape}")
    return out


def as_color_array(color, size: int) -> NDArray[np.float32]:
    if isinstance(color, str):  # e.g. color = "black"
        col = arr_color(color)
        return np.repeat(col[np.newaxis, :], size, axis=0)
    if isinstance(color, np.ndarray):
        if color.dtype.kind in "OU":
            if color.shape != (size,):
                raise ValueError(
                    f"Expected color array of shape ({size},), got {color.shape}"
                )
            return np.stack([arr_color(each) for each in color], axis=0)
        elif color.shape in [(3,), (4,)]:
            col = arr_color(color)
            return np.repeat(col[np.newaxis, :], size, axis=0)
        elif color.shape in [(size, 3), (size, 4)]:
            return color
        else:
            raise ValueError(
                "Color array must have shape (3,), (4,), (N, 3), or (N, 4) "
                f"but got {color.shape}"
            )
    arr = np.array(color)
    return as_color_array(arr, size)
