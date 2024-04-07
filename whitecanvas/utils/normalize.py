from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from cmap import Color
from numpy.typing import ArrayLike, NDArray

from whitecanvas.types import XYData
from whitecanvas.utils import type_check as _tc


def as_array_1d(x: ArrayLike, dtype=None) -> NDArray[np.number]:
    """Normalize the input as a 1D array."""
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got {x.ndim}D array")
    if x.dtype.kind not in "iuf":
        raise ValueError(f"Input {x!r} did not return a numeric array")
    return x


def normalize_xy(*args) -> tuple[NDArray[np.number], NDArray[np.number]]:
    """Normalize the input as two 1D array with the same shape."""
    if len(args) == 1:
        if isinstance(args[0], XYData):
            return args[0].x, args[0].y
        arr = np.asarray(args[0])
        if arr.dtype.kind not in "iuf":
            raise ValueError(f"Input {args[0]!r} did not return a numeric array")
        if arr.ndim == 1:
            ydata = arr
            xdata = np.arange(ydata.size)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            xdata = arr[:, 0]
            ydata = arr[:, 1]
        else:
            raise ValueError(
                f"Expected 1D array or 2D array with shape (N, 2), got {arr.shape}"
            )
    elif len(args) == 2:
        if np.isscalar(args[0]) and np.isscalar(args[1]):
            return np.array(args[0]), np.array(args[1])
        xdata = as_array_1d(args[0])
        if not hasattr(args[1], "__array__") and callable(args[1]):
            ydata = args[1](xdata)
        else:
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
    try:
        c = Color(color)
    except Exception:
        raise ValueError(f"Invalid input for a color: {color!r}") from None
    return np.array(c.rgba, dtype=np.float32)


def hex_color(color) -> str:
    """Normalize a color input to a #RRGGBBAA string."""
    return Color(color).hex


def rgba_str_color(color) -> str:
    """Normalize a color input to a rgba(r, g, b, a) string."""
    return Color(color).rgba_string


def as_any_1d_array(x: Any, size: int, dtype=None) -> np.ndarray:
    if _tc.is_not_array(x):
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
        if color.size == 0 and size == 0:
            return color
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
                f"Color array must have shape (3,), (4,), (N={size}, 3), or (N={size},"
                f" 4) but got\n{color!r}"
            )
    try:
        arr = np.array(color)
    except ValueError as e:
        if str(e).startswith("setting an array element with a sequence"):
            # this happens when ["red", [0.0, 1.0, 0.0, 1.0]] is given
            color_normed = np.stack(
                [Color(c).rgba for c in color],
                axis=0,
                dtype=np.float32,
            )
            return color_normed
        else:
            raise
    return as_color_array(arr, size)


def parse_texts(template: str, ndata: int, extra: Any | None = None) -> dict[str, Any]:
    """Parse a template string and return parameters of hover texts."""
    params = {}
    if extra is None:
        extra = {}
    elif isinstance(extra, Mapping):
        pass
    elif _tc.is_pandas_dataframe(extra):
        return parse_texts(template, ndata, extra=extra.to_dict(orient="list"))
    elif _tc.is_polars_dataframe(extra):
        return parse_texts(template, ndata, extra=extra.to_dict())
    else:
        raise TypeError("extra must be a mapping.")
    for k, v in extra.items():
        if "{" + k not in template:
            continue
        if _tc.is_not_array(v):
            _v = np.full(ndata, v)
        else:
            _v = as_any_1d_array(v, ndata)
        if _v.size != ndata:
            raise ValueError(f"Expected {_v.size} elements, got {ndata} for {k!r}.")
        params[k] = v
    return params
