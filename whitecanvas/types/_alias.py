from typing import Any, Iterable, Literal, Sequence, Union

import numpy as np
from cmap import Color, Colormap
from numpy.typing import NDArray

ColorType = Union[str, Iterable["int | float"], Color]
ColormapType = Union[str, Colormap, "list[ColorType]", Any]
Number = Union[int, float, np.number]
ArrayLike1D = Union[Sequence[Number], NDArray[np.number]]
HistBinType = Union[int, ArrayLike1D, str]
KdeBandWidthType = Union[float, Literal["scott", "silverman"]]


class _Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        cname = self.__class__.__name__.lstrip("_")
        return f"<{cname} object>"


class _Void(_Singleton):
    """Singleton that represents a void value."""
