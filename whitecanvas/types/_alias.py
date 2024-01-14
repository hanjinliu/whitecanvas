from typing import Any, Iterable, Sequence, Union

import numpy as np
from cmap import Color, Colormap
from numpy.typing import NDArray

ColorType = Union[str, Iterable["int | float"], Color]
ColormapType = Union[str, Colormap, Any]
Number = Union[int, float, np.number]
ArrayLike1D = Union[Sequence[Number], NDArray[np.number]]


class _Void:
    """A singleton class that represents a void value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
