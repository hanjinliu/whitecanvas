from typing import Any, Union, Iterable, Sequence
from cmap import Color, Colormap
import numpy as np

ColorType = Union[str, Iterable["int | float"], Color]
ColormapType = Union[str, Colormap, Any]
Number = Union[int, float, np.number]
ArrayLike1D = Sequence[Number]


class _Void:
    """A singleton class that represents a void value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
