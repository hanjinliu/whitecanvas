from typing import Any, Union, Iterable
from cmap import Color, Colormap

ColorType = Union[str, Iterable["int | float"], Color]
ColormapType = Union[str, Any]


class _Void:
    """A singleton class that represents a void value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
