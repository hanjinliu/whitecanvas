from typing import Union, Iterable
from cmap import Color

ColorType = Union[str, Iterable["int | float"], Color]


class _Void:
    """A singleton class that represents a void value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
