from typing import Union, Iterable
from cmap import Color

ColorType = Union[str, Iterable["int | float"], Color]


class _Void:
    """Internal sentinel class."""
