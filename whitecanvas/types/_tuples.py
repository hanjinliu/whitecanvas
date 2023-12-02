from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar
import numpy as np
from numpy.typing import NDArray

_D = TypeVar("_D", bound=np.floating)


class XYData(NamedTuple, Generic[_D]):
    """
    Tuple of x and y array.

    Used for data of Line, Markers etc.
    """

    x: NDArray[_D]
    y: NDArray[_D]

    def stack(self) -> NDArray[_D]:
        """Data as a stacked (N, 2) array."""
        return np.stack([self.x, self.y], axis=1)


class XYYData(NamedTuple, Generic[_D]):
    """
    Tuple of x, y0, and y1 array.

    Used for data of Bars, Errorbars etc.
    """

    x: NDArray[_D]
    y0: NDArray[_D]
    y1: NDArray[_D]

    @property
    def ycenter(self) -> NDArray[_D]:
        return (self.y0 + self.y1) / 2

    @property
    def ydiff(self) -> NDArray[_D]:
        return self.y1 - self.y0


class Rect(NamedTuple):
    """Rectangular range."""

    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self) -> float:
        """Width of the range."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Height of the range."""
        return self.top - self.bottom

    @property
    def size(self) -> tuple[float, float]:
        """Size (width, height) of the range.""" ""
        return self.width, self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center of the range."""
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2
