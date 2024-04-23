from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class XYData(NamedTuple):
    """
    Tuple of x and y array.

    Used for data of Line, Markers etc.
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]

    def stack(self) -> NDArray[np.floating]:
        """Data as a stacked (N, 2) array."""
        return np.stack([self.x, self.y], axis=1)


class XYYData(NamedTuple):
    """
    Tuple of x, y0, and y1 array.

    Used for data of Bars, Errorbars etc.
    """

    x: NDArray[np.floating]
    y0: NDArray[np.floating]
    y1: NDArray[np.floating]

    @property
    def ycenter(self) -> NDArray[np.floating]:
        return (self.y0 + self.y1) / 2

    @property
    def ydiff(self) -> NDArray[np.floating]:
        return self.y1 - self.y0


class XYTextData(NamedTuple):
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    text: NDArray[np.object_]


class XYZData(NamedTuple):
    """
    Tuple of x, y, and z array.

    Used for data of Surface etc.
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    z: NDArray[np.floating]

    def stack(self) -> NDArray[np.floating]:
        """Data as a stacked (N, 3) array."""
        return np.stack([self.x, self.y, self.z], axis=1)


class Rect(NamedTuple):
    """Rectangular range in left, right, bottom, top order."""

    left: float
    right: float
    bottom: float
    top: float

    def __repr__(self) -> str:
        return (
            f"Rect(left={self.left:.6g}, right={self.right:.6g}, "
            f"bottom={self.bottom:.6g}, top={self.top:.6g})"
        )

    @classmethod
    def with_check(cls, left: float, right: float, bottom: float, top: float):
        if left > right:
            raise ValueError("left must be less than or equal to right")
        if bottom > top:
            raise ValueError("bottom must be less than or equal to top")
        return cls(float(left), float(right), float(bottom), float(top))

    @classmethod
    def with_sort(cls, left: float, right: float, bottom: float, top: float):
        left, right = sorted([left, right])
        bottom, top = sorted([bottom, top])
        return cls(float(left), float(right), float(bottom), float(top))

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
        """Size (width, height) of the range."""
        return self.width, self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center of the range."""
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2
