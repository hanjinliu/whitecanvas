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

    @classmethod
    def from_dict(cls, data: dict[str, NDArray[np.floating]]) -> XYData:
        """Create XYData from a dictionary."""
        return cls(data["x"], data["y"])

    def to_dict(self) -> dict[str, NDArray[np.floating]]:
        """Data as a dictionary."""
        return {"x": self.x, "y": self.y}


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

    @classmethod
    def from_dict(cls, data: dict[str, NDArray[np.floating]]) -> XYYData:
        """Create XYYData from a dictionary."""
        return cls(data["x"], data["y0"], data["y1"])

    def to_dict(self) -> dict[str, NDArray[np.floating]]:
        """Data as a dictionary."""
        return {"x": self.x, "y0": self.y0, "y1": self.y1}


class XYTextData(NamedTuple):
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    text: NDArray[np.object_]

    @classmethod
    def from_dict(cls, data: dict[str, NDArray[np.floating]]) -> XYTextData:
        """Create XYTextData from a dictionary."""
        return cls(data["x"], data["y"], data["text"])

    def to_dict(self) -> dict[str, NDArray[np.floating]]:
        """Data as a dictionary."""
        return {"x": self.x, "y": self.y, "text": self.text}


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

    @classmethod
    def from_dict(cls, data: dict[str, NDArray[np.floating]]) -> XYZData:
        """Create XYZData from a dictionary."""
        return cls(data["x"], data["y"], data["z"])

    def to_dict(self) -> dict[str, NDArray[np.floating]]:
        """Data as a dictionary."""
        return {"x": self.x, "y": self.y, "z": self.z}


class XYVectorData(NamedTuple):
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    vx: NDArray[np.floating]
    vy: NDArray[np.floating]

    @classmethod
    def from_dict(cls, data: dict[str, NDArray[np.floating]]) -> XYVectorData:
        """Create XYVectorData from a dictionary."""
        return cls(data["x"], data["y"], data["vx"], data["vy"])

    def to_dict(self) -> dict[str, NDArray[np.floating]]:
        """Data as a dictionary."""
        return {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy}


class XYZVectorData(NamedTuple):
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    z: NDArray[np.floating]
    vx: NDArray[np.floating]
    vy: NDArray[np.floating]
    vz: NDArray[np.floating]

    @classmethod
    def from_dict(cls, data: dict[str, NDArray[np.floating]]) -> XYZVectorData:
        """Create XYZVectorData from a dictionary."""
        return cls(data["x"], data["y"], data["z"], data["vx"], data["vy"], data["vz"])

    def to_dict(self) -> dict[str, NDArray[np.floating]]:
        """Data as a dictionary."""
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "vx": self.vx, "vy": self.vy, "vz": self.vz
        }  # fmt: skip


class MeshData(NamedTuple):
    vertices: NDArray[np.floating]
    faces: NDArray[np.intp]

    @classmethod
    def from_dict(cls, data: dict[str, NDArray[np.floating]]) -> MeshData:
        """Create MeshData from a dictionary."""
        return cls(data["vertices"], data["faces"])

    def to_dict(self) -> dict[str, NDArray[np.floating]]:
        """Data as a dictionary."""
        return {"vertices": self.vertices, "faces": self.faces}


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

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> Rect:
        """Create Rect from a dictionary."""
        return cls(data["left"], data["right"], data["bottom"], data["top"])

    def to_dict(self) -> dict[str, float]:
        """Data as a dictionary."""
        return {
            "left": self.left,
            "right": self.right,
            "bottom": self.bottom,
            "top": self.top,
        }
