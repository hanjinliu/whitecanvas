from __future__ import annotations

from enum import Enum
from typing import NamedTuple, overload

import numpy as np
from numpy.typing import NDArray

from whitecanvas.tools._polygon_utils import is_in_polygon
from whitecanvas.types import Point, Rect, XYData


class PointSelection(Point):
    """A selection that defines a point."""


class LineSelection(NamedTuple):
    """A selection that defines a line."""

    start: Point
    end: Point


class SpanSelection(NamedTuple):
    """A selection that defines a span."""

    start: float
    end: float

    @overload
    def contains_point(self, point: tuple[float, float], /) -> bool: ...
    @overload
    def contains_point(self, x: float, y: float, /) -> bool: ...

    def contains_point(self, *args) -> bool:
        x_or_y = _point_to_xy(*args)[self._slice_index()]
        return self.start <= x_or_y <= self.end

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        points = _atleast_2d(points)
        if points.ndim == 2 and points.shape[1] == 2:
            xs = points[:, self._slice_index()]
            return (self.start <= xs) & (xs <= self.end)
        else:
            raise ValueError("points must be (2,) or (N, 2) array.")

    def _slice_index(self):
        raise NotImplementedError()


class XSpanSelection(SpanSelection):
    """A selection that defines a span along the x-axis."""

    def _slice_index(self) -> int:
        return 0


class YSpanSelection(SpanSelection):
    """A selection that defines a span along the y-axis."""

    def _slice_index(self) -> int:
        return 1


class RectSelection(Rect):
    """A selection that defines a rectangle."""

    @overload
    def contains_point(self, point: tuple[float, float], /) -> bool: ...
    @overload
    def contains_point(self, x: float, y: float, /) -> bool: ...

    def contains_point(self, *args) -> bool:
        x, y = _point_to_xy(*args)
        return self.left <= x <= self.right and self.bottom <= y <= self.top

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        points = _atleast_2d(points)
        if points.ndim == 2 and points.shape[1] == 2:
            xs = points[:, 0]
            ys = points[:, 1]
            return (
                (self.left <= xs)
                & (xs <= self.right)
                & (self.bottom <= ys)
                & (ys <= self.top)
            )
        else:
            raise ValueError("points must be (2,) or (N, 2) array.")


class PolygonSelection(XYData):
    """A selection that defines a polygon."""

    @overload
    def contains_point(self, point: tuple[float, float], /) -> bool: ...
    @overload
    def contains_point(self, x: float, y: float, /) -> bool: ...

    def contains_point(self, *args) -> bool:
        x, y = _point_to_xy(*args)
        return is_in_polygon(np.array([[x, y]]), self.stack())[0]

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        points = _atleast_2d(points)
        if points.ndim == 2 and points.shape[1] == 2:
            return is_in_polygon(points, self.stack())
        else:
            raise ValueError("points must be (2,) or (N, 2) array.")


class SelectionMode(Enum):
    NONE = "none"
    LINE = "line"
    POINT = "point"
    RECT = "rect"
    XSPAN = "xspan"
    YSPAN = "yspan"
    LASSO = "lasso"
    POLYGON = "polygon"


def _point_to_xy(*args) -> tuple[float, float]:
    if len(args) == 1:
        x, y = args[0]
    else:
        x, y = args
    return x, y


def _atleast_2d(points: NDArray[np.number]) -> NDArray[np.number]:
    if isinstance(points, XYData):
        return points.stack()
    return np.atleast_2d(points)
