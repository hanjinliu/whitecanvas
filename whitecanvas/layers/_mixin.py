from __future__ import annotations

from typing import Generic, Iterable, Iterator, TypeVar, overload, SupportsIndex
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.protocols import layer_protocols as _lp
from whitecanvas.layers._base import PrimitiveLayer, XYYData
from whitecanvas.types import LineStyle, FacePattern, ColorType, _Void, Alignment
from whitecanvas.utils.normalize import norm_color

_HasFaces = TypeVar("_HasFaces", bound=_lp.HasFaces)
_HasEdges = TypeVar("_HasEdges", bound=_lp.HasEdges)


class FaceMixin(PrimitiveLayer[_HasFaces]):
    @property
    def face_color(self) -> NDArray[np.floating]:
        """Face color of the bar."""
        return self._backend._plt_get_face_color()

    @face_color.setter
    def face_color(self, color):
        self._backend._plt_set_face_color(norm_color(color))

    @property
    def face_pattern(self) -> FacePattern:
        """Face fill pattern of the bars."""
        return FacePattern(self._backend._plt_get_face_pattern())

    @face_pattern.setter
    def face_pattern(self, style: str | FacePattern):
        self._backend._plt_set_face_pattern(FacePattern(style))


class MultiFaceMixin(PrimitiveLayer[_HasFaces]):
    @property
    def face_color(self) -> NDArray[np.floating]:
        """Face color of the bar."""
        return self._backend._plt_get_face_color()

    @face_color.setter
    def face_color(self, color):
        self._backend._plt_set_face_color(norm_color(color))

    @property
    def face_pattern(self) -> EnumArray[FacePattern]:
        """Face fill pattern of the bars."""
        return self._backend._plt_get_face_pattern()

    @face_pattern.setter
    def face_pattern(self, style: str | FacePattern | Iterable[str | FacePattern]):
        self._backend._plt_set_face_pattern(FacePattern(style))


class EdgeMixin(PrimitiveLayer[_HasEdges]):
    @property
    def edge_color(self) -> NDArray[np.floating]:
        """Edge color of the bar."""
        return self._backend._plt_get_edge_color()

    @edge_color.setter
    def edge_color(self, color):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def edge_width(self) -> float:
        return self._backend._plt_get_edge_width()

    @edge_width.setter
    def edge_width(self, width: float):
        self._backend._plt_set_edge_width(width)

    @property
    def edge_style(self) -> LineStyle:
        return LineStyle(self._backend._plt_get_edge_style())

    @edge_style.setter
    def edge_style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))


class MultiEdgeMixin(PrimitiveLayer[_HasEdges]):
    @property
    def edge_color(self) -> NDArray[np.floating]:
        """Edge color of the bar."""
        return self._backend._plt_get_edge_color()

    @edge_color.setter
    def edge_color(self, color):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def edge_width(self) -> NDArray[np.floating]:
        return self._backend._plt_get_edge_width()

    @edge_width.setter
    def edge_width(self, width: float):
        self._backend._plt_set_edge_width(width)

    @property
    def edge_style(self) -> EnumArray[LineStyle]:
        return self._backend._plt_get_edge_style()

    @edge_style.setter
    def edge_style(self, style: str | LineStyle | Iterable[str | LineStyle]):
        self._backend._plt_set_edge_style(LineStyle(style))


class LineMixin(PrimitiveLayer[_HasEdges]):
    @property
    def color(self) -> NDArray[np.floating]:
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def line_width(self):
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @line_width.setter
    def line_width(self, width):
        self._backend._plt_set_edge_width(width)

    @property
    def line_style(self) -> LineStyle:
        """Style of the line."""
        return LineStyle(self._backend._plt_get_edge_style())

    @line_style.setter
    def line_style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))


# just for typing
_E = TypeVar("_E", bound=Enum)


class EnumArray(Generic[_E]):
    @overload
    def __getitem__(self, key: SupportsIndex) -> _E:
        ...

    @overload
    def __getitem__(self, key: slice | NDArray[np.integer] | list[SupportsIndex]) -> EnumArray[_E]:
        ...

    def __iter__(self) -> Iterator[_E]:
        ...
