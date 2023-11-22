from __future__ import annotations

from typing import Generic, Iterator, TypeVar, overload, SupportsIndex, TYPE_CHECKING
from enum import Enum
from weakref import WeakValueDictionary
import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.protocols import layer_protocols as _lp
from whitecanvas.layers._base import PrimitiveLayer, XYYData
from whitecanvas.types import LineStyle, FacePattern, ColorType, _Void, Alignment, _Void
from whitecanvas.utils.normalize import norm_color

if TYPE_CHECKING:
    from typing_extensions import Self
_HasFaces = TypeVar("_HasFaces", bound=_lp.HasFaces)
_HasEdges = TypeVar("_HasEdges", bound=_lp.HasEdges)
_P = TypeVar("_P", bound=_lp.BaseProtocol)
_L = TypeVar("_L", bound=PrimitiveLayer)
_void = _Void()


# class MultiFaceMixin(PrimitiveLayer[_HasFaces]):
#     @property
#     def face_color(self) -> NDArray[np.floating]:
#         """Face color of the bar."""
#         return self._backend._plt_get_face_color()

#     @face_color.setter
#     def face_color(self, color):
#         self._backend._plt_set_face_color(norm_color(color))

#     @property
#     def face_pattern(self) -> EnumArray[FacePattern]:
#         """Face fill pattern of the bars."""
#         return self._backend._plt_get_face_pattern()

#     @face_pattern.setter
#     def face_pattern(self, style: str | FacePattern | Iterable[str | FacePattern]):
#         self._backend._plt_set_face_pattern(FacePattern(style))


# class MultiEdgeMixin(PrimitiveLayer[_HasEdges]):
#     @property
#     def edge_color(self) -> NDArray[np.floating]:
#         """Edge color of the bar."""
#         return self._backend._plt_get_edge_color()

#     @edge_color.setter
#     def edge_color(self, color):
#         self._backend._plt_set_edge_color(norm_color(color))

#     @property
#     def edge_width(self) -> NDArray[np.floating]:
#         return self._backend._plt_get_edge_width()

#     @edge_width.setter
#     def edge_width(self, width: float):
#         self._backend._plt_set_edge_width(width)

#     @property
#     def edge_style(self) -> EnumArray[LineStyle]:
#         return self._backend._plt_get_edge_style()

#     @edge_style.setter
#     def edge_style(self, style: str | LineStyle | Iterable[str | LineStyle]):
#         self._backend._plt_set_edge_style(LineStyle(style))


class LayerNamespace(Generic[_L]):
    def __init__(self, layer: _L | None = None) -> None:
        self.layer = layer
        self._instances: WeakValueDictionary[int, _L] = WeakValueDictionary()

    def __get__(self, obj, owner) -> Self:
        if obj is None:
            return self
        _id = id(obj)
        if (ns := self._instances.get(_id, None)) is None:
            ns = self._instances[_id] = self.__class__(obj)
        return ns


class FaceNamespace(LayerNamespace[_L]):
    @property
    def color(self) -> NDArray[np.floating]:
        return self.layer._backend._plt_get_face_color()

    @color.setter
    def color(self, value: ColorType):
        return self.layer._backend._plt_set_face_color(norm_color(value))

    @property
    def pattern(self) -> FacePattern:
        return self.layer._backend._plt_get_face_pattern()

    @pattern.setter
    def pattern(self, value: str | FacePattern):
        return self.layer._backend._plt_set_face_pattern(FacePattern(value))

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        pattern: FacePattern | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        pattern : FacePattern, optional
            Fill pattern of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if pattern is not _void:
            self.pattern = pattern
        if alpha is not _void:
            self.alpha = alpha
        return self.layer


class AggFaceNamespace(LayerNamespace[_L]):
    @property
    def color(self) -> NDArray[np.floating]:
        return self.layer._backend._plt_get_face_color()[0]

    @color.setter
    def color(self, value: ColorType):
        return self.layer._backend._plt_set_face_color(norm_color(value))

    @property
    def pattern(self) -> FacePattern:
        return self.layer._backend._plt_get_face_pattern()[0]

    @pattern.setter
    def pattern(self, value: str | FacePattern):
        return self.layer._backend._plt_set_face_pattern(FacePattern(value))

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        pattern: FacePattern | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        pattern : FacePattern, optional
            Fill pattern of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if pattern is not _void:
            self.pattern = pattern
        if alpha is not _void:
            self.alpha = alpha
        return self.layer


class EdgeNamespace(LayerNamespace[_L]):
    @property
    def color(self) -> NDArray[np.floating]:
        return self.layer._backend._plt_get_edge_color()

    @color.setter
    def color(self, value: ColorType):
        return self.layer._backend._plt_set_edge_color(norm_color(value))

    @property
    def style(self) -> LineStyle:
        return self.layer._backend._plt_get_edge_style()

    @style.setter
    def style(self, value: str | LineStyle):
        return self.layer._backend._plt_set_edge_style(LineStyle(value))

    @property
    def width(self) -> float:
        return self.layer._backend._plt_get_edge_width()

    @width.setter
    def width(self, value: float):
        if value < 0:
            raise ValueError(f"Edge width must be non-negative, got {value!r}")
        return self.layer._backend._plt_set_edge_width(value)

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self.layer


class AggEdgeNamespace(LayerNamespace[_L]):
    @property
    def color(self) -> NDArray[np.floating]:
        return self.layer._backend._plt_get_edge_color()[0]

    @color.setter
    def color(self, value: ColorType):
        return self.layer._backend._plt_set_edge_color(norm_color(value))

    @property
    def style(self) -> LineStyle:
        return self.layer._backend._plt_get_edge_style()[0]

    @style.setter
    def style(self, value: str | LineStyle):
        return self.layer._backend._plt_set_edge_style(LineStyle(value))

    @property
    def width(self) -> float:
        return self.layer._backend._plt_get_edge_width()[0]

    @width.setter
    def width(self, value: float):
        if value < 0:
            raise ValueError(f"Edge width must be non-negative, got {value!r}")
        return self.layer._backend._plt_set_edge_width(value)

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self.layer


class LineMixin(PrimitiveLayer[_HasEdges]):
    @property
    def color(self) -> NDArray[np.floating]:
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def width(self):
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @width.setter
    def width(self, width):
        self._backend._plt_set_edge_width(width)

    @property
    def style(self) -> LineStyle:
        """Style of the line."""
        return LineStyle(self._backend._plt_get_edge_style())

    @style.setter
    def style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if width is not _void:
            self.width = width
        if style is not _void:
            self.style = style
        if alpha is not _void:
            self.alpha = alpha
        if antialias is not _void:
            self.antialias = antialias
        return self


class FaceEdgeMixin(PrimitiveLayer[_P]):
    face = FaceNamespace()
    edge = EdgeNamespace()

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        if color is None:
            color = self.face.color
        self.edge.color = color
        self.edge.width = width
        self.edge.style = style
        self.edge.alpha = alpha
        return self


class AggFaceEdgeMixin(PrimitiveLayer[_P]):
    face = AggFaceNamespace()
    edge = AggEdgeNamespace()

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        if color is None:
            color = self.face.color
        self.edge.color = color
        self.edge.width = width
        self.edge.style = style
        self.edge.alpha = alpha
        return self


# just for typing
_E = TypeVar("_E", bound=Enum)


class EnumArray(Generic[_E]):
    @overload
    def __getitem__(self, key: SupportsIndex) -> _E:
        ...

    @overload
    def __getitem__(
        self, key: slice | NDArray[np.integer] | list[SupportsIndex]
    ) -> EnumArray[_E]:
        ...

    def __iter__(self) -> Iterator[_E]:
        ...
