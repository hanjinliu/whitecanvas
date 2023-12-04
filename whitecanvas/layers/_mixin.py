from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    TypeVar,
    overload,
    SupportsIndex,
    TYPE_CHECKING,
)
from enum import Enum
from weakref import WeakValueDictionary
import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.protocols import layer_protocols as _lp
from whitecanvas.layers._base import PrimitiveLayer
from whitecanvas.types import LineStyle, FacePattern, ColorType, _Void, _Void
from whitecanvas.utils.normalize import arr_color, as_color_array

if TYPE_CHECKING:
    from typing_extensions import Self

_HasEdges = TypeVar("_HasEdges", bound=_lp.HasEdges)
_P = TypeVar("_P", bound=_lp.BaseProtocol)
_L = TypeVar("_L", bound=PrimitiveLayer)
_void = _Void()


class LayerNamespace(ABC, Generic[_L]):
    _properties = ()

    def __init__(self, layer: _L | None = None) -> None:
        self._layer = layer
        self._instances: WeakValueDictionary[int, _L] = WeakValueDictionary()
        self.__setattr__ = self._setattr

    def __repr__(self) -> str:
        attrs = ", ".join(f"{p}={getattr(self, p)!r}" for p in self._properties)
        return f"{type(self).__name__}(layer={self._layer!r}, {attrs})"

    def __repr_pretty__(self, p, cycle):
        attrs = ",\n    ".join(f"{p}={getattr(self, p)!r}" for p in self._properties)
        p.text(f"{type(self).__name__}(\n    layer={self._layer!r}, {attrs}\n)")

    def __get__(self, obj, owner) -> Self:
        if obj is None:
            return self
        _id = id(obj)
        if (ns := self._instances.get(_id, None)) is None:
            ns = self._instances[_id] = self.__class__(obj)
        return ns

    def _setattr(self, name: str, value: Any) -> None:
        raise AttributeError(f"Cannot set attribute {name!r} on {self!r}")


class FaceNamespace(LayerNamespace[PrimitiveLayer[_lp.HasFaces]]):
    _properties = ("color", "pattern")

    @abstractproperty
    def color(self):
        raise NotImplementedError

    @abstractproperty
    def pattern(self):
        raise NotImplementedError

    @abstractproperty
    def alpha(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, color, pattern, alpha):
        ...


class EdgeNamespace(LayerNamespace[PrimitiveLayer[_lp.HasEdges]]):
    _properties = ("color", "style", "width")

    @abstractproperty
    def color(self):
        raise NotImplementedError

    @abstractproperty
    def width(self):
        raise NotImplementedError

    @abstractproperty
    def style(self):
        raise NotImplementedError

    @abstractproperty
    def alpha(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, color, width, style, alpha):
        ...


class MonoFace(FaceNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_face_color()

    @color.setter
    def color(self, value: ColorType):
        return self._layer._backend._plt_set_face_color(arr_color(value))

    @property
    def pattern(self) -> FacePattern:
        return self._layer._backend._plt_get_face_pattern()

    @pattern.setter
    def pattern(self, value: str | FacePattern):
        return self._layer._backend._plt_set_face_pattern(FacePattern(value))

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
        return self._layer


class ConstFace(FaceNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_face_color()[0]

    @color.setter
    def color(self, value: ColorType):
        return self._layer._backend._plt_set_face_color(arr_color(value))

    @property
    def pattern(self) -> FacePattern:
        return self._layer._backend._plt_get_face_pattern()[0]

    @pattern.setter
    def pattern(self, value: str | FacePattern):
        return self._layer._backend._plt_set_face_pattern(FacePattern(value))

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
        return self._layer


class MonoEdge(EdgeNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_edge_color()

    @color.setter
    def color(self, value: ColorType):
        return self._layer._backend._plt_set_edge_color(arr_color(value))

    @property
    def style(self) -> LineStyle:
        return self._layer._backend._plt_get_edge_style()

    @style.setter
    def style(self, value: str | LineStyle):
        return self._layer._backend._plt_set_edge_style(LineStyle(value))

    @property
    def width(self) -> float:
        return self._layer._backend._plt_get_edge_width()

    @width.setter
    def width(self, value: float):
        if value < 0:
            raise ValueError(f"Edge width must be non-negative, got {value!r}")
        return self._layer._backend._plt_set_edge_width(value)

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
        return self._layer


class ConstEdge(EdgeNamespace):
    _properties = ("color", "style", "width")

    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_edge_color()[0]

    @color.setter
    def color(self, value: ColorType):
        return self._layer._backend._plt_set_edge_color(arr_color(value))

    @property
    def style(self) -> LineStyle:
        return self._layer._backend._plt_get_edge_style()[0]

    @style.setter
    def style(self, value: str | LineStyle):
        return self._layer._backend._plt_set_edge_style(LineStyle(value))

    @property
    def width(self) -> float:
        return self._layer._backend._plt_get_edge_width()[0]

    @width.setter
    def width(self, value: float):
        if value < 0:
            raise ValueError(f"Edge width must be non-negative, got {value!r}")
        return self._layer._backend._plt_set_edge_width(value)

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
        return self._layer


class MultiFace(FaceNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        """Face color of the bar."""
        return self._layer._backend._plt_get_face_color()

    @color.setter
    def color(self, color):
        self._layer._backend._plt_set_face_color(
            as_color_array(color, self._layer.ndata)
        )

    @property
    def pattern(self) -> EnumArray[FacePattern]:
        """Face fill pattern of the bars."""
        return self._layer._backend._plt_get_face_pattern()

    @pattern.setter
    def pattern(self, pattern: str | FacePattern | Iterable[str | FacePattern]):
        if isinstance(pattern, str):
            pattern = FacePattern(pattern)
        elif hasattr(pattern, "__iter__"):
            pattern = [FacePattern(p) for p in pattern]
        self._layer._backend._plt_set_face_pattern(FacePattern(pattern))

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

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
        return self._layer


class MultiEdge(EdgeNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        """Edge color of the bar."""
        return self._layer._backend._plt_get_edge_color()

    @color.setter
    def color(self, color):
        self._layer._backend._plt_set_edge_color(
            as_color_array(color, self._layer.ndata)
        )

    @property
    def width(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float):
        self._layer._backend._plt_set_edge_width(width)

    @property
    def style(self) -> EnumArray[LineStyle]:
        return self._layer._backend._plt_get_edge_style()

    @style.setter
    def style(self, style: str | LineStyle | Iterable[str | LineStyle]):
        if isinstance(style, str):
            style = LineStyle(style)
        elif hasattr(style, "__iter__"):
            style = [LineStyle(s) for s in style]
        self._layer._backend._plt_set_edge_style(style)

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

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
        return self._layer


class LineMixin(PrimitiveLayer[_HasEdges]):
    @property
    def color(self) -> NDArray[np.floating]:
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        self._backend._plt_set_edge_color(arr_color(color))

    @property
    def width(self) -> float:
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float):
        if not isinstance(width, (int, float, np.number)):
            raise TypeError(f"Width must be a number, got {type(width)}")
        if width < 0:
            raise ValueError(f"Width must be non-negative, got {width!r}")
        self._backend._plt_set_edge_width(float(width))

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


_NFace = TypeVar("_NFace", bound=FaceNamespace)
_NEdge = TypeVar("_NEdge", bound=EdgeNamespace)


class _AbstractFaceEdgeMixin(PrimitiveLayer[_P], Generic[_P, _NFace, _NEdge]):
    face: _NFace
    edge: _NEdge
    _face_namespace: _NFace
    _edge_namespace: _NEdge

    @property
    def face(self) -> _NFace:
        return self._face_namespace

    @property
    def edge(self) -> _NEdge:
        return self._edge_namespace


class FaceEdgeMixin(_AbstractFaceEdgeMixin[_P, MonoFace, MonoEdge], Generic[_P]):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self._face_namespace = MonoFace(self)
        self._edge_namespace = MonoEdge(self)

    def with_face(
        self,
        color: ColorType | None = None,
        pattern: FacePattern | str = FacePattern.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the face properties."""
        if color is None:
            color = self.face.color
        self.face.color = color
        self.face.pattern = pattern
        self.face.alpha = alpha
        return self

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = self.face.color
        self.edge.color = color
        self.edge.width = width
        self.edge.style = style
        self.edge.alpha = alpha
        return self


class MultiFaceEdgeMixin(
    _AbstractFaceEdgeMixin[_P, _NFace, _NEdge], Generic[_P, _NFace, _NEdge]
):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self._face_namespace = ConstFace(self)
        self._edge_namespace = ConstEdge(self)

    def with_face(
        self,
        color: ColorType | None = None,
        pattern: FacePattern | str = FacePattern.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the face properties."""
        if color is None:
            color = self.face.color
        if not isinstance(self._face_namespace, ConstFace):
            self._face_namespace = ConstFace(self)  # type: ignore
        self.face.color = color
        self.face.pattern = pattern
        self.face.alpha = alpha
        return self

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = self.face.color
        if not isinstance(self._edge_namespace, ConstEdge):
            self._edge_namespace = ConstEdge(self)  # type: ignore
        self.edge.color = color
        self.edge.width = width
        self.edge.style = style
        self.edge.alpha = alpha
        return self

    def with_face_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        pattern: str | FacePattern | Sequence[str | FacePattern] = FacePattern.SOLID,
        alpha: float = 1,
    ) -> Self:
        if color is None:
            color = self.face.color
        if not isinstance(self._face_namespace, MultiFace):
            self._face_namespace = MultiFace(self)  # type: ignore
        self.face.color = color
        self.face.pattern = pattern
        self.face.alpha = alpha
        return self

    def with_edge_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = self.face.color
        if not isinstance(self._edge_namespace, MultiEdge):
            self._edge_namespace = MultiEdge(self)  # type: ignore
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
