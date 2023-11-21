from __future__ import annotations

from typing import Generic, TypeVar, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from whitecanvas.layers import PrimitiveLayer
from whitecanvas.types import ColorType, LineStyle, FacePattern, _Void
from whitecanvas.utils.normalize import norm_color

if TYPE_CHECKING:
    from typing_extensions import Self

_void = _Void()
_L = TypeVar("_L", bound=PrimitiveLayer)


class LayerNamespace(Generic[_L]):
    def __init__(self, layer: _L | None = None) -> None:
        self.layer = layer
        self._instances: dict[int, _L] = {}

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
        color: ColorType | _Void = _void,
        pattern: FacePattern | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
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
        return self.layer._backend._plt_get_edge_style(LineStyle(value))

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
