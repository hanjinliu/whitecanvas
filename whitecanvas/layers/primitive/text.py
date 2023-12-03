from __future__ import annotations

from typing import TYPE_CHECKING
import weakref

import numpy as np
from numpy.typing import NDArray

from whitecanvas.protocols import TextProtocol
from whitecanvas.layers._base import PrimitiveLayer
from whitecanvas.backend import Backend
from whitecanvas.types import (
    LineStyle,
    ColorType,
    FacePattern,
    _Void,
    Alignment,
)
from whitecanvas.utils.normalize import arr_color
from whitecanvas.theme import get_theme
from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg

_void = _Void()


class Text(PrimitiveLayer[TextProtocol]):
    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        *,
        color: ColorType = "black",
        size: float | None = None,
        rotation: float = 0.0,
        anchor: Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
        backend: Backend | str | None = None,
    ):
        super().__init__()
        self._backend = self._create_backend(Backend(backend), x, y, text)
        self._background_face = TextBackgroundFace(self)
        self._background_edge = TextBackgroundEdge(self)
        self.update(
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            fontfamily=fontfamily,
        )

    @property
    def name(self):
        return self.string

    @property
    def color(self):
        """Color of the text."""
        return self._backend._plt_get_text_color()

    @color.setter
    def color(self, color: ColorType | None):
        if color is None:
            color = get_theme().foreground_color
        self._backend._plt_set_text_color(arr_color(color))

    @property
    def size(self):
        """Size of the text."""
        return self._backend._plt_get_text_size()

    @size.setter
    def size(self, size: float | None):
        if size is None:
            size = get_theme().fontsize
        self._backend._plt_set_text_size(float(size))

    @property
    def string(self):
        """Text of the text."""
        return self._backend._plt_get_text()

    @string.setter
    def string(self, text: str):
        self._backend._plt_set_text(text)

    @property
    def anchor(self) -> Alignment:
        """Anchor of the text."""
        return self._backend._plt_get_text_anchor()

    @anchor.setter
    def anchor(self, anc: str | Alignment):
        self._backend._plt_set_text_anchor(Alignment(anc))

    @property
    def pos(self) -> NDArray[np.floating]:
        """Position of the text."""
        return np.asarray(self._backend._plt_get_text_position())

    @pos.setter
    def pos(self, position: tuple[float, float]):
        self._backend._plt_set_text_position(position)

    @property
    def rotation(self) -> float:
        """Rotation of the text."""
        return self._backend._plt_get_text_rotation()

    @rotation.setter
    def rotation(self, rotation: float):
        self._backend._plt_set_text_rotation(float(rotation))

    @property
    def fontfamily(self) -> str:
        """Font family of the text."""
        return self._backend._plt_get_text_fontfamily()

    @fontfamily.setter
    def fontfamily(self, fontfamily: str):
        if fontfamily is None:
            fontfamily = get_theme().fontfamily
        self._backend._plt_set_text_fontfamily(fontfamily)

    def update(
        self,
        color: ColorType | _Void = _void,
        size: float | _Void = _void,
        rotation: float | _Void = _void,
        anchor: Alignment | _Void = _void,
        fontfamily: str | _Void = _void,
    ) -> Text:
        if color is not _void:
            self.color = color
        if size is not _void:
            self.size = size
        if rotation is not _void:
            self.rotation = rotation
        if anchor is not _void:
            self.anchor = anchor
        if fontfamily is not _void:
            self.fontfamily = fontfamily
        return self


class TextNamespace:
    def __init__(self, text_layer: Text):
        self._layer_ref = weakref.ref(text_layer)

    def _layer(self) -> Text:
        layer = self._layer_ref()
        if layer is None:
            raise ReferenceDeletedError("The text layer has been deleted.")
        return layer


class TextBackgroundFace(TextNamespace):
    @property
    def color(self):
        return self._layer()._backend._plt_get_face_color()

    @color.setter
    def color(self, color: ColorType):
        self._layer()._backend._plt_set_face_color(arr_color(color))

    @property
    def face_pattern(self):
        return self._layer()._backend._plt_get_face_pattern()

    @face_pattern.setter
    def face_pattern(self, pattern: FacePattern):
        self._layer()._backend._plt_set_face_pattern(pattern)

    @property
    def alpha(self):
        return self.color[3]

    @alpha.setter
    def alpha(self, alpha: float):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1.")
        self.color = (*self.color[:3], alpha)

    def update(
        self,
        color: ColorType | _Void = _void,
        pattern: FacePattern | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> Text:
        if color is not _void:
            self.color = color
        if pattern is not _void:
            self.pattern = pattern
        if alpha is not _void:
            self.alpha = alpha
        return self._layer()


class TextBackgroundEdge(TextNamespace):
    @property
    def color(self):
        return self._layer()._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        self._layer()._backend._plt_set_edge_color(arr_color(color))

    @property
    def width(self):
        return self._layer()._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float):
        self._layer()._backend._plt_set_edge_width(width)

    @property
    def style(self):
        return self._layer()._backend._plt_get_edge_style()

    @style.setter
    def style(self, style: LineStyle):
        self._layer()._backend._plt_set_edge_style(style)

    @property
    def alpha(self):
        return self.color[3]

    @alpha.setter
    def alpha(self, alpha: float):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1.")
        self.color = (*self.color[:3], alpha)

    def update(
        self,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ) -> Text:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer()
