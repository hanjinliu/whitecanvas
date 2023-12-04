from __future__ import annotations

from typing import Sequence
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
    XYData,
    ArrayLike1D,
)
from whitecanvas.utils.normalize import (
    as_any_1d_array,
    as_array_1d,
    as_color_array,
    normalize_xy,
)
from whitecanvas.theme import get_theme
from whitecanvas._exceptions import ReferenceDeletedError


_void = _Void()


class TextBase(PrimitiveLayer[TextProtocol]):
    @property
    def ntexts(self):
        """Number of texts."""
        return len(self.string)

    @property
    def face(self):
        """Background face of the text."""
        return self._background_face

    @property
    def edge(self):
        """Background edge of the text."""
        return self._background_edge

    @property
    def string(self) -> list[str]:
        """Text of the text."""
        return self._backend._plt_get_text()

    @string.setter
    def string(self, text: str | list[str]):
        if isinstance(text, str):
            text = [text] * self.ntexts
        self._backend._plt_set_text(text)

    @property
    def pos(self) -> XYData:
        """Position of the text."""
        return XYData(*self._backend._plt_get_text_position())

    def set_pos(self, xpos: ArrayLike1D, ypos: ArrayLike1D):
        """Set the position of the text."""
        if xpos is None or ypos is None:
            x0, y0 = self.pos
            if xpos is None:
                xpos = x0
            if ypos is None:
                ypos = y0
        xdata, ydata = normalize_xy(xpos, ypos)
        if xdata.size != self.ntexts:
            raise ValueError(
                f"Length of x ({xdata.size}) and y ({ydata.size}) must be equal "
                f"to the number of texts ({self.ntexts})."
            )
        self._backend._plt_set_text_position((xdata, ydata))


class Texts(TextBase):
    def __init__(
        self,
        x: ArrayLike1D,
        y: ArrayLike1D,
        text: Sequence[str],
        *,
        name: str | None = None,
        color: ColorType = "black",
        size: float | None = None,
        rotation: float = 0.0,
        anchor: Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
        backend: Backend | str | None = None,
    ):
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), x, y, text)
        self._background_face = AggTextBackgroundFace(self)
        self._background_edge = AggTextBackgroundEdge(self)
        self.update(
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            fontfamily=fontfamily,
        )

    @property
    def color(self):
        """Color of the text."""
        return self._backend._plt_get_text_color()[0]

    @color.setter
    def color(self, color: ColorType | None):
        if color is None:
            color = get_theme().foreground_color
        col2d = as_color_array(color, self.ntexts)
        self._backend._plt_set_text_color(col2d)

    @property
    def size(self):
        """Size of the text."""
        return self._backend._plt_get_text_size()[0]

    @size.setter
    def size(self, size: float | None):
        if size is None:
            size = get_theme().fontsize
        self._backend._plt_set_text_size(np.full(self.ntexts, float(size)))

    @property
    def anchor(self) -> Alignment:
        """Anchor of the text."""
        return self._backend._plt_get_text_anchor()[0]

    @anchor.setter
    def anchor(self, anc: str | Alignment):
        self._backend._plt_set_text_anchor(Alignment(anc))

    @property
    def rotation(self) -> float:
        """Rotation of the text."""
        return self._backend._plt_get_text_rotation()[0]

    @rotation.setter
    def rotation(self, rotation: float):
        self._backend._plt_set_text_rotation(np.full(self.ntexts, float(rotation)))

    @property
    def fontfamily(self) -> str:
        """Font family of the text."""
        return self._backend._plt_get_text_fontfamily()[0]

    @fontfamily.setter
    def fontfamily(self, fontfamily: str):
        if fontfamily is None:
            fontfamily = get_theme().fontfamily
        elif not isinstance(fontfamily, str):
            raise TypeError(f"fontfamily must be a string, got {type(fontfamily)}.")
        self._backend._plt_set_text_fontfamily([fontfamily] * self.ntexts)

    def update(
        self,
        color: ColorType | _Void = _void,
        size: float | _Void = _void,
        rotation: float | _Void = _void,
        anchor: Alignment | _Void = _void,
        fontfamily: str | _Void = _void,
    ) -> Texts:
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


class HeteroText(TextBase):
    def __init__(
        self,
        x: ArrayLike1D,
        y: ArrayLike1D,
        text: Sequence[str],
        *,
        color: ColorType = "black",
        size: float | Sequence[float] | None = None,
        rotation: float | Sequence[float] = 0.0,
        anchor: str | Alignment | list[str | Alignment] = Alignment.BOTTOM_LEFT,
        fontfamily: str | Sequence[str] | None = None,
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
    def color(self):
        """Color of the text."""
        return self._backend._plt_get_text_color()

    @color.setter
    def color(self, color: ColorType | None):
        if color is None:
            color = get_theme().foreground_color
        col2d = as_color_array(color, self.ntexts)
        self._backend._plt_set_text_color(col2d)

    @property
    def size(self):
        """Size of the text."""
        return self._backend._plt_get_text_size()

    @size.setter
    def size(self, size: float | None):
        if size is None:
            size = get_theme().fontsize
        size = as_any_1d_array(size, self.ntexts)
        self._backend._plt_set_text_size(size)

    @property
    def anchor(self) -> Alignment:
        """Anchor of the text."""
        return self._backend._plt_get_text_anchor()

    @anchor.setter
    def anchor(self, anc: str | Alignment | list[str | Alignment]):
        if isinstance(anc, (str, Alignment)):
            anc = [Alignment(anc)] * self.ntexts
        self._backend._plt_set_text_anchor(anc)

    @property
    def rotation(self) -> float:
        """Rotation of the text."""
        return self._backend._plt_get_text_rotation()

    @rotation.setter
    def rotation(self, rotation: float):
        rotation = as_any_1d_array(rotation, self.ntexts)
        self._backend._plt_set_text_rotation(rotation)

    @property
    def fontfamily(self) -> str:
        """Font family of the text."""
        return self._backend._plt_get_text_fontfamily()

    @fontfamily.setter
    def fontfamily(self, fontfamily: str):
        if fontfamily is None:
            fontfamily = get_theme().fontfamily
        elif not isinstance(fontfamily, str):
            raise TypeError(f"fontfamily must be a string, got {type(fontfamily)}.")
        self._backend._plt_set_text_fontfamily([fontfamily] * self.ntexts)

    def update(
        self,
        color: ColorType | _Void = _void,
        size: float | _Void = _void,
        rotation: float | _Void = _void,
        anchor: Alignment | _Void = _void,
        fontfamily: str | _Void = _void,
    ) -> Texts:
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
    def __init__(self, text_layer: Texts):
        self._layer_ref = weakref.ref(text_layer)

    def _layer(self) -> Texts:
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
        color = as_color_array(color, self._layer().ntexts)
        self._layer()._backend._plt_set_face_color(color)

    @property
    def pattern(self):
        return self._layer()._backend._plt_get_face_pattern()

    @pattern.setter
    def pattern(self, pattern: FacePattern | list[FacePattern]):
        if isinstance(pattern, str):
            pattern = FacePattern(pattern)
        else:
            pattern = [FacePattern(p) for p in pattern]
        self._layer()._backend._plt_set_face_pattern(pattern)

    @property
    def alpha(self):
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, alpha: float):
        if isinstance(alpha, (int, float, np.number)):
            if not 0 <= alpha <= 1:
                raise ValueError("Alpha must be between 0 and 1.")
            alpha = np.full(self._layer().ntexts, float(alpha))
        else:
            alpha = as_array_1d(alpha)
            if np.any(alpha < 0) or np.any(alpha > 1):
                raise ValueError("Alpha must be between 0 and 1.")

        self.color = np.column_stack([self.color[:, :3], alpha])

    def update(
        self,
        color: ColorType | _Void = _void,
        pattern: str | FacePattern | list[str | FacePattern] | _Void = _void,
        alpha: float | Sequence[float] | _Void = _void,
    ) -> Texts:
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
        color = as_color_array(color, self._layer().ntexts)
        self._layer()._backend._plt_set_edge_color(color)

    @property
    def width(self):
        return self._layer()._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float):
        if isinstance(width, (int, float, np.number)):
            width = np.full(self._layer().ntexts, float(width))
        self._layer()._backend._plt_set_edge_width(width)

    @property
    def style(self):
        return self._layer()._backend._plt_get_edge_style()

    @style.setter
    def style(self, style: str | LineStyle | list[str | LineStyle]):
        if isinstance(style, str):
            style = LineStyle(style)
        else:
            style = [LineStyle(s) for s in style]
        self._layer()._backend._plt_set_edge_style(style)

    @property
    def alpha(self):
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, alpha: float):
        if isinstance(alpha, (int, float, np.number)):
            if not 0 <= alpha <= 1:
                raise ValueError("Alpha must be between 0 and 1.")
            alpha = np.full(self._layer().ntexts, float(alpha))
        else:
            alpha = as_array_1d(alpha)
            if np.any(alpha < 0) or np.any(alpha > 1):
                raise ValueError("Alpha must be between 0 and 1.")

        self.color = np.column_stack([self.color[:, :3], alpha])

    def update(
        self,
        color: ColorType | _Void = _void,
        style: str | LineStyle | list[str | LineStyle] | _Void = _void,
        width: float | Sequence[float] | _Void = _void,
        alpha: float | Sequence[float] | _Void = _void,
    ) -> Texts:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer()


class AggTextBackgroundFace(TextNamespace):
    @property
    def color(self):
        return self._layer()._backend._plt_get_face_color()[0]

    @color.setter
    def color(self, color: ColorType):
        color = as_color_array(color, self._layer().ntexts)
        self._layer()._backend._plt_set_face_color(color)

    @property
    def pattern(self):
        return self._layer()._backend._plt_get_face_pattern()[0]

    @pattern.setter
    def pattern(self, pattern: str | FacePattern):
        self._layer()._backend._plt_set_face_pattern(FacePattern(pattern))

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
    ) -> Texts:
        if color is not _void:
            self.color = color
        if pattern is not _void:
            self.pattern = pattern
        if alpha is not _void:
            self.alpha = alpha
        return self._layer()


class AggTextBackgroundEdge(TextNamespace):
    @property
    def color(self):
        return self._layer()._backend._plt_get_edge_color()[0]

    @color.setter
    def color(self, color: ColorType):
        color = as_color_array(color, self._layer().ntexts)
        self._layer()._backend._plt_set_edge_color(color)

    @property
    def width(self):
        return self._layer()._backend._plt_get_edge_width()[0]

    @width.setter
    def width(self, width: float):
        if width < 0:
            raise ValueError("Width must be non-negative.")
        w = np.full(self._layer().ntexts, width, dtype=np.float32)
        self._layer()._backend._plt_set_edge_width(w)

    @property
    def style(self):
        return self._layer()._backend._plt_get_edge_style()[0]

    @style.setter
    def style(self, style: str | LineStyle):
        self._layer()._backend._plt_set_edge_style(LineStyle(style))

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
    ) -> Texts:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer()
