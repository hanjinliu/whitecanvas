from __future__ import annotations

from typing import TYPE_CHECKING
import weakref
from numpy.typing import ArrayLike

from whitecanvas.protocols import TextProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.backend import Backend
from whitecanvas.types import Symbol, LineStyle, ColorType, FacePattern, _Void, Alignment
from whitecanvas.utils.normalize import norm_color
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
        size: float = 12,
        rotation: float = 0.0,
        anchor: Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str = "sans-serif",
        backend: Backend | str | None = None,
    ):
        self._backend = self._create_backend(Backend(backend), x, y, text)
        self._background_namespace = TextBackground(self)
        self.setup(
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
    def color(self, color: ColorType):
        self._backend._plt_set_text_color(norm_color(color))

    @property
    def size(self):
        """Size of the text."""
        return self._backend._plt_get_text_size()

    @size.setter
    def size(self, size: float):
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
    def pos(self):
        """Position of the text."""
        return self._backend._plt_get_text_position()

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
        self._backend._plt_set_text_fontfamily(fontfamily)

    def setup(
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

    def setup_background(
        self,
        face_color: ColorType | _Void = _void,
        face_pattern: FacePattern | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | _Void = _void,
    ) -> Text:
        if face_color is not _void:
            self._background_namespace.face_color = face_color
        if face_pattern is not _void:
            self._background_namespace.face_pattern = face_pattern
        if edge_color is not _void:
            self._background_namespace.edge_color = edge_color
        if edge_width is not _void:
            self._background_namespace.edge_width = edge_width
        if edge_style is not _void:
            self._background_namespace.edge_style = edge_style
        return self


class TextBackground:
    """The background namespace for the Text layer."""

    def __init__(self, text_layer: Text):
        self._layer_ref = weakref.ref(text_layer)

    def _layer(self) -> Text:
        layer = self._layer_ref()
        if layer is None:
            raise ReferenceDeletedError("The text layer has been deleted.")
        return layer

    @property
    def face_color(self):
        return self._layer()._backend._plt_get_face_color()

    @face_color.setter
    def face_color(self, color: ColorType):
        self._layer()._backend._plt_set_face_color(norm_color(color))

    @property
    def face_pattern(self):
        return self._layer()._backend._plt_get_face_pattern()

    @face_pattern.setter
    def face_pattern(self, pattern: FacePattern):
        self._layer()._backend._plt_set_face_pattern(pattern)

    @property
    def edge_color(self):
        return self._layer()._backend._plt_get_edge_color()

    @edge_color.setter
    def edge_color(self, color: ColorType):
        self._layer()._backend._plt_set_edge_color(norm_color(color))

    @property
    def edge_width(self):
        return self._layer()._backend._plt_get_edge_width()

    @edge_width.setter
    def edge_width(self, width: float):
        self._layer()._backend._plt_set_edge_width(width)

    @property
    def edge_style(self):
        return self._layer()._backend._plt_get_edge_style()

    @edge_style.setter
    def edge_style(self, style: LineStyle):
        self._layer()._backend._plt_set_edge_style(style)
