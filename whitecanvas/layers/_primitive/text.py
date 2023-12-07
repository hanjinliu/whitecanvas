from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

from whitecanvas.layers._mixin import (
    TextMixin,
    FaceNamespace,
    EdgeNamespace,
    FontNamespace,
)
from whitecanvas.layers._sizehint import xy_size_hint
from whitecanvas.backend import Backend
from whitecanvas.types import (
    ColorType,
    _Void,
    Alignment,
    XYData,
    ArrayLike1D,
)
from whitecanvas.utils.normalize import normalize_xy


_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)
_Font = TypeVar("_Font", bound=FontNamespace)
_void = _Void()


class Texts(TextMixin[_Face, _Edge, _Font]):
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
        family: str | None = None,
        backend: Backend | str | None = None,
    ):
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), x, y, text)
        self.update(
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            family=family,
        )
        pad = 0.0  # TODO: better padding
        self._x_hint, self._y_hint = xy_size_hint(x, y, pad, pad)

    @property
    def ntexts(self):
        """Number of texts."""
        return len(self.string)

    @property
    def string(self) -> list[str]:
        """Text strings."""
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

    @property
    def color(self):
        """Color of the text."""
        return self.font.color

    @color.setter
    def color(self, color: ColorType | None):
        self.font.color = color

    @property
    def size(self):
        """Size of the text."""
        return self.font.size

    @size.setter
    def size(self, size: float | None):
        self.font.size = size

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
    def family(self) -> str:
        """Font family of the text."""
        return self.font.family

    @family.setter
    def family(self, fontfamily: str):
        self.font.family = fontfamily

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        size: float | _Void = _void,
        rotation: float | _Void = _void,
        anchor: Alignment | _Void = _void,
        family: str | _Void = _void,
    ) -> Texts:
        if rotation is not _void:
            self.rotation = rotation
        if anchor is not _void:
            self.anchor = anchor
        self.font.update(color=color, size=size, family=family)
        return self
