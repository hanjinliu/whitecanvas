from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import numpy as np

from whitecanvas.backend import Backend
from whitecanvas.layers._mixin import (
    EdgeNamespace,
    FaceNamespace,
    FontNamespace,
    TextMixin,
)
from whitecanvas.layers._sizehint import xy_size_hint
from whitecanvas.types import (
    Alignment,
    ArrayLike1D,
    ColorType,
    XYData,
    XYTextData,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xy

if TYPE_CHECKING:
    from typing_extensions import Self

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
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        family: str | None = None,
        backend: Backend | str | None = None,
    ):
        super().__init__(name=name)
        x = as_array_1d(x)
        y = as_array_1d(y)
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

    def _get_layer_data(self) -> XYTextData:
        x, y = self._backend._plt_get_text_position()
        t = np.array(self._backend._plt_get_text(), dtype=np.object_)
        return XYTextData(x, y, t)

    def _norm_layer_data(self, data: Any) -> XYTextData:
        xpos, ypos, t = data
        if isinstance(t, str):
            t = [t] * self.ndata
        else:
            t = [str(t0) for t0 in t]
        xdata, ydata = normalize_xy(xpos, ypos)
        if len(xdata) != len(t):
            raise ValueError(
                f"Length of x ({len(xdata)}) and y ({len(ydata)}) must be equal "
                f"to the number of texts ({len(t)})."
            )
        return XYTextData(xdata, ydata, np.array(t, dtype=np.object_))

    def _set_layer_data(self, data: XYTextData):
        self._backend._plt_set_text_position((data.x, data.y))
        self._backend._plt_set_text(data.text.tolist())

    @property
    def ntexts(self):
        """Number of texts."""
        return self.ndata

    @property
    def ndata(self):
        """Number of texts."""
        return len(self.string)

    @property
    def string(self) -> list[str]:
        """Text strings."""
        return self._backend._plt_get_text()

    @string.setter
    def string(self, text: str | list[str]):
        if isinstance(text, str):
            text = [text] * self.ndata
        self._backend._plt_set_text(text)

    @property
    def pos(self) -> XYData:
        """Position of the text."""
        return XYData(*self._backend._plt_get_text_position())

    @pos.setter
    def pos(self, pos: tuple[ArrayLike1D, ArrayLike1D]):
        return self.set_pos(*pos)

    def set_pos(self, x: ArrayLike1D | None = None, y: ArrayLike1D | None = None):
        """Set the position of the text."""
        if x is None or y is None:
            x0, y0 = self.pos
            if x is None:
                x = x0
            if y is None:
                y = y0
        xdata, ydata = normalize_xy(x, y)
        if xdata.size != self.ndata:
            raise ValueError(
                f"Length of x ({xdata.size}) and y ({ydata.size}) must be equal "
                f"to the number of texts ({self.ndata})."
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
        return self._backend._plt_get_text_anchor()

    @anchor.setter
    def anchor(self, anc: str | Alignment):
        self._backend._plt_set_text_anchor(Alignment(anc))

    @property
    def rotation(self) -> float:
        """Rotation of the text."""
        rot = self._backend._plt_get_text_rotation()
        if len(rot) > 0:
            return self._backend._plt_get_text_rotation()[0]
        return 0.0

    @rotation.setter
    def rotation(self, rotation: float):
        self._backend._plt_set_text_rotation(np.full(self.ndata, float(rotation)))
        self.events.rotation.emit(rotation)

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

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a Band from a dictionary."""
        return cls(
            d["data"]["x"], d["data"]["y"], d["data"]["text"], name=d["name"],
            rotation=d["rotation"], anchor=d["anchor"], backend=backend,
        ).with_face(
            color=d["face"]["color"], hatch=d["face"]["hatch"]
        ).with_edge(
            color=d["edge"]["color"], width=d["edge"]["width"], style=d["edge"]["style"]
        ).with_font(
            color=d["font"]["color"], size=d["font"]["size"], family=d["font"]["family"]
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "data": self._get_layer_data().to_dict(),
            "name": self.name,
            "face": self.face.to_dict(),
            "edge": self.edge.to_dict(),
            "font": self.font.to_dict(),
            "anchor": self.anchor.value,
            "rotation": self.rotation,
        }
