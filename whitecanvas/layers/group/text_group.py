from __future__ import annotations
from typing import Sequence

from enum import Enum
import numpy as np
from numpy.typing import NDArray, ArrayLike
from whitecanvas.types import ColorType, _Void, Alignment
from whitecanvas.layers.primitive import Text
from whitecanvas.layers._base import LayerGroup
from whitecanvas.backend import Backend
from whitecanvas.utils.normalize import normalize_xy, norm_color

_void = _Void()


class TextGroup(LayerGroup):
    _children: list[Text]

    def __init__(self, texts: list[Text], name: str | None = None):
        if name is None:
            name = "TextGroup"
        super().__init__(texts, name=name)

    def __repr__(self) -> str:
        ntext = len(self._children)
        return f"{self.__class__.__name__}<{ntext} texts>"

    @classmethod
    def from_strings(
        cls,
        xs: ArrayLike,
        ys: ArrayLike,
        strings: Sequence[str],
        *,
        name: str | None = None,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str = "sans-serif",
        backend: Backend | str | None = None,
    ) -> TextGroup:
        xs, ys = normalize_xy(xs, ys)
        ndata = xs.size
        if len(strings) != ndata:
            raise ValueError(f"Expected {ndata} texts, got {len(strings)}")
        _zip = zip(
            xs,
            ys,
            strings,
            _as_color_array(color, ndata),
            _asarray_1d(size, ndata, dtype=np.float32),
            _asarray_1d(rotation, ndata, dtype=np.float32),
            _asarray_1d(anchor, ndata, dtype=object),
            _asarray_1d(fontfamily, ndata, dtype=object),
        )
        texts = [
            Text(
                x, y, v, color=_color, size=_size, rotation=_rot, anchor=_anc,
                fontfamily=_ff, backend=backend
            )
            for x, y, v, _color, _size, _rot, _anc, _ff in _zip
        ]  # fmt: skip
        return cls(texts, name=name)

    def nth(self, n: int) -> Text:
        """The central text layer."""
        return self._children[n]

    @property
    def string(self) -> list[str]:
        """The text values."""
        return np.array([tl.string for tl in self._children], dtype=object)

    @string.setter
    def string(self, values: list[str]):
        if len(values) != len(self._children):
            raise ValueError(f"Expected {len(self._children)} values, got {len(values)}")
        for tl, v in zip(self._children, values):
            tl.string = v

    @property
    def color(self) -> np.ndarray:
        """The text colors."""
        return np.stack([tl.color for tl in self._children], axis=0)

    @color.setter
    def color(self, colors: ColorType):
        values = _as_color_array(colors, len(self._children))
        for tl, v in zip(self._children, values):
            tl.color = v

    @property
    def size(self) -> np.ndarray:
        """The text sizes."""
        return np.array([tl.size for tl in self._children], dtype=np.float32)

    @size.setter
    def size(self, values: float):
        sizes = _asarray_1d(values, len(self._children), dtype=np.float32)
        for tl, v in zip(self._children, sizes):
            tl.size = v

    @property
    def rotation(self) -> np.ndarray:
        """The text rotations."""
        return np.array([tl.rotation for tl in self._children], dtype=np.float32)

    @rotation.setter
    def rotation(self, values: float):
        rots = _asarray_1d(values, len(self._children), dtype=np.float32)
        for tl, v in zip(self._children, rots):
            tl.rotation = v

    @property
    def anchor(self) -> np.ndarray:
        """The text anchors."""
        return np.array([tl.anchor for tl in self._children], dtype=object)

    @anchor.setter
    def anchor(self, values: str | Alignment):
        anchors = _asarray_1d(values, len(self._children), dtype=object)
        for tl, v in zip(self._children, anchors):
            tl.anchor = v

    @property
    def fontfamily(self) -> np.ndarray:
        """The text font families."""
        return np.array([tl.fontfamily for tl in self._children], dtype=object)

    @fontfamily.setter
    def fontfamily(self, values: str):
        ffs = _asarray_1d(values, len(self._children), dtype=object)
        for tl, v in zip(self._children, ffs):
            tl.fontfamily = v

    def setup(
        self,
        *,
        color: ColorType | Sequence[ColorType] | _Void = _void,
        size: float | Sequence[float] | _Void = _void,
        rotation: float | Sequence[float] | _Void = _void,
        anchor: str | Alignment | Sequence[str | Alignment] | _Void = _void,
        fontfamily: str | Sequence[str] | _Void = _void,
    ):
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


def _asarray_1d(x: float, size: int, dtype=None) -> np.ndarray:
    if np.isscalar(x) or isinstance(x, Enum):
        out = np.full((size,), x, dtype=dtype)
    else:
        out = np.asarray(x, dtype=dtype)
        if out.shape != (size,):
            raise ValueError(f"Expected shape ({size},), got {out.shape}")
    return out


def _as_color_array(color, size: int) -> NDArray[np.float32]:
    if isinstance(color, str):  # e.g. color = "black"
        col = norm_color(color)
        return np.repeat(col[np.newaxis, :], size, axis=0)
    if isinstance(color, np.ndarray):
        if color.shape in [(3,), (4,)]:
            col = norm_color(color)
            return np.repeat(col[np.newaxis, :], size, axis=0)
        elif color.shape in [(size, 3), (size, 4)]:
            return color
        else:
            raise ValueError("Color array must have shape (3,), (4,), (N, 3), or (N, 4) " f"but got {color.shape}")
    arr = np.array(color)
    if arr.dtype == object:
        if arr.shape != (size,):
            raise ValueError(f"Expected color array of shape ({size},), got {arr.shape}")
        return np.stack([norm_color(each) for each in arr], axis=0)
    return _as_color_array(arr, size)
