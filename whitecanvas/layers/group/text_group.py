from __future__ import annotations
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray, ArrayLike
from whitecanvas.types import ColorType, _Void, Alignment
from whitecanvas.layers.primitive import Text
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.backend import Backend
from whitecanvas.utils.normalize import (
    normalize_xy,
    as_any_1d_array,
    as_color_array,
)
from whitecanvas.theme import get_theme

_void = _Void()


class TextGroup(ListLayerGroup):
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
        color: ColorType | None = None,
        size: float | None = None,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
        backend: Backend | str | None = None,
    ) -> TextGroup:
        xs, ys = normalize_xy(xs, ys)
        ndata = xs.size
        if len(strings) != ndata:
            raise ValueError(f"Expected {ndata} texts, got {len(strings)}")
        theme = get_theme()
        if color is None:
            color = theme.foreground_color
        if fontfamily is None:
            fontfamily = theme.fontfamily
        if size is None:
            size = theme.fontsize
        _zip = zip(
            xs,
            ys,
            strings,
            as_color_array(color, ndata),
            as_any_1d_array(size, ndata, dtype=np.float32),
            as_any_1d_array(rotation, ndata, dtype=np.float32),
            as_any_1d_array(anchor, ndata, dtype=object),
            as_any_1d_array(fontfamily, ndata, dtype=object),
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
        return self._children[int(n)]

    @property
    def ntexts(self) -> int:
        """The number of text layers."""
        return len(self._children)

    @property
    def pos(self) -> np.ndarray:
        """The (x, y) positions of the child text layers."""
        return np.array([tl.pos for tl in self._children], dtype=np.float64)

    def set_pos(self, xs: ArrayLike | None, ys: ArrayLike | None):
        """Set the (x, y) positions of the child text layers."""
        # normalize input
        if xs is None and ys is None:
            raise ValueError("At least one of xs or ys must be specified")
        if xs is not None:
            xs = np.asarray(xs)
            if xs.shape != (len(self._children),):
                raise ValueError(
                    f"Expected shape ({len(self._children)},), got {xs.shape}"
                )
        if ys is not None:
            ys = np.asarray(ys)
            if ys.shape != (len(self._children),):
                raise ValueError(
                    f"Expected shape ({len(self._children)},), got {ys.shape}"
                )
        # update positions
        if xs is None and ys is not None:
            for tl, y in zip(self._children, ys):
                tl.pos = (tl.pos[0], y)
        elif xs is not None and ys is None:
            for tl, x in zip(self._children, xs):
                tl.pos = (x, tl.pos[1])
        elif xs is not None and ys is not None:
            for tl, x, y in zip(self._children, xs, ys):
                tl.pos = (x, y)
        else:
            raise RuntimeError  # unreachable

    @property
    def string(self) -> list[str]:
        """The text values."""
        return np.array([tl.string for tl in self._children], dtype=object)

    @string.setter
    def string(self, values: list[str]):
        if len(values) != len(self._children):
            raise ValueError(
                f"Expected {len(self._children)} values, got {len(values)}"
            )
        for tl, v in zip(self._children, values):
            tl.string = v

    @property
    def color(self) -> np.ndarray:
        """The text colors."""
        return np.stack([tl.color for tl in self._children], axis=0)

    @color.setter
    def color(self, colors: ColorType | None):
        if colors is None:
            colors = get_theme().foreground_color
        values = as_color_array(colors, len(self._children))
        for tl, v in zip(self._children, values):
            tl.color = v

    @property
    def size(self) -> np.ndarray:
        """The text sizes."""
        return np.array([tl.size for tl in self._children], dtype=np.float32)

    @size.setter
    def size(self, values: float | Iterable[float] | None):
        if values is None:
            values = get_theme().fontsize
        sizes = as_any_1d_array(values, len(self._children), dtype=np.float32)
        for tl, v in zip(self._children, sizes):
            tl.size = v

    @property
    def rotation(self) -> np.ndarray:
        """The text rotations."""
        return np.array([tl.rotation for tl in self._children], dtype=np.float32)

    @rotation.setter
    def rotation(self, values: float):
        rots = as_any_1d_array(values, len(self._children), dtype=np.float32)
        for tl, v in zip(self._children, rots):
            tl.rotation = v

    @property
    def anchor(self) -> np.ndarray:
        """The text anchors."""
        return np.array([tl.anchor for tl in self._children], dtype=object)

    @anchor.setter
    def anchor(self, values: str | Alignment):
        anchors = as_any_1d_array(values, len(self._children), dtype=object)
        for tl, v in zip(self._children, anchors):
            tl.anchor = v

    @property
    def fontfamily(self) -> np.ndarray:
        """The text font families."""
        return np.array([tl.fontfamily for tl in self._children], dtype=object)

    @fontfamily.setter
    def fontfamily(self, values: str | Iterable[str] | None):
        if values is None:
            values = get_theme().fontfamily
        ffs = as_any_1d_array(values, len(self._children), dtype=object)
        for tl, v in zip(self._children, ffs):
            tl.fontfamily = v

    def update(
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
