from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers._primitive import Line
from whitecanvas.layers.group._collections import LayerCollectionBase
from whitecanvas.types import LineStyle, XYData
from whitecanvas.utils.normalize import as_any_1d_array, as_color_array


class LineCollection(LayerCollectionBase[Line]):
    """Collection of lines."""

    def __init__(
        self,
        segments: list[Any],
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        lines = [Line([], [], backend=backend) for _ in segments]
        for line, seg in zip(lines, segments):
            line.data = seg
        super().__init__(lines, name=name)

    @property
    def data(self) -> list[XYData]:
        return [line.data for line in self]

    @data.setter
    def data(self, data: list[XYData]):
        ndata_in = len(data)
        ndata_now = len(self)
        if ndata_in > ndata_now:
            for _ in range(ndata_now, ndata_in):
                self.append(Line([], []))
        elif ndata_in < ndata_now:
            for _ in range(ndata_in, ndata_now):
                del self[-1]
        for line, d in zip(self, data):
            line.data = d

    @property
    def width(self) -> NDArray[np.float32]:
        return np.array([line.width for line in self], dtype=np.float32)

    @width.setter
    def width(self, width: float | Sequence[float]):
        if isinstance(width, float):
            _width = [width] * len(self)
        else:
            _width = np.asarray(width, dtype=np.float32)
        if len(width) != len(self):
            raise ValueError(
                f"width must be a float or a sequence of length {len(self)}"
            )
        for line, w in zip(self, _width):
            line.width = w

    @property
    def color(self) -> NDArray[np.float32]:
        return np.array([line.color for line in self], dtype=np.float32)

    @color.setter
    def color(self, color: str | Sequence[str]):
        col = as_color_array(color, len(self))
        for line, c in zip(self, col):
            line.color = c

    @property
    def style(self) -> list[LineStyle]:
        return np.array([line.style for line in self], dtype=np.float32)

    @style.setter
    def style(self, style: str | Sequence[str]):
        styles = as_any_1d_array(style, len(self))
        for line, s in zip(self, styles):
            line.style = s
