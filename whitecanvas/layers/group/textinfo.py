from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from whitecanvas.types import Alignment
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.layers.primitive import Text, Line


def _norm_bracket_data(
    pos0: tuple[float, float], pos1: tuple[float, float], capsize: float
):
    pos0 = np.array(pos0)
    pos1 = np.array(pos1)
    if np.all(pos0 == pos1):
        raise ValueError("pos0 and pos1 must be different")
    posc = (pos0 + pos1) / 2
    cap_vec = (pos1[::-1] - pos0[::-1]) * np.array([1, -1])
    pos0_cap = pos0 + cap_vec * capsize
    pos1_cap = pos1 + cap_vec * capsize
    line_data = np.stack([pos0_cap, pos0, pos1, pos1_cap], axis=0)
    text_pos = posc - cap_vec * capsize
    return line_data, text_pos


class BracketText(ListLayerGroup):
    """
    A group of shaped bracket and text.

    This layer group is useful for such as annotating p-values.

       text
    ┌────────┐
    """

    def __init__(
        self,
        pos0: tuple[float, float],
        pos1: tuple[float, float],
        string: str = "",
        capsize: float = 0.1,
        name: str | None = None,
    ):
        line_data, text_pos = _norm_bracket_data(pos0, pos1, capsize)
        text = Text(*text_pos, string)
        line = Line(
            line_data[:, 0], line_data[:, 1], name="bracket", width=1, color="black"
        )
        super().__init__([text, line], name=name)

    @property
    def text(self) -> Text:
        return self._children[0]

    @property
    def line(self) -> Line:
        return self._children[1]

    @property
    def capsize(self) -> float:
        """Cap size of the bracket."""
        return float(np.sqrt(np.sum((self.line.data[0] - self.line.data[1]) ** 2)))

    @capsize.setter
    def capsize(self, size: float):
        pos0 = self.line.data[1]
        pos1 = self.line.data[2]
        line_data, text_pos = _norm_bracket_data(pos0, pos1, size)
        self.line.set_data(line_data[:, 0], line_data[:, 1])
        self.text.pos = text_pos


class Panel(ListLayerGroup):
    """
    A rectangle titled with a text.

        title
    ┌───────────┐
    │           │
    │           │
    └───────────┘
    """

    def __init__(
        self,
        origin: tuple[float, float],
        width: float,
        height: float,
        *,
        title: str = "",
        name: str | None = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        bl = np.array(origin)
        tl = bl + np.array([0, height])
        br = bl + np.array([width, 0])
        tr = bl + np.array([width, height])
        text_pos = tl + np.array([width / 2, 0])
        text = Text(*text_pos, title, anchor=Alignment.BOTTOM)
        line_data = np.stack([tl, tr, br, bl, tl], axis=0)
        line = Line(
            line_data[:, 0], line_data[:, 1], name="panel", width=1, color="black"
        )
        super().__init__([text, line], name=name)

    @property
    def text(self) -> Text:
        """Text layer of this panel."""
        return self._children[0]

    @property
    def line(self) -> Line:
        """Line layer of this panel."""
        return self._children[1]

    @property
    def top_left(self) -> NDArray[np.floating]:
        """(x, y) of the top left corner of the panel."""
        return self.line.data.stack()[0]

    @property
    def top_right(self) -> NDArray[np.floating]:
        """(x, y) of the top right corner of the panel."""
        return self.line.data.stack()[1]

    @property
    def bottom_right(self) -> NDArray[np.floating]:
        """(x, y) of the bottom right corner of the panel."""
        return self.line.data.stack()[2]

    @property
    def bottom_left(self) -> NDArray[np.floating]:
        """(x, y) of the bottom left corner of the panel."""
        return self.line.data.stack()[3]

    @property
    def center(self) -> tuple[float, float]:
        """(x, y) of the center of the panel."""
        return (self.top_left + self.bottom_right) / 2

    @center.setter
    def center(self, pos: tuple[float, float]):
        dr = np.array(pos) - self.center
        line_data = self.line.data
        self.line.set_data(line_data.x + dr[0], line_data.y + dr[1])
        self.text.pos = self.text.pos + dr

    @property
    def width(self) -> float:
        """Width of the panel."""
        return self.top_right[0] - self.top_left[0]

    @width.setter
    def width(self, width: float):
        line_data = self.line.data
        if width <= 0:
            raise ValueError("width must be positive")
        w = width / 2
        dx = np.array([-w, w, w, -w, -w])
        self.line.set_data(xdata=line_data.x + dx)

    @property
    def height(self) -> float:
        """Height of the panel."""
        return self.bottom_left[1] - self.top_left[1]

    @height.setter
    def height(self, height: float):
        line_data = self.line.data
        if height <= 0:
            raise ValueError("height must be positive")
        h = height / 2
        dy = np.array([h, h, -h, -h, h])
        self.line.set_data(ydata=line_data.y + dy)
        self.text.pos = self.text.pos + np.array([0, h])
