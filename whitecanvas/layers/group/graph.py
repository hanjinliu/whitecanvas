from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from whitecanvas.types import _Void, ColorType, Alignment, XYData
from whitecanvas.layers._primitive import MultiLine, Markers, Texts
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.layers.group._offsets import TextOffset, NoOffset


class Graph(LayerContainer):
    def __init__(
        self,
        nodes: Markers,
        edges: MultiLine,
        texts: Texts,
        edges_data: NDArray[np.intp],
        name: str | None = None,
        offset: TextOffset = NoOffset(),
    ):
        self._edges_data = edges_data
        super().__init__([nodes, edges, texts], name=name)
        self._text_offset = offset

    def _default_ordering(self, n: int) -> list[int]:
        assert n == 3
        return [1, 0, 2]

    @property
    def nodes(self) -> Markers:
        """The nodes layer."""
        return self._children[0]

    @property
    def edges(self) -> MultiLine:
        """The edges layer."""
        return self._children[1]

    @property
    def texts(self) -> Texts:
        """The texts layer."""
        return self._children[2]

    @property
    def edge_indices(self) -> NDArray[np.intp]:
        """Current data of the edges."""
        return self._edges_data

    @property
    def text_offset(self) -> TextOffset:
        """Return the text offset."""
        return self._text_offset

    def add_text_offset(self, dx: Any, dy: Any):
        """Add offset to text positions."""
        _offset = self._text_offset._add(dx, dy)
        if self.texts.ntexts > 0:
            data = self.nodes.data
            xoff, yoff = _offset._asarray()
            self.texts.set_pos(data.x + xoff, data.y + yoff)
        self._text_offset = _offset

    def set_graph(self, nodes: NDArray[np.floating], edges: NDArray[np.intp]):
        """Set the graph data."""
        self.nodes.data = nodes
        self._edges_data = edges

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.CENTER,
        fontfamily: str | None = None,
        offset: tuple[Any, Any] | None = None,
    ):
        if isinstance(strings, str):
            strings = [strings] * self.nodes.data.x.size
        if offset is None:
            _offset = self._text_offset
        else:
            _offset = NoOffset()._add(*offset)

        xdata, ydata = self.nodes.data
        dx, dy = _offset._asarray()
        self.texts.string = strings
        self.texts.set_pos(xdata + dx, ydata + dy)
        self.texts.update(
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            family=fontfamily,
        )
        return self
