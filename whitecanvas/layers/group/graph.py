from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from whitecanvas.layers import _legend, _text_utils
from whitecanvas.layers._primitive import Markers, MultiLine, Texts
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.layers.group._offsets import NoOffset, TextOffset
from whitecanvas.types import Alignment, ColorType


class Graph(LayerContainer):
    def __init__(
        self,
        nodes: Markers,
        edges: MultiLine,
        texts: Texts,
        name: str | None = None,
        offset: TextOffset | None = None,
    ):
        if offset is None:
            offset = NoOffset()
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
    def text_offset(self) -> TextOffset:
        """Return the text offset."""
        return self._text_offset

    def with_text_offset(self, dx: Any, dy: Any):
        """Add offset to text positions."""
        _offset = self._text_offset._add(dx, dy)
        if self.texts.ndata > 0:
            data = self.nodes.data
            xoff, yoff = _offset._asarray()
            self.texts.set_pos(data.x + xoff, data.y + yoff)
        self._text_offset = _offset
        return self

    def set_graph(self, nodes: NDArray[np.floating], edges: NDArray[np.intp]):
        """Set the graph data."""
        self.nodes.data = nodes
        edge_data: list[np.ndarray] = []
        for edge in edges:
            edge_data.append(np.stack([nodes[i] for i in edge], axis=0))
        self.edges.data = edge_data

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

        strings = _text_utils.norm_label_text(strings, self.nodes.data)
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

    def _as_legend_item(self):
        line = self.edges._as_legend_item()
        markers = self.nodes._as_legend_item()
        return _legend.PlotLegendItem(line, markers)
