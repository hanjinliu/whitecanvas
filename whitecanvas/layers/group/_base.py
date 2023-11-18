from __future__ import annotations

from typing import Iterable, Iterator
from whitecanvas.layers._base import Layer, LayerGroup, PrimitiveLayer
from whitecanvas.protocols import BaseProtocol


class ListLayerGroup(LayerGroup):
    """
    A group of layers that will be treated as a single layer in the canvas.
    """

    def __init__(self, children: Iterable[Layer], name: str | None = None):
        self._children = list(children)
        self._name = name if name is not None else "LayerGroup"
        self._visible = True
        self._emit_layer_grouped()

    def _iter_children(self) -> Iterator[PrimitiveLayer[BaseProtocol]]:
        """Recursively iterate over all children."""
        for child in self._children:
            if isinstance(child, LayerGroup):
                yield from child._iter_children()
            else:
                yield child
