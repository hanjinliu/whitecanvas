from __future__ import annotations

from typing import Iterable, Iterator, Hashable
from whitecanvas.layers._base import Layer, LayerGroup


class ListLayerGroup(LayerGroup):
    def __init__(self, children: Iterable[Layer], name: str | None = None):
        super().__init__(name=name)
        self._children = list(children)
        self._ordering_indices = self._default_ordering(len(self._children))
        self._emit_layer_grouped()

    def _default_ordering(self, n: int) -> list[int]:
        """Return the default ordering of the children."""
        return list(range(n))

    def iter_children(self) -> Iterator[Layer]:
        """Recursively iterate over all children."""
        for _, child in sorted(
            zip(self._ordering_indices, self._children), key=lambda x: x[0]
        ):
            yield child

    @property
    def zorders(self) -> list[int]:
        return list(self._ordering_indices)

    @zorders.setter
    def zorders(self, zorders: list[int]):
        zorders = list(zorders)
        if set(zorders) != set(range(len(self._children))):
            raise ValueError(
                "zorders must be a permutation of numbers from 0 to N-1, "
                f"like [0, 2, 1, 3], but got {zorders}"
            )
        self._ordering_indices = zorders


class DictLayerGroup(LayerGroup):
    def __init__(self, children: dict[Hashable, Layer], name: str | None = None):
        super().__init__(name=name)
        self._children = dict(children)
        self._zorders = self._default_ordering(len(self._children))
        self._emit_layer_grouped()

    def iter_children(self) -> Iterator[Layer]:
        """Recursively iterate over all children."""
        yield from self._children.values()

    def _default_ordering(self, n: int) -> list[int]:
        """Return the default ordering of the children."""
        return list(range(n))

    @property
    def zorders(self) -> list[int]:
        return list(self._ordering_indices)

    @zorders.setter
    def zorders(self, zorders: list[int]):
        zorders = list(zorders)
        if set(zorders) != set(range(len(self._children))):
            raise ValueError(
                "zorders must be a permutation of numbers from 0 to N-1, "
                f"like [0, 2, 1, 3], but got {zorders}"
            )
        self._ordering_indices = zorders
