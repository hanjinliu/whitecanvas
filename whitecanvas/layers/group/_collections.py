from __future__ import annotations

import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Literal,
    MutableSequence,
    Sequence,
    TypeVar,
    overload,
)

from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers._base import Layer, LayerEvents, LayerGroup, PrimitiveLayer
from whitecanvas.layers._deserialize import construct_layers

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas import Canvas

_L = TypeVar("_L", bound=PrimitiveLayer)


class LayerContainerEvents(LayerEvents):
    reordered = Signal(list)


class RichContainerEvents(LayerContainerEvents):
    face = Signal(object)
    edge = Signal(object)


class LayerContainer(LayerGroup):
    """Layer group that stores children in a list."""

    events: LayerContainerEvents
    _events_class = LayerContainerEvents

    def __init__(
        self,
        children: Layer | Iterable[Layer],
        *more_children,
        name: str | None = None,
    ):
        if isinstance(children, Layer):
            children = [children]
        if more_children:
            children = list(children) + list(more_children)
        super().__init__(name=name)
        self._children = [_process_grouping(c, self) for c in children]
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
        """The z-orders of the children."""
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
        self.events.reordered.emit(zorders)

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a LineStep from a dictionary."""
        children = construct_layers(d["children"], backend=backend)
        return cls(children, name=d["name"])

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "children": [child.to_dict() for child in self._children],
            "name": self.name,
        }

    def _connect_canvas(self, canvas: Canvas):
        # TODO: connect plt.draw
        return super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        return super()._disconnect_canvas(canvas)

    def _as_legend_item(self):
        """Use the first layer as the main legend item."""
        if len(self._children) == 0:
            return _legend.EmptyLegendItem()
        return self._children[0]._as_legend_item()


class LayerTuple(LayerContainer, Sequence[Layer]):
    def __getitem__(self, key: int | str) -> Layer:
        if isinstance(key, str):
            for child in self.iter_children():
                if child.name == key:
                    return child
            raise ValueError(f"Layer {key!r} not found")
        return self._children[key]

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[Layer]:
        return self.iter_children()

    def __repr__(self) -> str:
        cname = type(self).__name__
        return f"{cname}{tuple(self.iter_children())!r}"

    def _as_legend_item(self) -> _legend.LegendItem:
        items = [(layer.name, layer._as_legend_item()) for layer in self]
        return _legend.LegendItemCollection(items)


class LayerCollectionBase(LayerContainer, MutableSequence[_L]):
    _children: list[_L]

    def __getitem__(self, n: int) -> _L:
        """The n-th layer."""
        if not hasattr(n, "__index__"):
            raise TypeError(f"Index must be an integer, not {type(n)}")
        return self._children[n]

    def __setitem__(self, n: int, layer: _L):
        raise NotImplementedError("Cannot set item in a LayerCollection")

    def __delitem__(self, n: int):
        """Delete the n-th layer."""
        if not hasattr(n, "__index__"):
            raise TypeError(f"Index must be an integer, not {type(n)}")
        line = self._children.pop(n)
        if _canvas := self._canvas_ref():
            _canvas._canvas()._plt_remove_layer(line._backend)
            line._disconnect_canvas(_canvas)
        return None

    def __iter__(self) -> Iterator[_L]:
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    def insert(self, n: int, layer: _L):
        if _canvas := self._canvas_ref():
            _canvas._canvas()._plt_add_layer(layer._backend)
            layer._connect_canvas(_canvas)
        _process_grouping(layer, self)
        self._children.insert(n, layer)
        self._ordering_indices.insert(n, len(self._ordering_indices))
        return None

    # fmt: off
    if TYPE_CHECKING:
        def iter_children(self) -> Iterator[_L]: ...
    # fmt: on


_L0 = TypeVar("_L0", bound=Layer)
_L1 = TypeVar("_L1", bound=Layer)


class MainAndOtherLayers(LayerTuple, Generic[_L0, _L1]):
    @overload
    def __getitem__(self, n: Literal[0]) -> _L0: ...

    @overload
    def __getitem__(self, n: Literal[0]) -> _L1: ...

    @overload
    def __getitem__(self, n: int) -> Layer: ...

    def __getitem__(self, n):
        """The n-th layer."""
        return super().__getitem__(n)

    def _insert(self, layer: Layer):
        if layer._canvas_ref() is not None:
            raise ValueError(f"{layer!r} is already added to a canvas")
        if _canvas := self._canvas_ref():
            if isinstance(layer, PrimitiveLayer):
                _canvas._canvas()._plt_add_layer(layer._backend)
            elif isinstance(layer, LayerGroup):
                for l in layer.iter_primitive():
                    _canvas._canvas()._plt_add_layer(l._backend)
            layer._connect_canvas(_canvas)
        _process_grouping(layer, self)
        self._children.insert(1, layer)
        self._ordering_indices.insert(1, len(self._ordering_indices))
        return None

    def _as_legend_item(self):
        """Use the first layer as the main legend item."""
        if len(self) == 0:
            # this should never happen, but just in case
            return _legend.EmptyLegendItem()
        return self[0]._as_legend_item()


def _process_grouping(l, parent: Layer) -> Layer:
    if not isinstance(l, Layer):
        raise TypeError(f"{l!r} is not a Layer")
    if l._group_layer_ref is not None:
        raise ValueError(f"{l!r} is already grouped")
    l._group_layer_ref = weakref.ref(parent)
    return l
