from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator, MutableSequence, Sequence, TypeVar

from psygnal import Signal

from whitecanvas.layers._base import Layer, LayerEvents, LayerGroup, PrimitiveLayer

if TYPE_CHECKING:
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

    def __init__(self, children: Iterable[Layer], name: str | None = None):
        super().__init__(name=name)
        self._children = [_check_layer(c) for c in children]
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

    def _connect_canvas(self, canvas: Canvas):
        # TODO: connect plt.draw
        return super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        return super()._disconnect_canvas(canvas)


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


class LayerCollectionBase(LayerContainer, MutableSequence[_L]):
    _children: list[_L]

    def __getitem__(self, n: int) -> _L:
        """The n-th markers layer."""
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
        self._children.insert(n, layer)
        self._ordering_indices.insert(n, len(self._ordering_indices))
        return None

    if TYPE_CHECKING:

        def iter_children(self) -> Iterator[_L]:
            ...


def _check_layer(l) -> Layer:
    if not isinstance(l, Layer):
        raise TypeError(f"{l!r} is not a Layer")
    if l._is_grouped:
        raise ValueError(f"{l!r} is already grouped")
    l._is_grouped = True
    return l
