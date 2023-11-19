from __future__ import annotations

from typing import Any, Iterable, overload, TypeVar
from psygnal import Signal, SignalGroup
from psygnal.containers import EventedList

from whitecanvas.layers import Layer, LayerGroup, PrimitiveLayer

_V = TypeVar("_V", bound=Any)


class LayerListEvents(SignalGroup):
    inserting = Signal(int)  # idx
    inserted = Signal(int, Layer)  # (idx, value)
    removing = Signal(int)  # idx
    removed = Signal(int, Layer)  # (idx, value)
    moving = Signal(int, int)  # (src_idx, dest_idx)
    moved = Signal(int, int, Layer)  # (src_idx, dest_idx, value)
    changed = Signal(object, Layer, Layer)  # (int | slice, old, new)
    reordered = Signal()
    renamed = Signal(int, str, str)  # (idx, old_name, new_name)


class LayerList(EventedList[Layer]):
    events: LayerListEvents
    _instances: dict[int, LayerList] = {}

    def __init__(self, data: Iterable[Layer] = ()):
        super().__init__(data, hashable=True, child_events=False)

    def __get__(self, instance, owner) -> LayerList:
        if instance is None:
            return self
        _id = id(instance)
        cls = self.__class__
        if (out := cls._instances.get(_id, None)) is None:
            out = cls._instances[_id] = cls()
        return out

    @overload
    def __getitem__(self, idx: int) -> Layer:
        ...

    @overload
    def __getitem__(self, idx: slice) -> LayerList:
        ...

    @overload
    def __getitem__(self, idx: str) -> Layer:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, str):
            for layer in self:
                if layer.name == idx:
                    return layer
            raise KeyError(idx)
        return super().__getitem__(idx)

    def get(self, idx: str, default: _V | None = None) -> Layer | _V | None:
        if isinstance(idx, str):
            for layer in self:
                if layer.name == idx:
                    return layer
            return default
        raise TypeError(f"LayerList.get() expected str, got {type(idx)}")

    def iter_primitives(self) -> Iterable[PrimitiveLayer]:
        for layer in self:
            if isinstance(layer, LayerGroup):
                yield from layer.iter_children_recursive()
            else:
                yield layer
