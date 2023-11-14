from __future__ import annotations

from typing import Any, Iterable, overload, TypeVar
from psygnal import Signal, SignalGroup
from psygnal.containers import EventedList

from neoplot.layers import AnyLayer

_V = TypeVar("_V", bound=Any)


class LayerListEvents(SignalGroup):
    inserting = Signal(int)  # idx
    inserted = Signal(int, AnyLayer)  # (idx, value)
    removing = Signal(int)  # idx
    removed = Signal(int, AnyLayer)  # (idx, value)
    moving = Signal(int, int)  # (src_idx, dest_idx)
    moved = Signal(int, int, AnyLayer)  # (src_idx, dest_idx, value)
    changed = Signal(object, AnyLayer, AnyLayer)  # (int | slice, old, new)
    reordered = Signal()
    renamed = Signal(int, str, str)  # (idx, old_name, new_name)


class LayerList(EventedList[AnyLayer]):
    events: LayerListEvents
    _instances: dict[int, LayerList] = {}

    def __init__(self, data: Iterable[AnyLayer] = ()):
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
    def __getitem__(self, idx: int) -> AnyLayer:
        ...

    @overload
    def __getitem__(self, idx: slice) -> LayerList:
        ...

    @overload
    def __getitem__(self, idx: str) -> AnyLayer:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, str):
            for layer in self:
                if layer.name == idx:
                    return layer
            raise KeyError(idx)
        return super().__getitem__(idx)

    def get(self, idx: str, default: _V | None = None) -> AnyLayer | _V | None:
        if isinstance(idx, str):
            for layer in self:
                if layer.name == idx:
                    return layer
            return default
        raise TypeError(f"LayerList.get() expected str, got {type(idx)}")
