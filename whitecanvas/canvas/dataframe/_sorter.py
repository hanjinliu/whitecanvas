from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

_T = TypeVar("_T")


class _SerializableSorter(ABC, Generic[_T]):
    @abstractmethod
    def __call__(self, x: Sequence[_T]) -> Sequence[_T]: ...
    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, str]) -> _SerializableSorter[_T]: ...
    @abstractmethod
    def to_dict(self) -> dict[str, str]: ...


class SimpleSorter(_SerializableSorter[_T]):
    def __init__(self, ascending: bool) -> None:
        self._ascending = ascending

    def __call__(self, x: Sequence[_T]) -> Sequence[_T]:
        return sorted(x, reverse=not self._ascending)

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> SimpleSorter[_T]:
        return cls(d["type"] == "ascending")

    def to_dict(self) -> dict[str, str]:
        return {"type": "ascending" if self._ascending else "descending"}


class ManualSorter(_SerializableSorter[_T]):
    def __init__(self, order: Sequence[_T]) -> None:
        if len(order) != len(set(order)):
            raise ValueError(f"Order contains duplicate values: {order!r}")
        self._order = list(order)

    @classmethod
    def norm_tuple(
        cls,
        order: Sequence[tuple[_T, ...] | _T],
    ) -> ManualSorter[tuple[_T, ...]]:
        order_normed = [u if isinstance(u, tuple) else (u,) for u in order]
        return cls(order_normed)

    def __call__(self, x: Sequence[_T]) -> Sequence[_T]:
        input_categories = set(x)
        for each in self._order:
            if each in input_categories:
                input_categories.discard(each)
        if input_categories:
            raise ValueError(
                f"Categories {input_categories!r} are not in the given order "
                f"{self._order!r}"
            )
        return self._order

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> ManualSorter[_T]:
        return cls(d["order"])

    def to_dict(self) -> dict[str, str]:
        return {"type": "manual", "order": self._order}


def construct_sorter(d: dict[str, str]) -> _SerializableSorter[_T]:
    if d["type"] == "ascending" or d["type"] == "descending":
        return SimpleSorter.from_dict(d)
    elif d["type"] == "manual":
        return ManualSorter.from_dict(d)
    else:
        raise ValueError(f"Unknown sorter type: {d['type']!r}")
