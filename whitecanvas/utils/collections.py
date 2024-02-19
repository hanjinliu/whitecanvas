from __future__ import annotations

from typing import Iterable, Iterator, MutableSet, TypeVar

_V = TypeVar("_V")


class OrderedSet(MutableSet[_V]):
    """Implementation of an ordered set using a dict."""

    def __init__(self, values: Iterable[_V] = ()):
        self._dict = dict.fromkeys(values, None)  # NOTE: python dict is ordered

    def __contains__(self, key: _V):
        return key in self._dict

    def __iter__(self) -> Iterator[_V]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def add(self, key: _V):
        self._dict[key] = None

    def discard(self, key: _V):
        self._dict.pop(key, None)
