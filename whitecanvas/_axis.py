from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, SupportsIndex


class DimAxis(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Name of the axis."""
        return self._name

    @abstractmethod
    def value(self) -> SupportsIndex:
        """Value of the axis."""

    @abstractmethod
    def set_value(self, value: Any):
        """Set the value of the axis."""


class RangeAxis(DimAxis):
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self._size = size
        self._value = 0

    def value(self) -> SupportsIndex:
        return self._value

    def set_value(self, value: Any):
        v = int(value)
        if not 0 <= v < self._size:
            raise ValueError(
                f"Size of axis {self!r} is {self._size} but got index {value!r}."
            )
        self._value = value

    def size(self) -> int:
        return self._size

    def set_size(self, size: int):
        self._size = size
        self._value = min(self._value, size - 1)


class CategoricalAxis(DimAxis):
    def __init__(self, name: str, categories: list[str]):
        super().__init__(name)
        self._mapper = {c: i for i, c in enumerate(categories)}
        self._value = categories[0]

    def value(self) -> SupportsIndex:
        return self._mapper[self._value]

    def set_value(self, value: Any):
        if value not in self._mapper:
            raise ValueError(
                f"Value must be one of {list(self._mapper)} but got {value!r}."
            )
        self._value = value

    def categories(self) -> list[str]:
        return list(self._mapper.keys())
