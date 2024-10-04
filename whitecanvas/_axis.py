from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, SupportsIndex, TypeVar, Union

_T = TypeVar("_T")


class DimAxis(ABC, Generic[_T]):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Name of the axis."""
        return self._name

    @abstractmethod
    def current_index(self) -> SupportsIndex:
        """Current int index of the axis."""

    @abstractmethod
    def current_value(self) -> _T:
        """Current value of the axis."""

    @abstractmethod
    def set_index(self, idx: int) -> None:
        """Set current index to the axis."""

    @abstractmethod
    def set_value(self, value: _T):
        """Set current value of the axis."""


class RangeAxis(DimAxis[int]):
    """An axis with a sequential coordinate."""

    def __init__(self, name: str, size: int):
        super().__init__(name)
        self._size = size
        self._value = 0

    def __repr__(self) -> str:
        return f"RangeAxis(name={self.name}, size={self.size})"

    def current_index(self) -> SupportsIndex:
        return self._value

    def current_value(self) -> Any:
        return self._value

    def set_index(self, idx: int):
        if not 0 <= idx < self._size:
            raise ValueError(f"Size of axis {self!r} is {self._size} but got {idx!r}.")
        self._value = idx

    def set_value(self, value: int):
        return self.set_index(int(value))

    @property
    def size(self) -> int:
        return self._size

    def set_size(self, size: int):
        self._size = size
        self._value = min(self._value, size - 1)


Category = tuple[Union[int, str], ...]


class CategoricalAxis(DimAxis[Category]):
    """An axis with a categorical coordinate."""

    def __init__(self, name: str, categories: list[Category]):
        super().__init__(name)
        self._categories = list(categories)
        self._mapper = {c: i for i, c in enumerate(categories)}
        self._value = self._categories[0]

    def __repr__(self) -> str:
        return f"CategoricalAxis(name={self.name}, categories={self.categories})"

    def current_index(self) -> SupportsIndex:
        return self._mapper[self._value]

    def current_value(self) -> Category:
        return self._value

    def set_index(self, idx: int) -> None:
        return self.set_value(self._categories[idx])

    def set_value(self, value: Category):
        if value not in self._mapper:
            raise ValueError(
                f"Value must be one of {list(self._mapper)} but got {value!r}."
            )
        self._value = value

    @property
    def categories(self) -> list[Category]:
        """List of categories of this axis."""
        return self._categories.copy()
