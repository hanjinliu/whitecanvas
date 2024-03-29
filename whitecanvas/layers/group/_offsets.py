from __future__ import annotations

from typing import Any

from whitecanvas.utils.normalize import as_array_1d
from whitecanvas.utils.type_check import is_real_number


class TextOffset:
    def _asarray(self) -> tuple[Any, Any]:
        raise NotImplementedError

    def _add(self, dx, dy) -> TextOffset:
        raise NotImplementedError


class NoOffset(TextOffset):
    def _asarray(self):
        return (0, 0)

    def _add(self, dx, dy) -> TextOffset:
        if is_real_number(dx) and is_real_number(dy):
            if dx == 0 and dy == 0:
                return self
            return ConstantOffset(dx, dy)
        else:
            return CustomOffset(as_array_1d(dx), as_array_1d(dy))

    def __repr__(self) -> str:
        return "<NoOffset>"


class ConstantOffset(TextOffset):
    def __init__(self, x: float, y: float):
        self._x, self._y = x, y

    def _asarray(self) -> tuple[float, float]:
        return (self._x, self._y)

    def _add(self, dx, dy) -> TextOffset:
        if is_real_number(dx) and is_real_number(dy):
            return ConstantOffset(self._x + dx, self._y + dy)
        else:
            return CustomOffset(as_array_1d(dx) + self._x, as_array_1d(dy) + self._y)

    def __repr__(self) -> str:
        return f"<ConstantOffset({self._x}, {self._y})>"


class CustomOffset(TextOffset):
    def __init__(self, x: Any, y: Any):
        self._x, self._y = x, y

    def _asarray(self) -> tuple[Any, Any]:
        return (self._x, self._y)

    def _add(self, dx, dy) -> TextOffset:
        return CustomOffset(as_array_1d(dx + self._x), as_array_1d(dy + self._y))

    def __repr__(self) -> str:
        return f"<CustomOffset({self._x}, {self._y})>"
