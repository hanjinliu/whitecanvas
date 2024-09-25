from __future__ import annotations

from typing import Any

import numpy as np

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

    @classmethod
    def from_dict(cls, _: dict[str, Any]) -> NoOffset:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "no"}


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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConstantOffset:
        return cls(d["x"], d["y"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "constant", "x": self._x, "y": self._y}


class CustomOffset(TextOffset):
    def __init__(self, x: Any, y: Any):
        self._x, self._y = x, y

    def _asarray(self) -> tuple[Any, Any]:
        return (self._x, self._y)

    def _add(self, dx, dy) -> TextOffset:
        return CustomOffset(as_array_1d(dx + self._x), as_array_1d(dy + self._y))

    def __repr__(self) -> str:
        return f"<CustomOffset({self._x}, {self._y})>"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CustomOffset:
        return cls(
            np.asarray(d["x"], dtype=np.float32), np.asarray(d["y"], dtype=np.float32)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "custom", "x": self._x, "y": self._y}


def parse_offset_dict(d: dict[str, Any]) -> TextOffset:
    if d["type"] == "no":
        return NoOffset.from_dict(d)
    elif d["type"] == "constant":
        return ConstantOffset.from_dict(d)
    elif d["type"] == "custom":
        return CustomOffset.from_dict(d)
    else:
        raise ValueError(f"Unknown offset type: {d['type']}")
