from __future__ import annotations

from enum import Enum


class _strEnum(Enum):
    def __eq__(self, other):
        if isinstance(other, str):
            if other in self._value_:
                return self.value == other
            elif (_upper := other.upper()) in self._name_:
                return self.name == _upper
            else:
                return False
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)


class LineStyle(_strEnum):
    SOLID = "-"
    DASH = "--"
    DASH_DOT = "-."
    DOT = ":"
    NONE = " "


class Symbol(_strEnum):
    CIRCLE = "o"
    SQUARE = "s"
    TRIANGLE_UP = "^"
    TRIANGLE_DOWN = "v"
    TRIANGLE_LEFT = "<"
    TRIANGLE_RIGHT = ">"
    DIAMOND = "D"
    CROSS = "x"
    PLUS = "+"
    STAR = "*"
    DOT = "."
    VBAR = "|"
    HBAR = "_"
    NONE = " "
