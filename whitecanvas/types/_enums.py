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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


class LineStyle(_strEnum):
    SOLID = "-"
    DASH = "--"
    DASH_DOT = "-."
    DOT = ":"
    NONE = " "


class FacePattern(_strEnum):
    SOLID = " "
    HORIZONTAL = "-"
    VERTICAL = "|"
    CROSS = "+"
    DIAGONAL_BACK = "/"
    DIAGONAL_FORWARD = "\\"
    DIAGONAL_CROSS = "x"
    DOTS = "."


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


class Modifier(_strEnum):
    SHIFT = "shift"
    CTRL = "ctrl"
    ALT = "alt"
    META = "meta"


class MouseButton(_strEnum):
    NONE = "none"
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"
    BACK = "back"
    FORWARD = "forward"


class MouseEventType(_strEnum):
    MOVE = "move"
    CLICK = "click"
    RELEASE = "release"
    DOUBLE_CLICK = "double_click"
