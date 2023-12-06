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


class FacePattern(_strEnum):
    SOLID = ""
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
    VBAR = "|"
    HBAR = "_"

    def has_face(self) -> bool:
        return self not in (
            Symbol.CROSS,
            Symbol.PLUS,
            Symbol.VBAR,
            Symbol.HBAR,
        )


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


class Alignment(_strEnum):
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"

    @classmethod
    def merge(cls, vertical, horizontal: Alignment) -> Alignment:
        if vertical not in (Alignment.TOP, Alignment.BOTTOM, Alignment.CENTER):
            raise ValueError(f"{vertical} is not a vertical alignment")
        if horizontal not in (Alignment.LEFT, Alignment.RIGHT, Alignment.CENTER):
            raise ValueError(f"{horizontal} is not a horizontal alignment")
        if vertical is Alignment.TOP:
            if horizontal is Alignment.LEFT:
                return Alignment.TOP_LEFT
            elif horizontal is Alignment.RIGHT:
                return Alignment.TOP_RIGHT
            elif horizontal is Alignment.CENTER:
                return Alignment.TOP
            else:
                raise RuntimeError  # unreachable
        elif vertical is Alignment.BOTTOM:
            if horizontal is Alignment.LEFT:
                return Alignment.BOTTOM_LEFT
            elif horizontal is Alignment.RIGHT:
                return Alignment.BOTTOM_RIGHT
            elif horizontal is Alignment.CENTER:
                return Alignment.BOTTOM
            else:
                raise RuntimeError  # unreachable
        elif vertical is Alignment.CENTER:
            return horizontal
        else:
            raise RuntimeError  # unreachable

    def split(self) -> tuple[Alignment, Alignment]:
        """Split the alignment into vertical and horizontal components."""
        name = self.name
        if "TOP" in name:
            vertical = Alignment.TOP
        elif "BOTTOM" in name:
            vertical = Alignment.BOTTOM
        else:
            vertical = Alignment.CENTER
        if "LEFT" in name:
            horizontal = Alignment.LEFT
        elif "RIGHT" in name:
            horizontal = Alignment.RIGHT
        else:
            horizontal = Alignment.CENTER
        return vertical, horizontal


class Orientation(_strEnum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"

    @classmethod
    def parse(cls, value):
        if isinstance(value, str):
            if value == "v":
                return cls.VERTICAL
            elif value == "h":
                return cls.HORIZONTAL
        return cls(value)

    def transpose(self):
        if self is self.VERTICAL:
            return self.HORIZONTAL
        else:
            return self.VERTICAL

    @property
    def is_vertical(self):
        return self is Orientation.VERTICAL
