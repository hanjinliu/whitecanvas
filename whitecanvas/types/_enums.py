from __future__ import annotations

from enum import Enum
from typing import Literal


class _StrEnum(Enum):
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
        return f"{self.__class__.__name__}.{self.name}<{self.value!r}>"


class LineStyle(_StrEnum):
    SOLID = "-"
    DASH = "--"
    DASH_DOT = "-."
    DOT = ":"


class Hatch(_StrEnum):
    SOLID = ""
    HORIZONTAL = "-"
    VERTICAL = "|"
    CROSS = "+"
    DIAGONAL_BACK = "/"
    DIAGONAL_FORWARD = "\\"
    DIAGONAL_CROSS = "x"
    DOTS = "."


class Symbol(_StrEnum):
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


class Modifier(_StrEnum):
    SHIFT = "shift"
    CTRL = "ctrl"
    ALT = "alt"
    META = "meta"


class MouseButton(_StrEnum):
    NONE = "none"
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"
    BACK = "back"
    FORWARD = "forward"


class MouseEventType(_StrEnum):
    MOVE = "move"
    CLICK = "click"
    RELEASE = "release"
    DOUBLE_CLICK = "double_click"


class Alignment(_StrEnum):
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


class LegendLocation(_StrEnum):
    TOP_CENTER = "top_center"
    BOTTOM_CENTER = "bottom_center"
    CENTER_LEFT = "center_left"
    CENTER_RIGHT = "center_right"
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    LEFT_SIDE_TOP = "left_side_top"
    LEFT_SIDE_CENTER = "left_side_center"
    LEFT_SIDE_BOTTOM = "left_side_bottom"
    RIGHT_SIDE_TOP = "right_side_top"
    RIGHT_SIDE_CENTER = "right_side_center"
    RIGHT_SIDE_BOTTOM = "right_side_bottom"
    TOP_SIDE_LEFT = "top_side_left"
    TOP_SIDE_CENTER = "top_side_center"
    TOP_SIDE_RIGHT = "top_side_right"
    BOTTOM_SIDE_LEFT = "bottom_side_left"
    BOTTOM_SIDE_CENTER = "bottom_side_center"
    BOTTOM_SIDE_RIGHT = "bottom_side_right"

    @property
    def is_side(self) -> bool:
        """True if the location is in any of the side locations."""
        return "SIDE" in self.name


LegendLocationStr = Literal[
    "top_center", "bottom_center", "center_left", "center_right", "center", "top_left",
    "top_right", "bottom_left", "bottom_right", "left_side_top", "left_side_center",
    "left_side_bottom", "right_side_top", "right_side_center", "right_side_bottom",
    "top_side_left", "top_side_center", "top_side_right", "bottom_side_left",
    "bottom_side_center", "bottom_side_right",
]  # fmt: skip


class Orientation(_StrEnum):
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

    def transpose(self) -> Orientation:
        """Transpose the orientation."""
        if self is self.VERTICAL:
            return self.HORIZONTAL
        else:
            return self.VERTICAL

    @property
    def is_vertical(self):
        return self is Orientation.VERTICAL

    @property
    def is_horizontal(self):
        return self is Orientation.HORIZONTAL


class Origin(_StrEnum):
    """
    Enum that define the center of image.

    1-----+  0: corner
    |     |  1: edge
    |  0  |  2: center (center of image)
    |     |
    +-----+
    """

    CORNER = "corner"
    EDGE = "edge"
    CENTER = "center"


class HistogramShape(_StrEnum):
    step = "step"
    polygon = "polygon"
    bars = "bars"


class HistogramKind(_StrEnum):
    count = "count"
    density = "density"
    probability = "probability"
    frequency = "frequency"
    percent = "percent"
