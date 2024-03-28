from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from whitecanvas.types._enums import Modifier, MouseButton, MouseEventType


class Point(NamedTuple):
    x: float
    y: float


@dataclass
class MouseEvent:
    button: MouseButton
    modifiers: tuple[Modifier, ...]
    pos: Point
    type: MouseEventType

    def __post_init__(self):
        self.pos = Point(*self.pos)  # normalize

    def update(self, other: MouseEvent):
        self.button = other.button
        self.modifiers = other.modifiers
        self.pos = other.pos
        self.type = other.type

    def _repr_simple(self):
        modifiers = "+".join(mod.name for mod in self.modifiers)
        return (
            f"button={self.button.name!r}, "
            f"{modifiers=!r}, pos={self.pos!r}, type={self.type.name!r}"
        )
