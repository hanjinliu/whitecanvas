from __future__ import annotations

from dataclasses import dataclass
from ._enums import MouseButton, Modifier, MouseEventType


@dataclass
class MouseEvent:
    button: MouseButton
    modifiers: tuple[Modifier, ...]
    pos: tuple[float, float]
    type: MouseEventType

    def update(self, other: MouseEvent):
        self.button = other.button
        self.modifiers = other.modifiers
        self.pos = other.pos
        self.type = other.type

    def _repr_simple(self):
        modifiers = "+".join(mod.name for mod in self.modifiers)
        return f"button={self.button.name!r}, {modifiers=!r}, pos={self.pos!r}, type={self.type.name!r}"
