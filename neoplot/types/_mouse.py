from __future__ import annotations

from dataclasses import dataclass
from ._enums import MouseButton, Modifier, MouseEventType


@dataclass
class MouseEvent:
    button: MouseButton
    modifiers: Modifier
    pos: tuple[float, float]
    type: MouseEventType
