from typing import Callable, TypeVar

from whitecanvas.protocols.canvas_protocol import (
    AxisProtocol,
    CanvasGridProtocol,
    CanvasProtocol,
    HasVisibility,
    TextLabelProtocol,
    TicksProtocol,
)
from whitecanvas.protocols.layer_protocols import (
    ArrowProtocol,
    BandProtocol,
    BarProtocol,
    BaseProtocol,
    ErrorbarProtocol,
    ImageProtocol,
    LineProtocol,
    MarkersProtocol,
    MultiLineProtocol,
    RangeDataProtocol,
    TextProtocol,
)

__all__ = [
    "ArrowProtocol",
    "BaseProtocol",
    "LineProtocol",
    "MultiLineProtocol",
    "MarkersProtocol",
    "BarProtocol",
    "BandProtocol",
    "ErrorbarProtocol",
    "TextProtocol",
    "ImageProtocol",
    "RangeDataProtocol",
    "CanvasProtocol",
    "CanvasGridProtocol",
    "HasVisibility",
    "TextLabelProtocol",
    "TicksProtocol",
    "AxisProtocol",
]

_C = TypeVar("_C")


def check_protocol(p: "type[BaseProtocol]") -> Callable[[_C], _C]:
    def _inner(c):
        if not isinstance(c, p):
            missing = set(filter(lambda x: x.startswith("_plt"), dir(p))) - set(dir(c))
            raise TypeError(
                f"{c!r} does not implement {p.__name__}. \n"
                f"Missing methods: {missing!r}"
            )
        return c

    return _inner
