from typing import TypeVar, Callable
from .layer_protocols import (
    BaseProtocol,
    LineProtocol,
    MultiLineProtocol,
    MarkersProtocol,
    BarProtocol,
    RangeDataProtocol,
    BandProtocol,
    ErrorbarProtocol,
    TextProtocol,
    ImageProtocol,
)
from .canvas_protocol import (
    CanvasProtocol,
    CanvasGridProtocol,
    HasVisibility,
    TextLabelProtocol,
    AxisProtocol,
)

__all__ = [
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
