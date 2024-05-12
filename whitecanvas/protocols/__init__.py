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
    BandProtocol,
    BarProtocol,
    BaseProtocol,
    ErrorbarProtocol,
    ImageProtocol,
    LineProtocol,
    MarkersProtocol,
    MeshProtocol,
    MultiLineProtocol,
    RangeDataProtocol,
    TextProtocol,
    VectorsProtocol,
)

__all__ = [
    "VectorsProtocol",
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
    "MeshProtocol",
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
