from typing_extensions import TypedDict, NotRequired

class LineDict(TypedDict):
    """Line style dictionary"""
    width: NotRequired[float]
    style: NotRequired[str]
    color: NotRequired[str]


class MarkerDict(TypedDict):
    """Marker style dictionary"""
    symbol: NotRequired[str]
    size: NotRequired[float]
    face_color: NotRequired[str]
    edge_color: NotRequired[str]
