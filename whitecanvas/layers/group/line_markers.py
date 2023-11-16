from __future__ import annotations

from whitecanvas.types import ColorType, _Void
from whitecanvas.layers.primitive import Line, Markers
from whitecanvas.layers._base import LayerGroup


_void = _Void()


class LineMarkers(LayerGroup):
    def __init__(self, line: Line, markers: Markers, name: str | None = None):
        super().__init__([line, markers], name=name)

    @property
    def line(self) -> Line:
        return self._children[0]

    @property
    def markers(self) -> Markers:
        return self._children[1]

    def setup_line(
        self,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        self.line.setup(color=color, line_width=width, style=style, antialias=antialias)
        return self

    def setup_markers(
        self,
        symbol: str | _Void = _void,
        size: float | _Void = _void,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: str | _Void = _void,
    ):
        if symbol is not _void:
            self.markers.symbol = symbol
        if size is not _void:
            self.markers.size = size
        if face_color is not _void:
            self.markers.face_color = face_color
        if edge_color is not _void:
            self.markers.edge_color = edge_color
        if edge_width is not _void:
            self.markers.edge_width = edge_width
        if edge_style is not _void:
            self.markers.edge_style = edge_style
        return self

    @property
    def data(self):
        return self.line.data

    def set_data(self, xdata=None, ydata=None):
        self.line.set_data(xdata, ydata)
        self.markers.set_data(xdata, ydata)
