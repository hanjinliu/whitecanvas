from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from plotly import graph_objects as go
from whitecanvas.protocols import MarkersProtocol, HeteroMarkersProtocol, check_protocol
from whitecanvas.types import LineStyle, Symbol, FacePattern
from whitecanvas.utils.normalize import arr_color, as_color_array, rgba_str_color
from ._base import (
    PlotlyLayer,
    to_plotly_linestyle,
    from_plotly_linestyle,
    to_plotly_marker_symbol,
    from_plotly_marker_symbol,
)


@check_protocol(MarkersProtocol)
class Markers(PlotlyLayer):
    def __init__(self, xdata, ydata):
        self._props = {
            "x": xdata,
            "y": ydata,
            "mode": "markers",
            "marker": {
                "color": "blue",
                "size": 10,
                "symbol": "circle",
                "line": {"width": 1, "color": "blue"},
            },
            "type": "scatter",
            "visible": True,
        }

    def _plt_get_data(self):
        return self._props["x"], self._props["y"]

    def _plt_set_data(self, xdata, ydata):
        self._props["x"] = xdata
        self._props["y"] = ydata

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return arr_color(self._props["marker"]["color"])

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self._props["marker"]["color"] = rgba_str_color(color)

    def _plt_get_face_pattern(self) -> FacePattern:
        return FacePattern.SOLID

    def _plt_set_face_pattern(self, pattern: FacePattern):
        pass

    def _plt_get_symbol(self) -> Symbol:
        return from_plotly_marker_symbol(self._props["marker"]["symbol"])

    def _plt_set_symbol(self, symbol: Symbol):
        self._props["marker"]["symbol"] = to_plotly_marker_symbol(symbol)

    def _plt_get_symbol_size(self) -> float:
        return self._props["marker"]["size"]

    def _plt_set_symbol_size(self, size: float):
        self._props["marker"]["size"] = size

    def _plt_get_edge_width(self) -> float:
        return self._props["marker"]["line"]["width"]

    def _plt_set_edge_width(self, width: float):
        self._props["marker"]["line"]["width"] = width

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle.SOLID

    def _plt_set_edge_style(self, style: LineStyle):
        pass

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return arr_color(self._props["marker"]["line"]["color"])

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._props["marker"]["line"]["color"] = rgba_str_color(color)


@check_protocol(HeteroMarkersProtocol)
class HeteroMarkers(PlotlyLayer):
    def __init__(self, xdata, ydata):
        ndata = len(xdata)
        self._props = {
            "x": xdata,
            "y": ydata,
            "mode": "markers",
            "marker": {
                "color": ["blue"] * ndata,
                "size": np.full(ndata, 10),
                "symbol": "circle",
                "line": {"width": np.ones(ndata), "color": ["blue"] * ndata},
            },
            "type": "scatter",
            "visible": True,
        }

    def _ndata(self) -> int:
        return len(self._props["x"])

    def _plt_get_data(self):
        return self._props["x"], self._props["y"]

    def _plt_set_data(self, xdata, ydata):
        self._props["x"] = xdata
        self._props["y"] = ydata

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return np.stack(arr_color(c) for c in self._props["marker"]["color"])

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color)
        self._props["marker"]["color"] = [rgba_str_color(c) for c in color]

    def _plt_get_face_pattern(self) -> list[FacePattern]:
        return [FacePattern.SOLID] * len(self._props["marker"]["color"])

    def _plt_set_face_pattern(self, pattern: FacePattern):
        pass

    def _plt_get_symbol(self) -> Symbol:
        return from_plotly_marker_symbol(self._props["marker"]["symbol"])

    def _plt_set_symbol(self, symbol: Symbol):
        self._props["marker"]["symbol"] = to_plotly_marker_symbol(symbol)

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return np.asarray(self._props["marker"]["size"])

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if isinstance(size, float):
            size = np.full(self._ndata(), size)
        self._props["marker"]["size"] = size

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.asarray(self._props["marker"]["line"]["width"])

    def _plt_set_edge_width(self, width: float):
        if isinstance(width, (int, float, np.number)):
            width = np.full(self._ndata(), width)
        self._props["marker"]["line"]["width"] = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [LineStyle.SOLID] * self._ndata()

    def _plt_set_edge_style(self, style: LineStyle):
        pass

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.stack(arr_color(c) for c in self._props["marker"]["line"]["color"])

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color)
        self._props["marker"]["line"]["color"] = [rgba_str_color(c) for c in color]
