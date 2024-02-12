from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from plotly import graph_objects as go

from whitecanvas.backend import _not_implemented
from whitecanvas.backend.plotly._base import (
    PlotlyHoverableLayer,
    from_plotly_marker_symbol,
    to_plotly_marker_symbol,
)
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Symbol
from whitecanvas.utils.normalize import arr_color, as_color_array, rgba_str_color
from whitecanvas.utils.type_check import is_real_number


@check_protocol(MarkersProtocol)
class Markers(PlotlyHoverableLayer[go.Scatter]):
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
            "showlegend": False,
            "visible": True,
            "customdata": [""] * ndata,
            "hovertemplate": "%{customdata}<extra></extra>",
        }
        PlotlyHoverableLayer.__init__(self)

    def _plt_get_ndata(self) -> int:
        return len(self._props["x"])

    def _plt_get_data(self):
        return self._props["x"], self._props["y"]

    def _plt_set_data(self, xdata, ydata):
        self._props["x"] = xdata
        self._props["y"] = ydata

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        color = self._props["marker"]["color"]
        if len(color) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack([arr_color(c) for c in color], axis=0)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, self._plt_get_ndata())
        self._props["marker"]["color"] = [rgba_str_color(c) for c in color]

    _plt_get_face_hatch, _plt_set_face_hatch = _not_implemented.face_patterns()

    def _plt_get_symbol(self) -> Symbol:
        return from_plotly_marker_symbol(self._props["marker"]["symbol"])

    def _plt_set_symbol(self, symbol: Symbol):
        self._props["marker"]["symbol"] = to_plotly_marker_symbol(symbol)

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return np.asarray(self._props["marker"]["size"])

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if is_real_number(size):
            size = np.full(self._plt_get_ndata(), size)
        self._props["marker"]["size"] = size

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.asarray(self._props["marker"]["line"]["width"])

    def _plt_set_edge_width(self, width: float):
        if is_real_number(width):
            width = np.full(self._plt_get_ndata(), width)
        self._props["marker"]["line"]["width"] = width

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_styles()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        color = self._props["marker"]["line"]["color"]
        if len(color) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack([arr_color(c) for c in color], axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, self._plt_get_ndata())
        self._props["marker"]["line"]["color"] = [rgba_str_color(c) for c in color]
