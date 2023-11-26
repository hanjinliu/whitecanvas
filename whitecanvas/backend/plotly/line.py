from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from plotly import graph_objects as go
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import arr_color, rgba_str_color
from ._base import PlotlyLayer, to_plotly_linestyle, from_plotly_linestyle


@check_protocol(LineProtocol)
class MonoLine(PlotlyLayer):
    def __init__(self, xdata, ydata):
        self._props = {
            "x": xdata,
            "y": ydata,
            "mode": "lines",
            "line": {"color": "blue", "width": 1, "dash": "solid", "simplify": False},
            "type": "scatter",
            "visible": True,
        }

    def _plt_get_data(self):
        return self._props["x"], self._props["y"]

    def _plt_set_data(self, xdata, ydata):
        self._props["x"] = xdata
        self._props["y"] = ydata

    def _plt_get_edge_width(self) -> float:
        return self._props["line"]["width"]

    def _plt_set_edge_width(self, width: float):
        self._props["line"]["width"] = width

    def _plt_get_edge_style(self) -> LineStyle:
        return from_plotly_linestyle(self._props["line"]["dash"])

    def _plt_set_edge_style(self, style: LineStyle):
        self._props["line"]["dash"] = to_plotly_linestyle(style)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return arr_color(self._props["line"]["color"])

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._props["line"]["color"] = rgba_str_color(color)

    def _plt_get_antialias(self) -> bool:
        return not self._props["line"]["simplify"]

    def _plt_set_antialias(self, antialias: bool):
        self._props["line"]["simplify"] = not antialias


class MultiLine(PlotlyLayer):
    def __init__(self, data: list[NDArray[np.floating]]):
        # In plotly, we can break a line into multiple segments by inserting None
        xdata = []
        ydata = []
        for each in data:
            each = each.tolist()
            xdata = xdata + each + [None]
            ydata = ydata + each + [None]
        # remove last None
        xdata.pop()
        ydata.pop()
        self._data = data

        self._props = {
            "x": xdata,
            "y": ydata,
            "mode": "lines",
            "line": {"color": "blue", "width": 1, "dash": "solid", "simplify": False},
            "type": "scatter",
            "visible": True,
        }

    def _plt_get_data(self):
        return self._data

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        xdata = []
        ydata = []
        for each in data:
            each = each.tolist()
            xdata = xdata + each + [None]
            ydata = ydata + each + [None]
        # remove last None
        xdata.pop()
        ydata.pop()
        self._data = data
        self._props["x"] = xdata
        self._props["y"] = ydata

    def _plt_get_edge_width(self) -> float:
        return self._props["line"]["width"]

    def _plt_set_edge_width(self, width: float):
        self._props["line"]["width"] = width

    def _plt_get_edge_style(self) -> LineStyle:
        return from_plotly_linestyle(self._props["line"]["dash"])

    def _plt_set_edge_style(self, style: LineStyle):
        self._props["line"]["dash"] = to_plotly_linestyle(style)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return arr_color(self._props["line"]["color"])

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._props["line"]["color"] = rgba_str_color(color)

    def _plt_get_antialias(self) -> bool:
        return not self._props["line"]["simplify"]

    def _plt_set_antialias(self, antialias: bool):
        self._props["line"]["simplify"] = not antialias
