from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from plotly import graph_objects as go

from whitecanvas.backend.plotly._base import (
    PlotlyHoverableLayer,
    from_plotly_linestyle,
    to_plotly_linestyle,
)
from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import arr_color, rgba_str_color


class MonoLine3D(PlotlyHoverableLayer[go.Scatter3d]):
    def __init__(self, xdata, ydata, zdata):
        ndata = len(xdata)
        self._props = {
            "x": xdata,
            "y": ydata,
            "z": zdata,
            "mode": "lines",
            "line": {"color": "blue", "width": 1, "dash": "solid"},
            "type": "scatter3d",
            "showlegend": False,
            "visible": True,
            "customdata": [""] * ndata,
            "hovertemplate": "%{customdata}<extra></extra>",
        }
        PlotlyHoverableLayer.__init__(self)

    def _plt_get_data(self):
        return self._props["x"], self._props["y"], self._props["z"]

    def _plt_set_data(self, xdata, ydata, zdata):
        self._props["x"] = xdata
        self._props["y"] = ydata
        self._props["z"] = zdata

    def _plt_get_ndata(self) -> int:
        return len(self._props["x"])

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
        return False

    def _plt_set_antialias(self, antialias: bool):
        pass
