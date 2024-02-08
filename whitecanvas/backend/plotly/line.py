from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend.plotly._base import (
    PlotlyLayer,
    from_plotly_linestyle,
    to_plotly_linestyle,
)
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import arr_color, rgba_str_color
from whitecanvas.utils.type_check import is_real_number


@check_protocol(LineProtocol)
class MonoLine(PlotlyLayer):
    def __init__(self, xdata, ydata):
        self._props = {
            "x": xdata,
            "y": ydata,
            "mode": "lines",
            "line": {"color": "blue", "width": 1, "dash": "solid", "simplify": False},
            "type": "scatter",
            "showlegend": False,
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


@check_protocol(MultiLineProtocol)
class MultiLine(PlotlyLayer):
    def __init__(self, data: list[NDArray[np.floating]]):
        # In plotly, we can break a line into multiple segments by inserting None
        xdata = []
        ydata = []
        for each in data:
            xdata = xdata + each[:, 0].tolist() + [None]
            ydata = ydata + each[:, 1].tolist() + [None]
        # remove last None
        if len(xdata) > 0:
            xdata.pop()
            ydata.pop()
        self._data = data

        self._props = {
            "x": xdata,
            "y": ydata,
            "mode": "lines",
            "line": {
                "color": "blue",
                "width": 1,
                "dash": "solid",
                "simplify": False,
            },
            "type": "scatter",
            "showlegend": False,
            "visible": True,
        }

    def _plt_get_data(self):
        return self._data

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        xdata = []
        ydata = []
        for each in data:
            xdata = xdata + each[:, 0].tolist() + [None]
            ydata = ydata + each[:, 1].tolist() + [None]
        # remove last None
        if len(xdata) > 0:
            xdata.pop()
            ydata.pop()
        self._data = data
        self._props["x"] = xdata
        self._props["y"] = ydata

    def _plt_get_edge_width(self) -> NDArray[np.float32]:
        width = self._props["line"]["width"]
        return np.full(len(self._data), width, dtype=np.float32)

    def _plt_set_edge_width(self, width):
        if is_real_number(width):
            w = width
        else:
            candidates = np.unique(width)
            if len(candidates) == 1:
                w = candidates[0]
            elif len(candidates) == 0:
                w = 0.0
            else:
                warnings.warn(
                    "plotly does not support multiple line widths. "
                    "Set to the first one.",
                    UserWarning,
                    stacklevel=2,
                )
                w = width[0]
        self._props["line"]["width"] = w

    def _plt_get_edge_style(self) -> list[LineStyle]:
        style = from_plotly_linestyle(self._props["line"]["dash"])
        return [style] * len(self._data)

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            self._props["line"]["dash"] = to_plotly_linestyle(style)
        else:
            candidates = set(style)
            if len(candidates) == 1:
                self._props["line"]["dash"] = to_plotly_linestyle(style[0])
            elif len(candidates) == 0:
                self._props["line"]["dash"] = "solid"
            else:
                warnings.warn(
                    "plotly does not support multiple line styles. "
                    "Set to the first one.",
                    UserWarning,
                    stacklevel=2,
                )
                self._props["line"]["dash"] = to_plotly_linestyle(style[0])

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        col = arr_color(self._props["line"]["color"])
        return np.stack([col] * len(self._data), axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            self._props["line"]["color"] = rgba_str_color(color)
        else:
            candidates = np.unique(color, axis=0)
            if len(candidates) == 1:
                self._props["line"]["color"] = rgba_str_color(color[0])
            elif len(candidates) == 0:
                self._props["line"]["color"] = "blue"
            else:
                warnings.warn(
                    "plotly does not support multiple line colors. "
                    "Set to the first one.",
                    UserWarning,
                    stacklevel=2,
                )
                self._props["line"]["color"] = rgba_str_color(color[0])

    def _plt_get_antialias(self) -> bool:
        return not self._props["line"]["simplify"]

    def _plt_set_antialias(self, antialias: bool):
        self._props["line"]["simplify"] = not antialias
