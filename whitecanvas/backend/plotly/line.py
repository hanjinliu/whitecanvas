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
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import arr_color, rgba_str_color
from whitecanvas.utils.type_check import is_real_number


@check_protocol(LineProtocol)
class MonoLine(PlotlyHoverableLayer[go.Scatter]):
    def __init__(self, xdata, ydata):
        ndata = len(xdata)
        self._props = {
            "x": xdata,
            "y": ydata,
            "mode": "lines",
            "line": {"color": "blue", "width": 1, "dash": "solid", "simplify": False},
            "type": "scatter",
            "showlegend": False,
            "visible": True,
            "customdata": [""] * ndata,
            "hovertemplate": "%{customdata}<extra></extra>",
        }
        PlotlyHoverableLayer.__init__(self)

    def _plt_get_data(self):
        return self._props["x"], self._props["y"]

    def _plt_set_data(self, xdata, ydata):
        self._props["x"] = xdata
        self._props["y"] = ydata

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
        return not self._props["line"]["simplify"]

    def _plt_set_antialias(self, antialias: bool):
        self._props["line"]["simplify"] = not antialias


@check_protocol(MultiLineProtocol)
class MultiLine(PlotlyHoverableLayer[go.Scatter]):
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
            "customdata": [""] * len(xdata),
            "hovertemplate": "%{customdata}<extra></extra>",
        }
        x = np.array(xdata, dtype=np.float32)
        self._nan_indices = np.where(np.isnan(x))[0]
        PlotlyHoverableLayer.__init__(self)

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

    def _plt_get_ndata(self) -> int:
        return len(self._data)

    def _plt_get_edge_width(self) -> NDArray[np.float32]:
        width = self._props["line"]["width"]
        return np.full(self._plt_get_ndata(), width, dtype=np.float32)

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
                _warn_multiple("line widths")
                w = width[0]
        self._props["line"]["width"] = w

    def _plt_get_edge_style(self) -> list[LineStyle]:
        style = from_plotly_linestyle(self._props["line"]["dash"])
        return [style] * self._plt_get_ndata()

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
                _warn_multiple("line styles")
                self._props["line"]["dash"] = to_plotly_linestyle(style[0])

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        col = arr_color(self._props["line"]["color"])
        return np.stack([col] * self._plt_get_ndata(), axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.size == 0:
            return
        if color.ndim == 1:
            self._props["line"]["color"] = rgba_str_color(color)
        else:
            candidates = np.unique(color, axis=0)
            if len(candidates) == 1:
                self._props["line"]["color"] = rgba_str_color(color[0])
            elif len(candidates) == 0:
                self._props["line"]["color"] = "blue"
            else:
                _warn_multiple("line colors")
                self._props["line"]["color"] = rgba_str_color(color[0])

    def _plt_get_antialias(self) -> bool:
        return not self._props["line"]["simplify"]

    def _plt_set_antialias(self, antialias: bool):
        self._props["line"]["simplify"] = not antialias

    def _update_hover_texts(self, fig):
        if self._hover_texts is None:
            return

        dif = np.diff(self._nan_indices, prepend=0, append=len(self._props["x"]))
        # if x = [1, 2, None, 3, 4, 2, None, 3, 2]
        # then dif = [2, 4, 3]
        dif[1:] -= 1
        if len(self._hover_texts) != len(dif):
            warnings.warn(
                "The length of the hover text does not match the number of lines. "
                "Ignoring.",
                UserWarning,
                stacklevel=2,
            )
            return
        customdata = []
        for t, d in zip(self._hover_texts, dif):
            customdata.extend([t] * d)
            customdata.append("")
        customdata.pop()

        fig.update_traces(
            customdata=customdata,
            selector={"uid": self._props["uid"]},
        )

    def _plt_connect_pick_event(self, callback):
        def _new_cb(indices: list[int]):
            return callback([np.where(self._nan_indices < i)[0].size for i in indices])

        return super()._plt_connect_pick_event(_new_cb)


def _warn_multiple(prop: str):
    return warnings.warn(
        f"plotly does not support multiple {prop}. Set to the first one.",
        UserWarning,
        stacklevel=4,
    )
