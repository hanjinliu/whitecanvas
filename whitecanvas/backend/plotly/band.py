from __future__ import annotations

import numpy as np
from plotly import graph_objects as go

from whitecanvas.backend._not_implemented import face_hatch
from whitecanvas.backend.plotly._base import (
    PlotlyHoverableLayer,
    from_plotly_linestyle,
    to_plotly_linestyle,
)
from whitecanvas.protocols import BandProtocol, check_protocol
from whitecanvas.types import LineStyle, Orientation
from whitecanvas.utils.normalize import arr_color, rgba_str_color

# NOTE: plotly does not support hover text on the fill area.
# see https://github.com/plotly/plotly.py/issues/2399


@check_protocol(BandProtocol)
class Band(PlotlyHoverableLayer[go.Scatter]):
    def __init__(
        self,
        t: np.ndarray,
        ydata0: np.ndarray,
        ydata1: np.ndarray,
        orient: Orientation,
    ):
        if orient.is_vertical:
            x = np.concatenate([t, t[::-1]])
            y = np.concatenate([ydata0, ydata1[::-1]])
        else:
            x = np.concatenate([ydata0, ydata1[::-1]])
            y = np.concatenate([t, t[::-1]])
        self._props = {
            "x": x,
            "y": y,
            "mode": "lines",
            "fill": "toself",
            "fillcolor": "blue",
            # "hoverinfo": "skip", ... this also disables clicked event.
            "type": "scatter",
            "line": {"color": "blue", "width": 1, "dash": "solid", "simplify": False},
            "customdata": [""] * t.size,
            "hovertemplate": "%{customdata}<extra></extra>",
            "showlegend": False,
            "visible": True,
        }
        PlotlyHoverableLayer.__init__(self)

    ##### XYYDataProtocol #####
    def _plt_get_vertical_data(self):
        x = self._props["x"]
        y = self._props["y"]
        nx = len(x) // 2
        ny = len(y) // 2
        return x[:nx], y[:ny], y[ny:][::-1]

    def _plt_get_horizontal_data(self):
        x = self._props["x"]
        y = self._props["y"]
        nx = len(x) // 2
        ny = len(y) // 2
        return y[:ny], x[:nx], x[nx:][::-1]

    def _plt_set_vertical_data(self, t, ydata0, ydata1):
        x = np.concatenate([t, t[::-1]])
        y = np.concatenate([ydata0, ydata1[::-1]])
        self._props["x"] = x
        self._props["y"] = y

    def _plt_set_horizontal_data(self, t, ydata0, ydata1):
        x = np.concatenate([ydata0, ydata1[::-1]])
        y = np.concatenate([t, t[::-1]])
        self._props["x"] = x
        self._props["y"] = y

    def _plt_get_face_color(self):
        return arr_color(self._props["fillcolor"])

    def _plt_set_face_color(self, color):
        self._props["fillcolor"] = rgba_str_color(color)

    _plt_get_face_hatch, _plt_set_face_hatch = face_hatch()

    def _plt_get_edge_color(self):
        return arr_color(self._props["line"]["color"])

    def _plt_set_edge_color(self, color):
        self._props["line"]["color"] = rgba_str_color(color)

    def _plt_get_edge_width(self):
        return self._props["line"]["width"]

    def _plt_set_edge_width(self, width: float):
        self._props["line"]["width"] = width

    def _plt_get_edge_style(self):
        return from_plotly_linestyle(self._props["line"]["dash"])

    def _plt_set_edge_style(self, style: LineStyle):
        self._props["line"]["dash"] = to_plotly_linestyle(style)

    def _plt_get_antialias(self) -> bool:
        return not self._props["line"]["simplify"]

    def _plt_set_antialias(self, antialias: bool):
        self._props["line"]["simplify"] = not antialias

    def _plt_set_hover_text(self, text: str):
        self._hover_texts = [text]
        fig = self._fig_ref()
        if fig is not None:
            self._update_hover_texts(fig)

    def _update_hover_texts(self, fig: go.Figure):
        if self._hover_texts is None:
            return

        fig.update_traces(
            customdata=self._hover_texts * self._props["x"].size,
            selector={"uid": self._props["uid"]},
        )
