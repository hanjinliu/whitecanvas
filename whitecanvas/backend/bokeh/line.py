from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from cmap import Color
from whitecanvas.types import LineStyle
from whitecanvas.protocols import LineProtocol, MultiLinesProtocol, check_protocol

import bokeh.models as bk_models
from ._base import BokehLayer, to_bokeh_line_style, from_bokeh_line_style


@check_protocol(LineProtocol)
class MonoLine(BokehLayer):
    def __init__(self, xdata, ydata):
        self._data = bk_models.ColumnDataSource(data=dict(x=xdata, y=ydata))
        self._model = bk_models.Line(x="x", y="y", line_join="round", line_cap="round")

    def _plt_get_visible(self) -> bool:
        return self._model.visible

    def _plt_set_visible(self, visible: bool):
        self._model.visible = visible

    def _plt_set_zorder(self, zorder: int):
        pass

    def _plt_get_data(self):
        return self._data.data["x"], self._data.data["y"]

    def _plt_set_data(self, xdata, ydata):
        self._data.data = dict(x=xdata, y=ydata)

    def _plt_get_edge_width(self) -> float:
        return self._model.line_width

    def _plt_set_edge_width(self, width: float):
        self._model.line_width = width

    def _plt_get_edge_style(self) -> LineStyle:
        return from_bokeh_line_style(self._model.line_dash)

    def _plt_set_edge_style(self, style: LineStyle):
        self._model.line_dash = to_bokeh_line_style(style)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(Color(self._model.line_color).rgba)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._model.line_color = Color(color).hex

    def _plt_get_antialias(self) -> bool:
        return self._model.line_join == "round"

    def _plt_set_antialias(self, antialias: bool):
        self._model.line_join = "round" if antialias else "miter"
        self._model.line_cap = "round" if antialias else "butt"
