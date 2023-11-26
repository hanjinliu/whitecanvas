from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from whitecanvas.types import LineStyle
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol

import bokeh.models as bk_models

from whitecanvas.utils.normalize import arr_color, hex_color
from ._base import BokehLayer, to_bokeh_line_style, from_bokeh_line_style


@check_protocol(LineProtocol)
class MonoLine(BokehLayer[bk_models.Line]):
    def __init__(self, xdata, ydata):
        self._data = bk_models.ColumnDataSource(data=dict(x=xdata, y=ydata))
        self._model = bk_models.Line(x="x", y="y", line_join="round", line_cap="round")

    def _plt_get_visible(self) -> bool:
        return self._model.visible

    def _plt_set_visible(self, visible: bool):
        self._model.visible = visible

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
        return np.array(arr_color(self._model.line_color))

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._model.line_color = hex_color(color)

    def _plt_get_antialias(self) -> bool:
        return self._model.line_join == "round"

    def _plt_set_antialias(self, antialias: bool):
        self._model.line_join = "round" if antialias else "miter"
        self._model.line_cap = "round" if antialias else "butt"


@check_protocol(MultiLineProtocol)
class MultiLine(BokehLayer[bk_models.MultiLine]):
    def __init__(self, data: list[NDArray[np.number]]):
        xdata = []
        ydata = []
        for seg in data:
            xdata.append(seg[:, 0])
            ydata.append(seg[:, 1])
        self._data = bk_models.ColumnDataSource(
            dict(
                x=xdata,
                y=ydata,
                color="black",
                width=1,
                style="solid",
            ),
        )
        self._model = bk_models.MultiLine(
            xs="x",
            ys="y",
            line_color="color",
            line_width="width",
            line_dash="style",
            line_join="round",
            line_cap="round",
        )

    def _plt_get_visible(self) -> bool:
        return self._model.visible

    def _plt_set_visible(self, visible: bool):
        self._model.visible = visible

    def _plt_get_data(self):
        xs, ys = self._data.data["x"], self._data.data["y"]
        out = []
        for x, y in zip(xs, ys):
            out.append(np.stack([x, y]), axis=1)
        return out

    def _plt_set_data(self, data):
        xdata = []
        ydata = []
        for seg in data:
            xdata.append(seg[:, 0])
            ydata.append(seg[:, 1])
        self._data.data = dict(x=xdata, y=ydata)

    def _plt_get_edge_width(self) -> float:
        return self._data.data["width"]

    def _plt_set_edge_width(self, width: float):
        self._data.data["width"] = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return from_bokeh_line_style(self._data.data["style"])

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        self._data.data["style"] = to_bokeh_line_style(style)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return arr_color(self._data.data["color"])

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._data.data["color"] = hex_color(color)

    def _plt_get_antialias(self) -> bool:
        return self._model.line_join == "round"

    def _plt_set_antialias(self, antialias: bool):
        self._model.line_join = "round" if antialias else "miter"
        self._model.line_cap = "round" if antialias else "butt"
