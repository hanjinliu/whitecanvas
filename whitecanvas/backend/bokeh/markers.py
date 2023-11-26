from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from whitecanvas.types import LineStyle, Symbol, FacePattern
from whitecanvas.protocols import MarkersProtocol, HeteroMarkersProtocol, check_protocol

import bokeh.models as bk_models

from whitecanvas.utils.normalize import arr_color, hex_color
from ._base import (
    BokehLayer,
    HeteroLayer,
    to_bokeh_line_style,
    from_bokeh_line_style,
    to_bokeh_symbol,
    from_bokeh_symbol,
    to_bokeh_hatch,
    from_bokeh_hatch,
)


@check_protocol(MarkersProtocol)
class Markers(BokehLayer[bk_models.Scatter]):
    def __init__(self, xdata, ydata):
        self._data = bk_models.ColumnDataSource(data=dict(x=xdata, y=ydata))
        self._model = bk_models.Scatter(x="x", y="y")
        self._size = 10.0
        self._visible = True

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._model.size = self._size
        else:
            self._model.size = 0
        self._visible = visible

    def _plt_get_data(self):
        return self._data.data["x"], self._data.data["y"]

    def _plt_set_data(self, xdata, ydata):
        self._data.data = dict(x=xdata, y=ydata)

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return arr_color(self._model.fill_color)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self._model.fill_color = hex_color(color)

    def _plt_get_face_pattern(self) -> FacePattern:
        return from_bokeh_hatch(self._model.hatch_pattern)

    def _plt_set_face_pattern(self, pattern: FacePattern):
        self._model.hatch_pattern = to_bokeh_hatch(pattern)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return arr_color(self._model.line_color)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._model.line_color = hex_color(color)

    def _plt_get_edge_width(self) -> float:
        return self._model.line_width

    def _plt_set_edge_width(self, width: float):
        self._model.line_width = width

    def _plt_get_edge_style(self) -> LineStyle:
        return from_bokeh_line_style(self._model.line_dash)

    def _plt_set_edge_style(self, style: LineStyle):
        self._model.line_dash = to_bokeh_line_style(style)

    def _plt_get_symbol(self) -> Symbol:
        sym = self._model.marker
        rot = int(round(self._model.angle / np.pi * 2))
        return from_bokeh_symbol(sym, rot)

    def _plt_set_symbol(self, symbol: Symbol):
        sym, rot = to_bokeh_symbol(symbol)
        self._model.marker = sym
        self._model.angle = rot * np.pi / 2

    def _plt_get_symbol_size(self) -> float:
        return self._size

    def _plt_set_symbol_size(self, size: float):
        self._size = size
        if self._visible:
            self._model.size = size


@check_protocol(HeteroMarkersProtocol)
class HeteroMarkers(HeteroLayer[bk_models.Scatter]):
    def __init__(self, xdata, ydata):
        ndata = len(xdata)
        self._data = bk_models.ColumnDataSource(
            data=dict(
                x=xdata,
                y=ydata,
                size=np.full(ndata, 10.0),
                face_color=["blue"] * ndata,
                edge_color=["black"] * ndata,
                width=np.zeros(ndata),
                pattern=[" "] * ndata,
                style=["solid"] * ndata,
            )
        )
        self._model = bk_models.Scatter(
            x="x",
            y="y",
            line_color="edge_color",
            line_width="width",
            fill_color="face_color",
            hatch_pattern="pattern",
            line_dash="style",
        )

    def _plt_get_visible(self) -> bool:
        return self._model.visible

    def _plt_set_visible(self, visible: bool):
        self._model.visible = visible

    def _plt_get_data(self):
        return self._data.data["x"], self._data.data["y"]

    def _plt_set_data(self, xdata, ydata):
        self._data.data = dict(x=xdata, y=ydata)

    def _plt_get_symbol(self) -> Symbol:
        sym = self._model.marker
        rot = int(round(self._model.angle / np.pi * 2))
        return from_bokeh_symbol(sym, rot)

    def _plt_set_symbol(self, symbol: Symbol):
        sym, rot = to_bokeh_symbol(symbol)
        self._model.marker = sym
        self._model.angle = rot * np.pi / 2

    def _plt_get_symbol_size(self) -> float:
        return self._model.size

    def _plt_set_symbol_size(self, size: float):
        self._model.size = size

    def _plt_get_ndata(self) -> int:
        return len(self._data.data["x"])
