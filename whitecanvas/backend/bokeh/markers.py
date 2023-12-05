from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from whitecanvas.types import Symbol
from whitecanvas.protocols import MarkersProtocol, check_protocol

import bokeh.models as bk_models
from bokeh.events import Tap

from ._base import HeteroLayer, to_bokeh_symbol, from_bokeh_symbol


@check_protocol(MarkersProtocol)
class Markers(HeteroLayer[bk_models.Scatter]):
    def __init__(self, xdata, ydata):
        ndata = len(xdata)
        self._data = bk_models.ColumnDataSource(
            data=dict(
                x=xdata,
                y=ydata,
                sizes=np.full(ndata, 10.0),
                face_color=["blue"] * ndata,
                edge_color=["black"] * ndata,
                width=np.zeros(ndata),
                pattern=[" "] * ndata,
                style=["solid"] * ndata,
                hovertexts=[""] * ndata,
            )
        )
        self._model = bk_models.Scatter(
            x="x",
            y="y",
            size="sizes",
            line_color="edge_color",
            line_width="width",
            fill_color="face_color",
            hatch_pattern="pattern",
            line_dash="style",
        )
        self._visible = True

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

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return self._data.data["sizes"]

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if isinstance(size, (int, float, np.number)):
            size = np.full(self._plt_get_ndata(), size)
        self._data.data["sizes"] = size

    def _plt_get_ndata(self) -> int:
        return len(self._data.data["x"])

    def _plt_connect_pick_event(self, callback):
        self._model.on_event(Tap, lambda e: print(e))
        # self._model.on_event(Tap, lambda e: callback(e.indices))
        pass  # TODO

    def _plt_set_hover_text(self, text: list[str]):
        self._data.data["hovertexts"] = text
