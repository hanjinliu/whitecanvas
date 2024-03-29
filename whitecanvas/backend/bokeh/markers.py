from __future__ import annotations

import bokeh.models as bk_models
import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend.bokeh._base import (
    HeteroLayer,
    SupportsMouseEvents,
    from_bokeh_symbol,
    to_bokeh_symbol,
)
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Symbol
from whitecanvas.utils.type_check import is_real_number


@check_protocol(MarkersProtocol)
class Markers(HeteroLayer[bk_models.Scatter], SupportsMouseEvents):
    def __init__(self, xdata, ydata):
        ndata = len(xdata)
        self._data = bk_models.ColumnDataSource(
            data={
                "x": xdata,
                "y": ydata,
                "sizes": np.full(ndata, 10.0),
                "face_color": ["blue"] * ndata,
                "edge_color": ["black"] * ndata,
                "width": np.zeros(ndata),
                "pattern": [" "] * ndata,
                "style": ["solid"] * ndata,
                "hovertexts": [""] * ndata,
            }
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

    def _plt_set_data(self, xdata: NDArray[np.number], ydata: NDArray[np.number]):
        ndata = self._data.data["x"].size
        cur_data = self._data.data.copy()
        cur_data["x"] = xdata
        cur_data["y"] = ydata
        cols_to_update = [
            "sizes", "face_color", "edge_color", "width", "pattern", "style",
            "hovertexts"
        ]  # fmt: skip
        if xdata.size < ndata:
            for key in cols_to_update:
                cur_data[key] = cur_data[key][: xdata.size]
        elif xdata.size > ndata:
            for key in cols_to_update:
                cur_data[key] = np.concatenate(
                    [cur_data[key], np.full(xdata.size - ndata, cur_data[key][-1])]
                )
        self._data.data = cur_data

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
        if is_real_number(size):
            size = np.full(self._plt_get_ndata(), size)
        self._data.data["sizes"] = size

    def _plt_get_ndata(self) -> int:
        return len(self._data.data["x"])
