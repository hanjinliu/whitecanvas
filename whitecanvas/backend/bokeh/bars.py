from __future__ import annotations

import bokeh.models as bk_models
import numpy as np

from whitecanvas.backend.bokeh._base import HeteroLayer, SupportsMouseEvents
from whitecanvas.protocols import BarProtocol, check_protocol


@check_protocol(BarProtocol)
class Bars(HeteroLayer[bk_models.Quad], SupportsMouseEvents):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        ndata = len(xlow)
        self._visible = True
        self._data = bk_models.ColumnDataSource(
            data={
                "x0": xlow,
                "x1": xhigh,
                "y0": ylow,
                "y1": yhigh,
                "face_color": ["blue"] * ndata,
                "edge_color": ["black"] * ndata,
                "width": np.zeros(ndata),
                "pattern": [" "] * ndata,
                "style": ["solid"] * ndata,
                "hovertexts": np.array([""] * len(xlow)),
            }
        )
        self._model = bk_models.Quad(
            left="x0",
            right="x1",
            bottom="y0",
            top="y1",
            fill_alpha=1.0,
            line_color="edge_color",
            line_width="width",
            fill_color="face_color",
            hatch_pattern="pattern",
            line_dash="style",
        )

    ##### XXYYDataProtocol #####
    def _plt_get_data(self):
        return (
            self._data.data["x0"],
            self._data.data["x1"],
            self._data.data["y0"],
            self._data.data["y1"],
        )

    def _plt_set_data(self, x0, x1, y0, y1):
        cur_data = self._data.data.copy()
        ndata: int = cur_data["x0"].size
        cur_data.update({"x0": x0, "x1": x1, "y0": y0, "y1": y1})
        cols_to_update = [
            "face_color", "edge_color", "width", "pattern", "style", "hovertexts"
        ]  # fmt: skip
        if x0.size < ndata:
            for key in cols_to_update:
                cur_data[key] = cur_data[key][: x0.size]
        elif x0.size > ndata:
            for key in cols_to_update:
                cur_data[key] = np.concatenate(
                    [cur_data[key], np.full(x0.size - ndata, cur_data[key][-1])]
                )
        self._data.data = cur_data

    def _plt_get_ndata(self) -> int:
        return len(self._data.data["x0"])
