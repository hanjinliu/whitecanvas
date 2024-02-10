from __future__ import annotations

import bokeh.models as bk_models
import numpy as np

from whitecanvas.backend.bokeh._base import (
    BokehLayer,
    SupportsMouseEvents,
    from_bokeh_hatch,
    to_bokeh_hatch,
)
from whitecanvas.protocols import BandProtocol, check_protocol
from whitecanvas.types import Hatch, LineStyle, Orientation
from whitecanvas.utils.normalize import arr_color, hex_color


@check_protocol(BandProtocol)
class Band(BokehLayer["bk_models.VArea | bk_models.HArea"], SupportsMouseEvents):
    def __init__(
        self,
        t: np.ndarray,
        ydata0: np.ndarray,
        ydata1: np.ndarray,
        orient: Orientation,
    ):
        self._data = bk_models.ColumnDataSource(
            data={
                "t": t,
                "y0": ydata0,
                "y1": ydata1,
                "hovertexts": np.array([""] * len(t)),
            }
        )
        if orient.is_vertical:
            self._model = bk_models.VArea(x="t", y1="y0", y2="y1")
        else:
            self._model = bk_models.HArea(y="t", x1="y0", x2="y1")
        self._edge_color = np.zeros(4)
        self._edge_width = 0
        self._edge_style = LineStyle.SOLID
        self._visible = True
        self._face_color = self._model.fill_color

    def _plt_get_visible(self):
        return self._visible

    def _plt_set_visible(self, visible):
        if visible:
            self._model.fill_color = "#00000000"
        else:
            self._model.fill_color = self._face_color
        self._visible = visible

    ##### XYYDataProtocol #####
    def _plt_get_vertical_data(self):
        data = self._data.data
        return data["t"], data["y0"], data["y1"]

    _plt_get_horizontal_data = _plt_get_vertical_data

    def _plt_set_vertical_data(self, t, ydata0, ydata1):
        self._data.data = {"t": t, "y0": ydata0, "y1": ydata1}

    _plt_set_horizontal_data = _plt_set_vertical_data

    def _plt_get_face_color(self):
        return arr_color(self._face_color)

    def _plt_set_face_color(self, color):
        self._face_color = hex_color(color)
        if self._visible:
            self._model.fill_color = self._face_color

    def _plt_get_face_hatch(self) -> Hatch:
        return from_bokeh_hatch(self._model.hatch_pattern)

    def _plt_set_face_hatch(self, pattern: Hatch):
        self._model.hatch_pattern = to_bokeh_hatch(pattern)

    def _plt_get_edge_color(self):
        return self._edge_color

    def _plt_set_edge_color(self, color):
        self._edge_color = arr_color(color)

    def _plt_get_edge_width(self):
        return self._edge_width

    def _plt_set_edge_width(self, width: float):
        self._edge_width = width

    def _plt_get_edge_style(self):
        return self._edge_style

    def _plt_set_edge_style(self, style: LineStyle):
        self._edge_style = style

    def _plt_set_hover_text(self, text: str):
        self._data.data["hovertexts"] = [text] * len(self._data.data["t"])
