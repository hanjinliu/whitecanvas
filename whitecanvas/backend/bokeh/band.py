from __future__ import annotations
from typing import Literal

import numpy as np
from cmap import Color
from whitecanvas.protocols import BandProtocol, check_protocol
from whitecanvas.types import FacePattern, LineStyle

import bokeh.models as bk_models
from ._base import (
    BokehLayer,
    to_bokeh_line_style,
    from_bokeh_line_style,
    to_bokeh_hatch,
    from_bokeh_hatch,
)


@check_protocol(BandProtocol)
class Band(BokehLayer[bk_models.VArea | bk_models.HArea]):
    def __init__(
        self,
        t: np.ndarray,
        ydata0: np.ndarray,
        ydata1: np.ndarray,
        orient: Literal["vertical", "horizontal"],
    ):
        self._data = bk_models.ColumnDataSource(data=dict(t=t, y0=ydata0, y1=ydata1))
        if orient == "vertical":
            self._model = bk_models.VArea(x="t", y1="y0", y2="y1")
        else:
            self._model = bk_models.HArea(y="t", x1="y0", y1="y1")

    def _plt_get_visible(self):
        return self._model.visible

    def _plt_set_visible(self, visible):
        self._model.visible = visible

    def _plt_set_zorder(self, zorder: int):
        pass

    ##### XYYDataProtocol #####
    def _plt_get_vertical_data(self):
        data = self._data.data
        return data["t"], data["y0"], data["y1"]

    _plt_get_horizontal_data = _plt_get_vertical_data

    def _plt_set_vertical_data(self, t, ydata0, ydata1):
        self._data.data = dict(t=t, y0=ydata0, y1=ydata1)

    _plt_set_horizontal_data = _plt_set_vertical_data

    def _plt_get_face_color(self):
        return np.array(Color(self._model.fill_color).rgba)

    def _plt_set_face_color(self, color):
        self._model.fill_color = Color(color).hex

    def _plt_get_face_pattern(self) -> FacePattern:
        return from_bokeh_hatch(self._model.hatch_pattern)

    def _plt_set_face_pattern(self, pattern: FacePattern):
        self._model.hatch_pattern = to_bokeh_hatch(pattern)

    def _plt_get_edge_color(self):
        return np.array(Color(self._model.line_color).rgba)

    def _plt_set_edge_color(self, color):
        self._model.line_color = Color(color).hex

    def _plt_get_edge_width(self):
        return self._model.line_width

    def _plt_set_edge_width(self, width: float):
        self._model.line_width = width

    def _plt_get_edge_style(self):
        return from_bokeh_line_style(self._model.line_dash)

    def _plt_set_edge_style(self, style: LineStyle):
        self._model.line_dash = to_bokeh_line_style(style)

    def _plt_get_antialias(self) -> bool:
        return self._model.line_join == "round"

    def _plt_set_antialias(self, antialias: bool):
        self._model.line_join = "round" if antialias else "miter"
        self._model.line_cap = "round" if antialias else "butt"
