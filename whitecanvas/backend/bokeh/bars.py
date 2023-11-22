from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from cmap import Color
from whitecanvas.protocols import BarProtocol, check_protocol
from whitecanvas.types import FacePattern, LineStyle

import bokeh.models as bk_models
from ._base import (
    BokehLayer,
    to_bokeh_line_style,
    from_bokeh_line_style,
    to_bokeh_hatch,
    from_bokeh_hatch,
)


@check_protocol(BarProtocol)
class Bars(BokehLayer[bk_models.Quad]):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        ndata = len(xlow)
        self._data = bk_models.ColumnDataSource(
            data=dict(
                x0=xlow,
                x1=xhigh,
                y0=ylow,
                y1=yhigh,
                face_color=["blue"] * ndata,
                edge_color=["black"] * ndata,
                width=np.zeros(ndata),
                pattern=[" "] * ndata,
                style=["solid"] * ndata,
            )
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

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self._model.visible

    def _plt_set_visible(self, visible: bool):
        self._model.visible = visible

    def _plt_set_zorder(self, zorder: int):
        pass

    ##### XXYYDataProtocol #####
    def _plt_get_data(self):
        return (
            self._data.data["x0"],
            self._data.data["x1"],
            self._data.data["y0"],
            self._data.data["y1"],
        )

    def _plt_set_data(self, x0, x1, y0, y1):
        self._data.data = dict(x0=x0, x1=x1, y0=y0, y1=y1)

    ##### HasFace protocol #####

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return np.stack([Color(c).rgba for c in self._model.fill_color], axis=1)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = [Color(color).hex] * len(self._data.data["x"])
        else:
            color = [Color(c).hex for c in color]
        self._model.fill_color = color

    def _plt_get_face_pattern(self) -> list[FacePattern]:
        return [from_bokeh_hatch(p) for p in self._model.hatch_pattern]

    def _plt_set_face_pattern(self, pattern: FacePattern | list[FacePattern]):
        if isinstance(pattern, FacePattern):
            ptn = [pattern] * len(self._data.data["x"])
        else:
            ptn = [to_bokeh_hatch(p) for p in pattern]
        self._model.hatch_pattern = ptn

    ##### HasEdges protocol #####

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self._model.line_width

    def _plt_set_edge_width(self, width: float):
        if np.isscalar(width):
            width = np.full(len(self._data.data["x"]), width)
        self._model.line_width = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_bokeh_line_style(d) for d in self._model.line_dash]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * len(self._data.data["x"])
        val = [to_bokeh_line_style(s) for s in style]
        self._model.line_dash = val

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.stack([Color(c).rgba for c in self._model.line_color], axis=1)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = [Color(color).hex] * len(self._data.data["x"])
        else:
            color = [Color(c).hex for c in color]
        self._model.line_color = color
