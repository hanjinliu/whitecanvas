from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from whitecanvas.protocols import BarProtocol, check_protocol
from whitecanvas.types import FacePattern, LineStyle
from whitecanvas.utils.normalize import rgba_str_color, arr_color
from ._base import PlotlyLayer


@check_protocol(BarProtocol)
class Bars(PlotlyLayer):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        x = (xlow + xhigh) / 2
        ndata = len(x)
        self._props = {
            "x": x,
            "y": yhigh,
            "width": xhigh - xlow,
            "base": ylow,
            "marker": {
                "color": ["blue"] * ndata,
                "pattern": {
                    "shape": "-",
                },
                "line": {
                    "width": [1] * ndata,
                    "color": ["blue"] * ndata,
                },
            },
            "visible": True,
            "type": "bar",
        }

    ##### XXYYDataProtocol #####
    def _plt_get_data(self):
        half_width = self._props["width"] / 2
        x = self._props["x"]
        ylow = self._props["base"]
        yhigh = self._props["y"]
        xlow = x - half_width / 2
        xhigh = x + half_width / 2
        return xlow, xhigh, ylow, yhigh

    def _plt_set_data(self, x0, x1, y0, y1):
        self._props["x"] = (x0 + x1) / 2
        self._props["y"] = y1
        self._props["width"] = x1 - x0
        self._props["base"] = y0

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        color = self._props["marker"]["color"]
        if len(color) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack([arr_color(c) for c in color], axis=0)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = [rgba_str_color(color)] * len(self._props["x"])
        else:
            color = [rgba_str_color(c) for c in color]
        self._props["marker"]["color"] = color

    def _plt_get_face_pattern(self) -> list[FacePattern]:
        return [FacePattern(p) for p in self._props["marker"]["pattern"]["shape"]]

    def _plt_set_face_pattern(self, pattern: FacePattern | list[FacePattern]):
        if isinstance(pattern, FacePattern):
            ptn = [pattern.value] * len(self._props["x"])
        else:
            ptn = [ptn.value for ptn in pattern]
        self._props["marker"]["pattern"]["shape"] = ptn

    ##### HasEdges protocol #####

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.array(self._props["marker"]["line"]["width"])

    def _plt_set_edge_width(self, width: float):
        if np.isscalar(width):
            width = np.full(len(self._props["x"]), width)
        self._props["marker"]["line"]["width"] = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [LineStyle.SOLID] * len(self._props["x"])

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        pass

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        color = self._props["line"]["color"]
        if len(color) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack([arr_color(c) for c in color], axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = [rgba_str_color(color)] * len(self._props["x"])
        else:
            color = [rgba_str_color(c) for c in color]
        self._props["marker"]["line"]["color"] = color
