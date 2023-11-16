from __future__ import annotations

import numpy as np

from matplotlib.collections import PolyCollection
from whitecanvas.protocols import FillBetweenProtocol, check_protocol
from whitecanvas.types import FacePattern, LineStyle


@check_protocol(FillBetweenProtocol)
class FillBetween(PolyCollection):
    def __init__(
        self,
        xdata: np.ndarray,
        ydata0: np.ndarray,
        ydata1: np.ndarray,
    ):
        verts = np.concatenate(
            [
                np.stack([xdata, ydata0], axis=1),
                np.stack([xdata[::-1], ydata1[::-1]], axis=1),
            ],
            axis=0,
        )
        super().__init__([verts], closed=True)
        self._xdata = xdata
        self._y0 = ydata0
        self._y1 = ydata1

    def _plt_get_visible(self):
        return self.get_visible()

    def _plt_set_visible(self, visible):
        self.set_visible(visible)

    def _plt_get_zorder(self) -> int:
        return self.get_zorder()

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

    ##### XYYDataProtocol #####
    def _plt_get_data(self):
        return self._xdata, self._y0, self._y1

    def _plt_set_data(self, xdata, ydata0, ydata1):
        verts = np.concatenate(
            [
                np.stack([xdata, ydata0], axis=1),
                np.stack([xdata[::-1], ydata1[::-1]], axis=1),
            ],
            axis=0,
        )
        self.set_verts([verts])
        self._xdata = xdata
        self._y0 = ydata0
        self._y1 = ydata1

    def _plt_get_face_color(self):
        return self.get_facecolor()[0]

    def _plt_set_face_color(self, color):
        self.set_facecolor(color)

    def _plt_get_face_pattern(self) -> FacePattern:
        return FacePattern(self.get_hatch())

    def _plt_set_face_pattern(self, style: FacePattern):
        self.set_hatch(style.value)

    def _plt_get_edge_color(self):
        return self.get_edgecolor()[0]

    def _plt_set_edge_color(self, color):
        self.set_edgecolor(color)

    def _plt_get_edge_width(self):
        return self.get_linewidth()[0]

    def _plt_set_edge_width(self, width: float):
        self.set_linewidth(width)

    def _plt_get_edge_style(self):
        return LineStyle(self.get_linestyle()[0])

    def _plt_set_edge_style(self, style: LineStyle):
        self.set_linestyle(style.value)

    def _plt_get_antialias(self):
        return self.get_antialiased()

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)
