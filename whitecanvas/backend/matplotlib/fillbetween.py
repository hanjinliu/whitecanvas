from __future__ import annotations
from typing import Literal

import numpy as np

from matplotlib.collections import PolyCollection
from whitecanvas.protocols import BandProtocol, check_protocol
from whitecanvas.types import FacePattern, LineStyle


@check_protocol(BandProtocol)
class Band(PolyCollection):
    def __init__(
        self,
        t: np.ndarray,
        ydata0: np.ndarray,
        ydata1: np.ndarray,
        orient: Literal["vertical", "horizontal"],
    ):
        if orient == "vertical":
            fw = np.stack([t, ydata0], axis=1)
            bw = np.stack([t[::-1], ydata1[::-1]], axis=1)
        elif orient == "horizontal":
            fw = np.stack([ydata0, t], axis=1)
            bw = np.stack([ydata1[::-1], t[::-1]], axis=1)
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")
        verts = np.concatenate([fw, bw], axis=0)
        super().__init__([verts], closed=True)
        self._t = t
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
    def _plt_get_vertical_data(self):
        return self._t, self._y0, self._y1

    def _plt_get_horizontal_data(self):
        return self._t, self._y0, self._y1

    def _plt_set_vertical_data(self, t, ydata0, ydata1):
        verts = np.concatenate(
            [
                np.stack([t, ydata0], axis=1),
                np.stack([t[::-1], ydata1[::-1]], axis=1),
            ],
            axis=0,
        )
        self.set_verts([verts])
        self._t = t
        self._y0 = ydata0
        self._y1 = ydata1

    def _plt_set_horizontal_data(self, t, ydata0, ydata1):
        verts = np.concatenate(
            [
                np.stack([ydata0, t], axis=1),
                np.stack([ydata1[::-1], t[::-1]], axis=1),
            ],
            axis=0,
        )
        self.set_verts([verts])
        self._t = t
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
