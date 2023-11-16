from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle
from whitecanvas.protocols import BarProtocol, check_protocol
from whitecanvas.types import FacePattern


@check_protocol(BarProtocol)
class Bars(BarContainer):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        patches = []
        width = xhigh - xlow
        height = yhigh - ylow
        for x, y, dx, dy in zip(xlow, ylow, width, height):
            r = Rectangle(xy=(x, y), width=dx, height=dy)
            r.get_path()._interpolation_steps = 100
            patches.append(r)
        super().__init__(patches)

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.patches[0].get_visible()

    def _plt_set_visible(self, visible: bool):
        for patch in self.patches:
            patch.set_visible(visible)

    def _plt_get_zorder(self) -> int:
        return self.patches[0].get_zorder()

    def _plt_set_zorder(self, zorder: int):
        for patch in self.patches:
            patch.set_zorder(zorder)

    ##### XXYYDataProtocol #####
    def _plt_get_data(self):
        n = len(self.patches)
        x0 = np.empty(n, dtype=np.float32)
        x1 = np.empty(n, dtype=np.float32)
        y0 = np.empty(n, dtype=np.float32)
        y1 = np.empty(n, dtype=np.float32)
        for i, patch in enumerate(self.patches):
            x0[i] = patch.get_x()
            x1[i] = patch.get_x() + patch.get_width()
            y0[i] = patch.get_y()
            y1[i] = patch.get_y() + patch.get_height()
        return x0, x1, y0, y1

    def _plt_set_data(self, x0, x1, y0, y1):
        n = len(self.patches)
        ninput = len(x0)
        if n != ninput:
            raise ValueError(f"Existing data has {n} bars but trying to set {ninput} bars.")
        for patch, x0i, x1i, y0i, y1i in zip(self.patches, x0, x1, y0, y1):
            patch.set_x(x0i)
            patch.set_width(x1i - x0i)
            patch.set_y(y0i)
            patch.set_height(y1i - y0i)

    ##### HasFace protocol #####

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self.patches[0].get_facecolor()

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        for patch in self.patches:
            patch.set_facecolor(color)

    def _plt_get_face_pattern(self) -> FacePattern:
        return FacePattern(self.patches[0].get_hatch())

    def _plt_set_face_pattern(self, pattern: FacePattern):
        for patch in self.patches:
            patch.set_hatch(pattern.value)

    ##### HasEdges protocol #####

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.patches[0].get_edgecolor()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        for patch in self.patches:
            patch.set_edgecolor(color)

    def _plt_get_edge_style(self) -> str:
        return self.patches[0].get_linestyle()

    def _plt_set_edge_style(self, style: str):
        for patch in self.patches:
            patch.set_linestyle(style)

    def _plt_get_edge_width(self) -> float:
        return self.patches[0].get_linewidth()

    def _plt_set_edge_width(self, width: float):
        for patch in self.patches:
            patch.set_linewidth(width)
