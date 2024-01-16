from __future__ import annotations

import numpy as np
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.protocols import BarProtocol, check_protocol
from whitecanvas.types import Hatch, LineStyle
from whitecanvas.utils.normalize import as_color_array


@check_protocol(BarProtocol)
class Bars(BarContainer, MplLayer):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        patches = []
        width = xhigh - xlow
        height = yhigh - ylow
        for x, y, dx, dy in zip(xlow, ylow, width, height):
            r = Rectangle(xy=(x, y), width=dx, height=dy, linestyle="-")
            r.get_path()._interpolation_steps = 100
            patches.append(r)
        super().__init__(patches)
        self._visible = True

    def _plt_get_visible(self):
        return self._visible

    def _plt_set_visible(self, visible):
        for patch in self.patches:
            patch.set_visible(visible)
        self._visible = visible

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
        for patch, x0i, x1i, y0i, y1i in zip(self.patches, x0, x1, y0, y1):
            patch.set_x(x0i)
            patch.set_width(x1i - x0i)
            patch.set_y(y0i)
            patch.set_height(y1i - y0i)

    ##### HasFace protocol #####

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return np.stack([patch.get_facecolor() for patch in self.patches], axis=0)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, size=len(self.patches))
        for patch, c in zip(self.patches, color):
            patch.set_facecolor(c)

    def _plt_get_face_pattern(self) -> list[Hatch]:
        return [Hatch(patch.get_hatch() or "") for patch in self.patches]

    def _plt_set_face_pattern(self, pattern: Hatch | list[Hatch]):
        if isinstance(pattern, Hatch):
            pattern = [pattern] * len(self.patches)
        for pat, patch in zip(pattern, self.patches):
            patch.set_hatch(None if pat is Hatch.SOLID else pat.value)

    ##### HasEdges protocol #####

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.stack([patch.get_edgecolor() for patch in self.patches], axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, size=len(self.patches))
        for patch, c in zip(self.patches, color):
            patch.set_edgecolor(c)

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [LineStyle(patch.get_linestyle()) for patch in self.patches]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * len(self.patches)
        for patch, s in zip(self.patches, style):
            patch.set_linestyle(s.value)

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.array([patch.get_linewidth() for patch in self.patches])

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if np.isscalar(width):
            width = [width] * len(self.patches)
        for patch, w in zip(self.patches, width):
            patch.set_linewidth(w)
