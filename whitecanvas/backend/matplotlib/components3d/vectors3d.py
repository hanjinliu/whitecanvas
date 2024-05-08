from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.quiver import Quiver
from mpl_toolkits.mplot3d import proj3d
from numpy.typing import NDArray

from whitecanvas.backend.matplotlib._base import FakeAxes, MplLayer
from whitecanvas.types import LineStyle

if TYPE_CHECKING:
    from whitecanvas.backend.matplotlib.components3d import Canvas3D


class Vectors3D(Quiver, MplLayer):
    def __init__(self, x0, x1, y0, y1, z0, z1):
        super().__init__(
            FakeAxes(), x0, y0, x1, y1, angles="xy", scale_units="xy", scale=1
        )
        self._3d_x = x0
        self._3d_y = y0
        self._3d_z = z0
        self._3d_u = x1 - x0
        self._3d_v = y1 - y0
        self._3d_w = z1 - z0

    def do_3d_projection(self, renderer=None):
        x, y, z = proj3d.proj_transform(self._3d_x, self._3d_y, self._3d_z, self.axes.M)
        u, v, w = proj3d.proj_transform(self._3d_u, self._3d_v, self._3d_w, self.axes.M)
        self.set_UVC(u, v)
        self.set_offsets(np.column_stack([x, y]))
        return np.min(z)

    def post_add(self, canvas: Canvas3D):
        self.transform = canvas._axes.transData
        self.set_offset_transform(canvas._axes.transData)
        self._axes = canvas._axes

    def _plt_get_data(self):
        return self._3d_x, self._3d_y, self._3d_z, self._3d_u, self._3d_v, self._3d_w

    def _plt_set_data(self, x0, x1, y0, y1, z0, z1):
        self._3d_x = x0
        self._3d_y = y0
        self._3d_z = z0
        self._3d_u = x1 - x0
        self._3d_v = y1 - y0
        self._3d_w = z1 - z0
        self.do_3d_projection()

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> float:
        return self.get_linewidth()

    def _plt_set_edge_width(self, width: float):
        self.set_linewidth(width)

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle(self.get_linestyle())

    def _plt_set_edge_style(self, style: LineStyle):
        self.set_linestyle(style.value)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_edgecolor()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_color(color)

    def _plt_get_antialias(self) -> bool:
        return self.get_antialiased()

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)
