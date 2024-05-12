from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.quiver import Quiver
from numpy.typing import NDArray

from whitecanvas.backend.matplotlib._base import FakeAxes, MplLayer
from whitecanvas.protocols import VectorsProtocol, check_protocol
from whitecanvas.types import LineStyle

if TYPE_CHECKING:
    from whitecanvas.backend.matplotlib.canvas import Canvas


@check_protocol(VectorsProtocol)
class Vectors(Quiver, MplLayer):
    def __init__(self, x0, dx, y0, dy):
        super().__init__(
            FakeAxes(), x0, y0, dx, dy, angles="xy", scale_units="xy", scale=1
        )

    def _plt_get_data(self):
        return self.X, self.U, self.Y, self.V

    def _plt_set_data(self, x0, dx, y0, dy):
        self.set_UVC(dx, dy)
        self.set_offsets(np.column_stack([x0, y0]))

    def post_add(self, canvas: Canvas):
        self.transform = canvas._axes.transData
        self.set_offset_transform(canvas._axes.transData)
        self._axes = canvas._axes

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
