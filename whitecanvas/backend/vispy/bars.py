from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from vispy.scene import visuals
from whitecanvas.protocols import BarProtocol, check_protocol
from whitecanvas.types import LineStyle, FacePattern


@check_protocol(BarProtocol)
class Bars(visuals.Rectangle):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        super().__init__(
            x0=xlow, x1=xhigh, y0=ylow, y1=yhigh,
        )  # fmt: skip

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    def _plt_set_zorder(self, zorder: int):
        pass

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.opts["x0"], self.opts["x1"], self.opts["y0"], self.opts["y1"]

    def _plt_set_data(self, xlow, xhigh, ylow, yhigh):
        self.setOpts(x0=xlow, x1=xhigh, y0=ylow, y1=yhigh)

    ##### HasFace protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self.color

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self.color = color

    def _plt_get_face_pattern(self) -> FacePattern:
        return FacePattern.SOLID

    def _plt_set_face_pattern(self, pattern: FacePattern):
        pass

    ##### HasEdges protocol #####
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.border_color

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.border_color = color

    def _plt_get_edge_width(self) -> float:
        return self.border.width

    def _plt_set_edge_width(self, width: float):
        self.border.width = width

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle.SOLID

    def _plt_set_edge_style(self, style: LineStyle):
        pass
