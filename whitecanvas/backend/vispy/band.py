from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from vispy.scene import visuals
from whitecanvas.protocols import BandProtocol, check_protocol
from whitecanvas.types import LineStyle, FacePattern


@check_protocol(BandProtocol)
class Band(visuals.Polygon):
    def __init__(self, t, ydata0, ydata1, orient: Literal["vertical", "horizontal"]):
        if orient == "vertical":
            fw = np.stack([t, ydata0], axis=1)
            bw = np.stack([t[::-1], ydata1[::-1]], axis=1)
        elif orient == "horizontal":
            fw = np.stack([ydata0, t], axis=1)
            bw = np.stack([ydata1[::-1], t[::-1]], axis=1)
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")
        verts = np.concatenate([fw, bw], axis=0)
        self._edge_style = LineStyle.SOLID
        super().__init__(verts)
        self._t = t
        self._y0 = ydata0
        self._y1 = ydata1

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    def _plt_set_zorder(self, zorder: int):
        pass

    ##### XYDataProtocol #####
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
        self.pos = verts
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
        self.pos = verts
        self._t = t
        self._y0 = ydata0
        self._y1 = ydata1

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
