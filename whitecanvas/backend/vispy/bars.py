from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from vispy.scene import visuals
from whitecanvas.protocols import BarProtocol, check_protocol
from whitecanvas.types import LineStyle, Hatch
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.backend import _not_implemented


@check_protocol(BarProtocol)
class Bars(visuals.Compound):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        center = np.stack([(xlow + xhigh) / 2, (ylow + yhigh) / 2], axis=1)
        rectangles: list[visuals.Rectangle] = []
        for c, w, h in zip(center, xhigh - xlow, yhigh - ylow):
            w = abs(w)
            h = abs(h)
            if w == 0:
                w = h * 1e-6
            elif h == 0:
                h = w * 1e-6
            rectangles.append(
                visuals.Rectangle(center=c, width=w, height=h, border_width=0)
            )
        super().__init__(rectangles)
        self.unfreeze()
        self._rectangles = rectangles

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        data = []
        for rect in self._rectangles:
            xc, yc = rect.center
            w = rect.width
            h = rect.height
            data.append([xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2])
        return tuple(np.array(data).T)

    def _plt_set_data(self, xlow, xhigh, ylow, yhigh):
        center = np.stack([(xlow + xhigh) / 2, (ylow + yhigh) / 2], axis=1)
        for rect, c, w, h in zip(self._rectangles, center, xhigh - xlow, yhigh - ylow):
            rect.center = c
            rect.width = w
            rect.height = h

    def _plt_get_ndata(self):
        return len(self._rectangles)

    ##### HasFace protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        colors = []
        for rect in self._rectangles:
            colors.append(rect.color)
        return np.stack(colors, axis=0)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, size=self._plt_get_ndata())
        for rect, c in zip(self._rectangles, color):
            rect.color = c

    _plt_get_face_pattern, _plt_set_face_pattern = _not_implemented.face_patterns()

    ##### HasEdges protocol #####
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        colors = []
        for rect in self._rectangles:
            colors.append(rect.border_color)
        return np.stack(colors, axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, size=self._plt_get_ndata())
        for rect, c in zip(self._rectangles, color):
            rect.border_color = c

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        # NOTE: vispy forgot to add `border_width` to `Rectangle` class?
        widths = []
        for rect in self._rectangles:
            widths.append(rect._border_width)
        return np.array(widths)

    def _plt_set_edge_width(self, width: float):
        if np.isscalar(width):
            width = np.full(self._plt_get_ndata(), width)
        for rect, w in zip(self._rectangles, width):
            rect._border_width = w

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_styles()
