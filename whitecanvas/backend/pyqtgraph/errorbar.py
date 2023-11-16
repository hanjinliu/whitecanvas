from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from qtpy import QtGui
import pyqtgraph as pg
from whitecanvas.protocols import ErrorbarProtocol, check_protocol
from whitecanvas.types import LineStyle
from ._qt_utils import (
    array_to_qcolor,
    from_qt_line_style,
    to_qt_line_style,
)


@check_protocol(ErrorbarProtocol)
class Errorbars(pg.ErrorBarItem):
    def __init__(self, t, y0, y1, orient: Literal["vertical", "horizontal"]):
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        if orient == "vertical":
            yc = (y0 + y1) / 2
            super().__init__(x=t, y=yc, height=y1 - y0, beam=0, pen=pen)
        elif orient == "horizontal":
            xc = (y0 + y1) / 2
            super().__init__(y=t, x=xc, width=y1 - y0, beam=0, pen=pen)
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        self.setVisible(visible)

    def _plt_get_zorder(self) -> int:
        return self.zValue()

    def _plt_set_zorder(self, zorder: int):
        self.setZValue(zorder)

    ##### XYDataProtocol #####
    def _plt_get_vertical_data(self):
        x = self.opts["x"]
        yc = self.opts["y"]
        dy = self.opts["height"] / 2
        return x, yc - dy, yc + dy

    def _plt_set_vertical_data(self, t, y0, y1):
        self.setData(x=t, y=y0, height=y1 - y0)

    def _plt_get_horizontal_data(self):
        y = self.opts["y"]
        xc = self.opts["x"]
        dx = self.opts["width"] / 2
        return y, xc - dx, xc + dx

    def _plt_set_horizontal_data(self, y, x0, x1):
        self.setData(y=y, x=x0, width=x1 - x0)

    ##### HasEdges protocol #####
    def _get_pen(self) -> QtGui.QPen:
        return self.opts["pen"]

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        rgba = self._get_pen().color().getRgbF()
        return np.array(rgba)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        pen = self._get_pen()
        pen.setColor(array_to_qcolor(color))
        self.setData(pen=pen)

    def _plt_get_edge_width(self) -> float:
        return self._get_pen().widthF()

    def _plt_set_edge_width(self, width: float):
        pen = self._get_pen()
        pen.setWidthF(width)
        self.setData(pen=pen)

    def _plt_get_edge_style(self) -> LineStyle:
        return from_qt_line_style(self._get_pen().style())

    def _plt_set_edge_style(self, style: LineStyle):
        pen = self._get_pen()
        pen.setStyle(to_qt_line_style(style))
        self.setData(pen=pen)

    ##### ErrorbarProtocol #####

    def _plt_get_capsize(self) -> float:
        return self.opts["beam"]

    def _plt_set_capsize(self, capsize: float, orient: Literal["vertical", "horizontal"]):
        self.setData(beam=capsize)
