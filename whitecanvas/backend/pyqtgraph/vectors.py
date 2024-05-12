from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy import QtGui

from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._qt_utils import (
    from_qt_line_style,
    to_qt_line_style,
)
from whitecanvas.backend.pyqtgraph.line import MultiLine
from whitecanvas.types import LineStyle


class ArrowHeads(pg.ItemGroup):
    def __init__(self, pos: NDArray[np.floating], angle: NDArray[np.floating]):
        super().__init__()
        for i in range(pos.shape[0]):
            item = pg.ArrowItem(
                pos=pos[i],
                angle=angle[i],
                pen=QtGui.QPen(QtGui.QColor(0, 0, 0, 0)),
                headLen=15,
                pxMode=True,
            )
            self.addItem(item)

    def setData(self, pos: NDArray[np.floating], angle: NDArray[np.floating]):
        ndata = pos.shape[0]
        children = self.childItems()
        scene = self.scene()
        if ndata < len(children):
            for item in self.childItems()[ndata:]:
                scene.removeItem(item)
        elif ndata > len(children):
            for _ in range(ndata - len(children)):
                item = pg.ArrowItem(
                    pen=QtGui.QPen(QtGui.QColor(0, 0, 0, 0)),
                    headLen=15,
                    pxMode=True,
                )
                self.addItem(item)
        for i, item in enumerate(self.childItems()):
            item.setStyle(pos=pos[i], angle=angle[i])

    def setPens(self, pens: list[QtGui.QPen]):
        for pen, item in zip(pens, self.childItems()):
            # item.setPen(pen)
            item.setBrush(QtGui.QBrush(pen.color()))

    if TYPE_CHECKING:

        def childItems(self) -> list[pg.ArrowItem]: ...


class Vectors(pg.ItemGroup, PyQtLayer):
    def __init__(self, x0, dx, y0, dy):
        super().__init__()
        angles = np.degrees(np.arctan2(dy, -dx))
        start = np.column_stack([x0, y0])
        vec = np.column_stack([dx, dy])
        stop = start + vec
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        self._brush = QtGui.QBrush(pen.color())
        arrow_heads = ArrowHeads(stop, angles)
        arrow_tails = MultiLine(np.stack([start, stop], axis=1))
        self.addItem(arrow_heads)
        self.addItem(arrow_tails)
        self._start = start
        self._vec = vec
        self._pen_width = pen.widthF()
        self._pen_style = from_qt_line_style(pen.style())
        self._arrow_heads = arrow_heads
        self._arrow_tails = arrow_tails

    def _plt_get_data(self):
        return self._start[:, 0], self._start[:, 1], self._vec[:, 0], self._vec[:, 1]

    def _plt_set_data(self, x0, dx, y0, dy):
        angles = np.degrees(np.arctan2(dy, -dx))
        start = np.column_stack([x0, y0])
        vec = np.column_stack([dx, dy])
        self._arrow_heads.setData(start + vec, angles)
        self._arrow_tails._plt_set_data(np.stack([start, start + vec], axis=1))
        self._start = start
        self._vec = vec

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> float:
        return self._pen_width

    def _plt_set_edge_width(self, width: float):
        self._arrow_tails._plt_set_edge_width(width)
        self._pen_width = width

    def _plt_get_edge_style(self) -> LineStyle:
        return self._pen_style

    def _plt_set_edge_style(self, style: LineStyle):
        self._arrow_tails._plt_set_edge_style(style)
        self._pen_style = to_qt_line_style(style)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self._arrow_tails._plt_get_edge_color()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._arrow_tails._plt_set_edge_color(color)
        self._arrow_heads.setPens(self._arrow_tails._qpens)

    def _plt_get_antialias(self) -> bool:
        return True

    def _plt_set_antialias(self, antialias: bool):
        pass
