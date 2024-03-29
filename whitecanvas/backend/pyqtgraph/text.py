from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy import QtGui

from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._qt_utils import (
    array_to_qcolor,
    from_qt_brush_style,
    from_qt_line_style,
    to_qt_brush_style,
    to_qt_line_style,
)
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.types import Alignment, Hatch, LineStyle
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number


@check_protocol(TextProtocol)
class Texts(pg.ItemGroup, PyQtLayer):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        super().__init__()
        for x0, y0, text0 in zip(x, y, text):
            self.addItem(SingleText(x0, y0, text0))
        self._font_family = "Arial"
        self._align = Alignment.BOTTOM_LEFT

    if TYPE_CHECKING:

        def childItems(self) -> list[SingleText]:
            ...

    ##### TextProtocol #####

    def _plt_get_text(self) -> list[str]:
        return [t.toPlainText() for t in self.childItems()]

    def _plt_set_text(self, text: list[str]):
        for t, text0 in zip(self.childItems(), text):
            t.setPlainText(text0)

    def _plt_get_ndata(self) -> int:
        return len(self.childItems())

    def _plt_get_text_color(self):
        return np.array([t.color.getRgbF() for t in self.childItems()])

    def _plt_set_text_color(self, color):
        color = as_color_array(color, self._plt_get_ndata())
        for t, color0 in zip(self.childItems(), color):
            t.setColor(array_to_qcolor(color0))

    def _plt_get_text_size(self) -> float:
        return [t._plt_get_text_size() for t in self.childItems()]

    def _plt_set_text_size(self, size: float | NDArray[np.floating]):
        if is_real_number(size):
            size = np.full(self._plt_get_ndata(), size)
        for t, size0 in zip(self.childItems(), size):
            t._plt_set_text_size(size0)

    def _plt_get_text_position(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        pos = np.array([t.pos() for t in self.childItems()])
        if pos.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        return pos[:, 0], pos[:, 1]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        x, y = position
        if x.size > self._plt_get_ndata():
            for _ in range(x.size - self._plt_get_ndata()):
                self.addItem(SingleText(0, 0, ""))
        elif x.size < self._plt_get_ndata():
            for _ in range(self._plt_get_ndata() - x.size):
                self.childItems()[-1].setParentItem(None)
        for t, x0, y0 in zip(self.childItems(), x, y):
            t.setPos(x0, y0)

    def _plt_get_text_anchor(self) -> list[Alignment]:
        return self._align

    def _plt_set_text_anchor(self, anc: Alignment):
        for t in self.childItems():
            t._plt_set_text_anchor(anc)
        self._align = anc

    def _plt_get_text_rotation(self) -> float:
        return [t.angle for t in self.childItems()]

    def _plt_set_text_rotation(self, rotation: float):
        if is_real_number(rotation):
            rotation = np.full(self._plt_get_ndata(), rotation)
        for t, rotation0 in zip(self.childItems(), rotation):
            t.setAngle(rotation0)

    def _plt_get_text_fontfamily(self) -> str:
        return self._font_family

    def _plt_set_text_fontfamily(self, fontfamily: str):
        for t in self.childItems():
            font = t._get_qfont()
            font.setFamily(fontfamily)
            t.setFont(font)
        self._font_family = fontfamily

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.array([t._get_brush().color().getRgbF() for t in self.childItems()])

    def _plt_set_face_color(self, color):
        color = as_color_array(color, self._plt_get_ndata())
        for t, color0 in zip(self.childItems(), color):
            brush = t._get_brush()
            brush.setColor(array_to_qcolor(color0))
            t.fill = brush

    def _plt_get_face_hatch(self) -> list[Hatch]:
        return [from_qt_brush_style(s._get_brush().style()) for s in self.childItems()]

    def _plt_set_face_hatch(self, pattern: Hatch | list[Hatch]):
        if isinstance(pattern, Hatch):
            pattern = [pattern] * self._plt_get_ndata()
        for t, pattern0 in zip(self.childItems(), pattern):
            brush = t._get_brush()
            brush.setStyle(to_qt_brush_style(pattern0))
            t.fill = brush

    ##### HasEdges #####

    def _plt_get_edge_color(self):
        return np.array(
            [t._get_pen().color().getRgbF() for t in self.childItems()],
            dtype=np.float32,
        )

    def _plt_set_edge_color(self, color):
        color = as_color_array(color, self._plt_get_ndata())
        for t, color0 in zip(self.childItems(), color):
            pen = t._get_pen()
            pen.setColor(array_to_qcolor(color0))
            t.border = pen

    def _plt_get_edge_width(self) -> float:
        return np.array(
            [t._get_pen().widthF() for t in self.childItems()], dtype=np.float32
        )

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if is_real_number(width):
            width = np.full(self._plt_get_ndata(), width)
        for t, width0 in zip(self.childItems(), width):
            pen = t._get_pen()
            pen.setWidthF(width0)
            t.border = pen

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_qt_line_style(s._get_pen().style()) for s in self.childItems()]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * self._plt_get_ndata()
        for t, style0 in zip(self.childItems(), style):
            pen = t._get_pen()
            pen.setStyle(to_qt_line_style(style0))
            t.border = pen


class SingleText(pg.TextItem, PyQtLayer):
    color: QtGui.QColor

    def __init__(self, x: float, y: float, text: str):
        super().__init__(text=text, anchor=(0, 0))
        self._plt_set_text_position([x, y])
        self._alignment = Alignment.BOTTOM_LEFT

    ##### TextProtocol #####

    def _get_qfont(self) -> QtGui.QFont:
        return self.textItem.font()

    def _plt_get_text_size(self) -> float:
        return self._get_qfont().pointSize()

    def _plt_set_text_size(self, size: float):
        font = self._get_qfont()
        font.setPointSizeF(size)
        self.setFont(font)

    def _plt_get_text_position(self) -> tuple[float, float]:
        return self.pos()

    def _plt_set_text_position(self, position: tuple[float, float]):
        self.setPos(*position)

    def _plt_get_text_anchor(self) -> Alignment:
        return self._alignment

    def _plt_set_text_anchor(self, anc: Alignment):
        self.setAnchor(_split_anchor(anc))
        self._alignment = anc

    def _get_brush(self) -> QtGui.QBrush:
        return self.fill

    def _get_pen(self) -> QtGui.QPen:
        return self.border


@lru_cache(maxsize=9)
def _split_anchor(anc: Alignment):
    va, ha = anc.split()
    if va is Alignment.TOP:
        yanc = 0
    elif va is Alignment.BOTTOM:
        yanc = 1
    else:
        yanc = 0.5
    if ha is Alignment.LEFT:
        xanc = 0
    elif ha is Alignment.RIGHT:
        xanc = 1
    else:
        xanc = 0.5
    return xanc, yanc
