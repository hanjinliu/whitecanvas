from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from qtpy import QtGui
import pyqtgraph as pg

from whitecanvas.types import Alignment, FacePattern, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.utils.normalize import as_color_array
from ._qt_utils import (
    from_qt_brush_style,
    from_qt_line_style,
    to_qt_brush_style,
    to_qt_line_style,
    array_to_qcolor,
)


@check_protocol(TextProtocol)
class Texts(pg.ItemGroup, PyQtLayer):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        super().__init__()
        for x0, y0, text0 in zip(x, y, text):
            self.addItem(SingleText(x0, y0, text0))

    if TYPE_CHECKING:

        def childItems(self) -> list[SingleText]:
            ...

    ##### TextProtocol #####

    def _plt_get_text(self) -> list[str]:
        return [t.toPlainText() for t in self.childItems()]

    def _plt_set_text(self, text: list[str]):
        for t, text0 in zip(self.childItems(), text):
            t.setPlainText(text0)

    def _plt_get_text_color(self):
        return np.array([t.color.getRgbF() for t in self.childItems()])

    def _plt_set_text_color(self, color):
        color = as_color_array(color, len(self.childItems()))
        for t, color0 in zip(self.childItems(), color):
            t.setColor(array_to_qcolor(color0))

    def _plt_get_text_size(self) -> float:
        return [t._plt_get_text_size() for t in self.childItems()]

    def _plt_set_text_size(self, size: float | NDArray[np.floating]):
        if isinstance(size, (int, float, np.number)):
            size = np.full(len(self.childItems()), size)
        for t, size0 in zip(self.childItems(), size):
            t._plt_set_text_size(size0)

    def _plt_get_text_position(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        pos = np.array([t.pos() for t in self.childItems()])
        return pos[:, 0], pos[:, 1]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        x, y = position
        for t, x0, y0 in zip(self.childItems(), x, y):
            t.setPos(x0, y0)

    def _plt_get_text_anchor(self) -> list[Alignment]:
        return [t._plt_get_text_anchor() for t in self.childItems()]

    def _plt_set_text_anchor(self, anc: Alignment | list[Alignment]):
        if isinstance(anc, Alignment):
            anc = [anc] * len(self.childItems())
        for t, anc0 in zip(self.childItems(), anc):
            t._plt_set_text_anchor(anc0)

    def _plt_get_text_rotation(self) -> float:
        return [t.angle for t in self.childItems()]

    def _plt_set_text_rotation(self, rotation: float):
        if isinstance(rotation, (int, float, np.number)):
            rotation = np.full(len(self.childItems()), rotation)
        for t, rotation0 in zip(self.childItems(), rotation):
            t.setAngle(rotation0)

    def _plt_get_text_fontfamily(self) -> list[str]:
        return [t._get_qfont().family() for t in self.childItems()]

    def _plt_set_text_fontfamily(self, fontfamily: str | list[str]):
        if isinstance(fontfamily, str):
            fontfamily = [fontfamily] * len(self.childItems())
        for t, fontfamily0 in zip(self.childItems(), fontfamily):
            font = t._get_qfont()
            font.setFamily(fontfamily0)
            t.setFont(font)

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.array([t._get_brush().color().getRgbF() for t in self.childItems()])

    def _plt_set_face_color(self, color):
        color = as_color_array(color, len(self.childItems()))
        for t, color0 in zip(self.childItems(), color):
            brush = t._get_brush()
            brush.setColor(array_to_qcolor(color0))
            t.fill = brush

    def _plt_get_face_pattern(self) -> list[FacePattern]:
        return from_qt_brush_style(self._get_brush().style())

    def _plt_set_face_pattern(self, pattern: FacePattern | list[FacePattern]):
        if isinstance(pattern, FacePattern):
            pattern = [pattern] * len(self.childItems())
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
        color = as_color_array(color, len(self.childItems()))
        for t, color0 in zip(self.childItems(), color):
            pen = t._get_pen()
            pen.setColor(array_to_qcolor(color0))
            t.border = pen

    def _plt_get_edge_width(self) -> float:
        return np.array(
            [t._get_pen().widthF() for t in self.childItems()], dtype=np.float32
        )

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if isinstance(width, (int, float, np.number)):
            width = np.full(len(self.childItems()), width)
        for t, width0 in zip(self.childItems(), width):
            pen = t._get_pen()
            pen.setWidthF(width0)
            t.border = pen

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return from_qt_line_style(self._get_pen().style())

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * len(self.childItems())
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

    def _plt_get_text(self) -> str:
        return self.toPlainText()

    def _plt_set_text(self, text: str):
        self.setPlainText(text)

    def _plt_get_text_color(self):
        return np.array(self.color.getRgbF())

    def _plt_set_text_color(self, color):
        self.setColor(array_to_qcolor(color))

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

    def _plt_get_text_rotation(self) -> float:
        return self.angle

    def _plt_set_text_rotation(self, rotation: float):
        self.setAngle(rotation)

    def _plt_get_text_fontfamily(self) -> str:
        return self._get_qfont().family()

    def _plt_set_text_fontfamily(self, fontfamily: str):
        font = self._get_qfont()
        font.setFamily(fontfamily)
        self.setFont(font)

    ##### HasFaces #####

    def _get_brush(self) -> QtGui.QBrush:
        return self.fill

    def _get_pen(self) -> QtGui.QPen:
        return self.border

    def _plt_get_face_color(self):
        return np.array(self._get_brush().color().getRgbF())

    def _plt_set_face_color(self, color):
        brush = self._get_brush()
        brush.setColor(array_to_qcolor(color))
        self.fill = brush

    def _plt_get_face_pattern(self) -> FacePattern:
        return from_qt_brush_style(self._get_brush().style())

    def _plt_set_face_pattern(self, pattern: FacePattern):
        brush = self._get_brush()
        brush.setStyle(to_qt_brush_style(pattern))
        self.fill = brush

    ##### HasEdges #####

    def _plt_get_edge_color(self):
        return np.array(self._get_pen().color().getRgbF())

    def _plt_set_edge_color(self, color):
        pen = self._get_pen()
        pen.setColor(array_to_qcolor(color))
        self.border = pen

    def _plt_get_edge_width(self) -> float:
        return self._get_pen().widthF()

    def _plt_set_edge_width(self, width: float):
        pen = self._get_pen()
        pen.setWidthF(width)
        self.border = pen

    def _plt_get_edge_style(self) -> LineStyle:
        return from_qt_line_style(self._get_pen().style())

    def _plt_set_edge_style(self, style: LineStyle):
        pen = self._get_pen()
        pen.setStyle(to_qt_line_style(style))
        self.border = pen


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
