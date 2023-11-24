from __future__ import annotations
import numpy as np

from qtpy import QtGui
import pyqtgraph as pg

from whitecanvas.types import Alignment, FacePattern, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from ._qt_utils import (
    from_qt_brush_style,
    from_qt_line_style,
    to_qt_brush_style,
    to_qt_line_style,
    array_to_qcolor,
)


@check_protocol(TextProtocol)
class Text(pg.TextItem, PyQtLayer):
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
        self.setAnchor((xanc, yanc))
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
