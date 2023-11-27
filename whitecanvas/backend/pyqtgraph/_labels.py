from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
import numpy as np
from qtpy import QtCore
from qtpy.QtGui import QFont, QPen
from cmap import Color
from whitecanvas.types import LineStyle
from ._qt_utils import array_to_qcolor

if TYPE_CHECKING:
    from qtpy import QtWidgets as QtW
    import pyqtgraph as pg
    from .canvas import Canvas


class _CanvasComponent:
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)


class Title(_CanvasComponent):
    def _plt_get_visible(self) -> bool:
        return self._canvas()._plot_item.titleLabel.isVisible()

    def _plt_set_visible(self, visible: bool):
        self._canvas()._plot_item.titleLabel.setVisible(visible)

    def _plt_get_text(self) -> str:
        return self._canvas()._plot_item.titleLabel.text

    def _plt_set_text(self, text: str):
        self._canvas()._plot_item.setTitle(text)

    def _plt_get_color(self):
        return self._canvas()._plot_item.titleLabel.opts["color"]

    def _plt_set_color(self, color):
        self._canvas()._plot_item.setTitle(self._plt_get_text(), color=color)

    def _plt_get_size(self) -> int:
        pt = self._canvas()._plot_item.titleLabel.opts["size"]
        return int(pt[:-2])

    def _plt_set_size(self, size: int):
        self._canvas()._plot_item.setTitle(self._plt_get_text(), size=f"{size}pt")

    def _plt_get_fontfamily(self) -> str:
        return self._canvas()._plot_item.titleLabel.item.font().family()

    def _plt_set_fontfamily(self, family: str):
        self._canvas()._plot_item.titleLabel.item.setFont(
            QFont(family, self._plt_get_size())
        )


class AxisLabel(_CanvasComponent):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._css = {
            "color": "#FFFFFF",
            "font-size": "11pt",
            "font-family": "Arial",
        }
        self._axis = axis

    def _get_axis(self) -> pg.AxisItem:
        self._canvas()._plot_item.getAxis(self._axis)

    def _get_label(self) -> QtW.QGraphicsTextItem:
        return self._get_axis().label

    def _plt_get_visible(self) -> bool:
        return self._get_label().isVisible()

    def _plt_set_visible(self, visible: bool):
        self._get_axis().showLabel(visible)

    def _plt_get_text(self) -> str:
        return self._get_axis().labelText

    def _plt_set_text(self, text: str):
        self._get_axis().setLabel(text, **self._css)

    def _plt_get_color(self):
        return np.array(Color(self._css["color"]))

    def _plt_set_color(self, color):
        css = self._css.copy()
        css["color"] = Color(color).hex
        self._get_axis().setLabel(self._plt_get_text(), **css)

    def _plt_get_size(self) -> int:
        return int(self._css["font-size"][:-2])

    def _plt_set_size(self, size: int):
        css = self._css.copy()
        css["font-size"] = f"{size}pt"
        self._get_axis().setLabel(self._plt_get_text(), **css)

    def _plt_get_fontfamily(self) -> str:
        return self._css["font-family"]

    def _plt_set_fontfamily(self, family: str):
        css = self._css.copy()
        css["font-family"] = family
        self._get_axis().setLabel(self._plt_get_text(), **css)


class Axis(_CanvasComponent):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis

    def _plt_get_axis(self) -> pg.AxisItem:
        return self._canvas()._plot_item.getAxis(self._axis)

    def _plt_get_limits(self) -> tuple[float, float]:
        return self._plt_get_axis().range

    def _plt_set_limits(self, limits: tuple[float, float]):
        if self._axis == "bottom":
            self._canvas()._viewbox().setXRange(*limits, padding=0)
        else:
            self._canvas()._viewbox().setYRange(*limits, padding=0)

    def _plt_get_color(self):
        return np.array(self._plt_get_axis().textPen().color().toRgbF())

    def _plt_set_color(self, color):
        pen = QPen(array_to_qcolor(color))
        pen.setCosmetic(True)
        self._plt_get_axis().setTextPen(pen)
        self._plt_get_axis().setPen(pen)

    def _plt_flip(self) -> None:
        viewbox: pg.ViewBox = self._plt_get_axis().linkedView()
        if self._axis == "bottom":
            viewbox.invertX()
        elif self._axis == "left":
            viewbox.invertY()

    def _plt_set_grid_state(self, visible: bool, color, width: float, style: LineStyle):
        if visible:
            grid = color[3] * 255
        else:
            grid = False
        axis = self._plt_get_axis()
        axis.setGrid(grid)
        # tick disappears by unknown reason.


class Ticks(_CanvasComponent):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis
        self._pen = QPen(array_to_qcolor(np.array([0.0, 0.0, 0.0, 1.0])))
        self._visible = True
        self._plt_get_axis().setTickFont(QFont("sans-serif"))  # avoid None

    def _plt_get_axis(self) -> pg.AxisItem:
        return self._canvas()._plot_item.getAxis(self._axis)

    def _plt_get_text(self) -> tuple[list[float], list[str]]:
        axis = self._plt_get_axis()
        if axis._tickLevels is not None:
            major: list[tuple[float, str]] = axis._tickLevels[0]
            values = []
            strings = []
            for val, string in major:
                values.append(val)
                strings.append(string)
        else:
            bounds = axis.mapRectFromParent(axis.geometry())
            span = (bounds.topLeft(), bounds.topRight())
            p0: QtCore.QPointF = axis.mapToDevice(span[0])
            p1: QtCore.QPointF = axis.mapToDevice(span[1])
            length_in_pixel = np.hypot(p0.x() - p1.x(), p0.y() - p1.y())
            tickvals = axis.tickValues(*axis.range, length_in_pixel)
            values = []
            for tickval in tickvals:
                values += list(tickval[1])
            values.sort()
            strings = values
        return (values, strings)

    def _plt_set_text(self, text: tuple[list[float], list[str]]):
        axis = self._plt_get_axis()
        axis.setTicks([[(pos, label) for pos, label in zip(*text)]])

    def _plt_reset_text(self):
        self._plt_get_axis().setTicks(None)

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        axis = self._plt_get_axis()
        if visible:
            pen = self._pen
        else:
            pen = QPen(array_to_qcolor(np.zeros(4)))
        axis.setTickPen(pen)
        axis.setTextPen(pen)
        self._visible = visible

    def _get_font(self) -> QFont:
        return self._plt_get_axis().style['tickFont']

    def _plt_get_size(self) -> float:
        return self._get_font().pointSizeF()

    def _plt_set_size(self, size: str):
        font = self._get_font()
        font.setPointSizeF(size)
        self._plt_get_axis().setTickFont(font)

    def _plt_get_fontfamily(self) -> str:
        return self._get_font().family()

    def _plt_set_fontfamily(self, font):
        qfont = self._get_font()
        qfont.setFamily(font)
        self._plt_get_axis().setTickFont(qfont)

    def _plt_get_color(self):
        return np.array(self._pen.color().getRgbF())

    def _plt_set_color(self, color):
        pen = self._pen
        pen.setColor(array_to_qcolor(color))
        self._plt_get_axis().setTickPen(pen)
        self._plt_get_axis().setTextPen(pen)
