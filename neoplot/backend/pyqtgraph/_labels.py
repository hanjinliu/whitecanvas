from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
import numpy as np
from qtpy.QtGui import QFont, QPen
from cmap import Color
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
        self._canvas()._plot_item.titleLabel.item.setFont(QFont(family, self._plt_get_size()))


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
        return tuple(self._plt_get_axis().range)

    def _plt_set_limits(self, limits: tuple[float, float]):
        self._plt_get_axis().setRange(*limits)

    def _plt_set_ticks(self, ticks: list[tuple[float, str]]):
        self._plt_get_axis().setTicks(ticks)

    def _plt_get_color(self):
        return np.array(self._plt_get_axis().textPen().color().toRgbF())

    def _plt_set_color(self, color):
        pen = QPen(array_to_qcolor(color))
        self._plt_get_axis().setTextPen(pen)
        self._plt_get_axis().setPen(pen)
