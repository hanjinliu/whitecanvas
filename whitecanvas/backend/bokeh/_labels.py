from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
import numpy as np
from cmap import Color

if TYPE_CHECKING:
    from .canvas import Canvas
    from bokeh.models import Axis as BokehAxis, Label as BokehLabel


class _CanvasComponent:
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)

    @property
    def _plot(self):
        return self._canvas()._plot


class Title(_CanvasComponent):
    def _plt_get_visible(self) -> bool:
        return self._plot.title.visible

    def _plt_set_visible(self, visible: bool):
        self._plot.title.visible = visible

    def _plt_get_text(self) -> str:
        return self._plot.title.text

    def _plt_set_text(self, text: str):
        self._plot.title.text = text

    def _plt_get_color(self):
        return np.array(Color(self._plot.title.text_color).rgba)

    def _plt_set_color(self, color):
        self._plot.title.text_color = Color(color).hex

    def _plt_get_size(self) -> int:
        return int(self._plot.title.text_font_size.rstrip("pt"))

    def _plt_set_size(self, size: int):
        self._plot.title.text_font_size = f"{size}pt"

    def _plt_get_fontfamily(self) -> str:
        return self._plot.title.text_font

    def _plt_set_fontfamily(self, family: str):
        self._plot.title.text_font = family


class X:
    def _plt_get_axis(self) -> BokehAxis:
        return self._plot.xaxis


class Y:
    def _plt_get_axis(self) -> BokehAxis:
        return self._plot.yaxis


class Axis(_CanvasComponent):
    def __init__(self, canvas: Canvas):
        super().__init__(canvas)
        self._flipped = False

    def _plt_get_axis(self) -> BokehAxis:
        raise NotImplementedError

    def _plt_get_visible(self) -> bool:
        return self._plot.xaxis.visible

    def _plt_set_visible(self, visible: bool):
        self._plot.xaxis.visible = visible

    def _plt_get_limits(self) -> tuple[float, float]:
        return self._plt_get_axis().bounds

    def _plt_set_limits(self, limits: tuple[float, float]):
        if self._flipped:
            limits = limits[::-1]
        self._plt_get_axis().bounds = limits

    def _plt_get_ticks(self) -> list[tuple[float, str]]:
        return self._plt_get_axis().ticker.ticks

    def _plt_set_ticks(self, ticks: list[tuple[float, str]]):
        self._plt_get_axis().ticker.ticks = ticks

    def _plt_get_color(self):
        return np.array(Color(self._plt_get_axis().axis_label_text_color).rgba)

    def _plt_set_color(self, color):
        self._plt_get_axis().axis_label_text_color = Color(color).hex

    def _plt_flip(self) -> None:
        self._plt_set_limits(self._plt_get_limits()[::-1])
        self._flipped = not self._flipped


class Label(_CanvasComponent):
    def __init__(self, canvas: Canvas):
        super().__init__(canvas)
        self._text = ""
        self._visible = True

    def _plt_get_axis(self) -> BokehAxis:
        raise NotImplementedError

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible
        self._plt_get_axis().axis_label = None

    def _plt_get_text(self) -> str:
        return self._text

    def _plt_set_text(self, text: str):
        self._text = text
        self._plt_get_axis().axis_label = text

    def _plt_get_color(self):
        return np.array(Color(self._plt_get_axis().axis_label_text_color).rgba)

    def _plt_set_color(self, color):
        self._plt_get_axis().axis_label_text_color = Color(color).hex

    def _plt_get_size(self) -> int:
        return int(self._plt_get_axis().axis_label_text_font_size.rstrip("pt"))

    def _plt_set_size(self, size: int):
        self._plt_get_axis().axis_label_text_font_size = f"{size}pt"

    def _plt_get_fontfamily(self) -> str:
        return self._plt_get_axis().axis_label_text_font

    def _plt_set_fontfamily(self, family: str):
        self._plt_get_axis().axis_label_text_font = family


class Ticks(_CanvasComponent):
    def __init__(self, canvas: Canvas):
        super().__init__(canvas)
        self._visible = True

    def _plt_get_axis(self) -> BokehAxis:
        raise NotImplementedError

    def _plt_get_text(self) -> tuple[list[float], list[str]]:
        return tuple(zip(*self._plt_get_axis().ticker.ticks))

    def _plt_set_text(self, text: tuple[list[float], list[str]]):
        self._plt_get_axis().ticker.ticks = text

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible
        self._plt_get_axis().ticker = None

    def _plt_get_size(self) -> float:
        return self._plt_get_axis().major_label_text_font_size

    def _plt_set_size(self, size: int):
        self._plt_get_axis().major_label_text_font_size = f"{size}pt"

    def _plt_get_fontfamily(self) -> str:
        return self._plt_get_axis().major_label_text_font

    def _plt_set_fontfamily(self, font):
        self._plt_get_axis().major_label_text_font = font

    def _plt_get_color(self):
        return np.array(Color(self._plt_get_axis().major_label_text_color).rgba)

    def _plt_set_color(self, color):
        self._plt_get_axis().major_label_text_color = Color(color).hex


class XAxis(X, Axis):
    pass


class YAxis(Y, Axis):
    pass


class XLabel(X, Label):
    pass


class YLabel(Y, Label):
    pass


class XTicks(X, Ticks):
    pass


class YTicks(Y, Ticks):
    pass
