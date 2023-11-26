from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
import numpy as np
from whitecanvas.utils.normalize import rgba_str_color

if TYPE_CHECKING:
    from .canvas import Canvas
    from plotly.graph_objs.layout import Title as PlotlyTitle


class _CanvasComponent:
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)


class _SupportsTitle:
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)
        self._visible = True
        self._text = ""

    def _get_title(self) -> PlotlyTitle:
        raise NotImplementedError

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._get_title().text = self._text
        else:
            self._get_title().text = ""
        self._visible = visible

    def _plt_get_text(self) -> str:
        return self._get_title().text

    def _plt_set_text(self, text: str):
        self._get_title().text = text
        self._text = text

    def _plt_get_color(self):
        return self._get_title().font.color

    def _plt_set_color(self, color):
        self._get_title().font.color = rgba_str_color(color)

    def _plt_get_size(self) -> int:
        return self._get_title().font.size

    def _plt_set_size(self, size: int):
        self._get_title().font.size = size

    def _plt_get_fontfamily(self) -> str:
        return self._get_title().font.family

    def _plt_set_fontfamily(self, family: str):
        self._get_title().font.family = family


class Title(_SupportsTitle):
    def _get_title(self):
        return self._canvas()._fig.layout.title


class AxisLabel(_SupportsTitle):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis

    def _get_title(self):
        return self._canvas()._fig.layout[self._axis + "axis"].title


class Axis(_CanvasComponent):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis

    def _plt_get_axis(self):
        return self._canvas()._fig.layout[self._axis + "axis"]

    def _plt_get_limits(self) -> tuple[float, float]:
        lim = self._plt_get_axis().range
        if lim is None:
            lim = (0, 1)  # TODO: how to get the limits?
        return lim

    def _plt_set_limits(self, limits: tuple[float, float]):
        self._plt_get_axis().range = limits

    def _plt_get_color(self):
        # color of the axis itself
        return np.array(self._plt_get_axis().linecolor)

    def _plt_set_color(self, color):
        self._plt_get_axis().linecolor = rgba_str_color(color)

    def _plt_flip(self) -> None:
        if self._plt_get_axis().autorange is None:
            self._plt_get_axis().autorange = 'reversed'
        else:
            self._plt_get_axis().autorange = None


class Ticks(_CanvasComponent):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis
        self._visible = True

    def _plt_get_axis(self):
        return self._canvas()._fig.layout[self._axis + "axis"]

    def _plt_get_text(self) -> tuple[list[float], list[str]]:
        return (
            self._plt_get_axis().tickvals,
            self._plt_get_axis().ticktext,
        )

    def _plt_set_text(self, text: tuple[list[float], list[str]]):
        self._plt_get_axis().tickvals = text[0]
        self._plt_get_axis().ticktext = text[1]

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._plt_get_axis().showticklabels = visible
        self._visible = visible

    def _plt_get_size(self) -> float:
        return self._plt_get_axis().tickfont.size

    def _plt_set_size(self, size: str):
        self._plt_get_axis().tickfont.size = size

    def _plt_get_fontfamily(self) -> str:
        return self._plt_get_axis().tickfont.family

    def _plt_set_fontfamily(self, font):
        self._plt_get_axis().tickfont.family = font

    def _plt_get_color(self):
        return np.array(self._plt_get_axis().tickfont.color)

    def _plt_set_color(self, color):
        self._plt_get_axis().tickfont.color = rgba_str_color(color)
