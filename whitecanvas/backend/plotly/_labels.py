from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from cmap import Color

from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import rgba_str_color

if TYPE_CHECKING:
    from plotly.graph_objs.layout import Title as PlotlyTitle

    from whitecanvas.backend.plotly.canvas import Canvas


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
    def __init__(self, canvas: Canvas):
        super().__init__(canvas)
        canvas._fig.layout.title = {"text": "", "x": 0.5, "xanchor": "center"}

    def _get_title(self):
        return self._canvas()._fig.layout.title


class AxisLabel(_SupportsTitle):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis

    def _get_title(self):
        layout = self._canvas()._subplot_layout()
        return getattr(layout, self._axis).title


class AxisLabel3D(AxisLabel):
    def _get_title(self):
        layout = self._canvas()._subplot_layout()
        return getattr(layout.scene, self._axis).title


class Axis(_CanvasComponent):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis

    @property
    def name(self) -> str:
        return self._axis

    def _plt_get_axis(self):
        return getattr(self._canvas()._subplot_layout(), self._axis)

    def _plt_get_limits(self) -> tuple[float, float]:
        lim = self._plt_get_axis().range
        if lim is None:
            lim = (0, 1)  # TODO: how to get the limits?
        return lim

    def _plt_set_limits(self, limits: tuple[float, float]):
        self._plt_get_axis().range = limits

    def _plt_get_color(self):
        # color of the axis itself
        return np.fromiter(Color(self._plt_get_axis().linecolor), dtype=np.float32)

    def _plt_set_color(self, color):
        self._plt_get_axis().linecolor = rgba_str_color(color)

    def _plt_flip(self) -> None:
        if self._plt_get_axis().autorange is None:
            self._plt_get_axis().autorange = "reversed"
        else:
            self._plt_get_axis().autorange = None

    def _plt_set_grid_state(self, visible: bool, color, width: float, style: LineStyle):
        axis = self._plt_get_axis()
        axis.showgrid = visible
        axis.gridcolor = rgba_str_color(color)
        axis.gridwidth = width


class Axis3D(Axis):
    def _plt_get_axis(self):
        return getattr(self._canvas()._subplot_layout().scene, self._axis)


class Ticks(_CanvasComponent):
    def __init__(self, canvas: Canvas, axis: str):
        super().__init__(canvas)
        self._axis = axis
        self._visible = True

    def _plt_get_axis(self):
        layout = self._canvas()._subplot_layout()
        return getattr(layout, self._axis)

    def _plt_get_tick_labels(self) -> tuple[list[float], list[str]]:
        texts = self._plt_get_axis().ticktext
        if texts is None:
            return [], []
        return self._plt_get_axis().tickvals, list(texts)

    def _plt_override_labels(self, pos: list[float], labels: list[str]):
        self._plt_get_axis().tickvals = pos
        self._plt_get_axis().ticktext = labels

    def _plt_reset_override(self):
        self._plt_get_axis().tickvals = None
        self._plt_get_axis().ticktext = None

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
        return np.fromiter(Color(self._plt_get_axis().tickfont.color), dtype=np.float32)

    def _plt_set_color(self, color):
        self._plt_get_axis().tickfont.color = rgba_str_color(color)

    def _plt_get_text_rotation(self) -> float:
        return -self._plt_get_axis().tickangle

    def _plt_set_text_rotation(self, angle: float):
        self._plt_get_axis().tickangle = -angle


class Ticks3D(Ticks):
    def _plt_get_axis(self):
        layout = self._canvas()._subplot_layout()
        return getattr(layout.scene, self._axis)
