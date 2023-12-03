from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
import numpy as np
from cmap import Color
from whitecanvas.types import LineStyle
from ._base import to_bokeh_line_style

if TYPE_CHECKING:
    from .canvas import Canvas
    from bokeh.models import Axis as BokehAxis, Grid as BokehGrid, DataRange1d


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

    def _plt_get_axis(self) -> BokehAxis:
        raise NotImplementedError

    def _plt_get_visible(self) -> bool:
        return self._plot.xaxis.visible

    def _plt_set_visible(self, visible: bool):
        self._plot.xaxis.visible = visible

    def _plt_get_color(self):
        return np.array(Color(self._plt_get_axis().axis_label_text_color).rgba)

    def _plt_set_color(self, color):
        self._plt_get_axis().axis_label_text_color = Color(color).hex


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
        return tuple(zip(*self._plt_get_axis().ticker))

    def _plt_set_text(self, text: tuple[list[float], list[str]]):
        pos, labels = text
        self._plt_get_axis().ticker = pos
        self._plt_get_axis().major_label_overrides = dict(zip(pos, labels))

    def _plt_reset_text(self):
        self._plt_get_axis().ticker = None
        self._plt_get_axis().major_label_overrides = None

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
    def _plt_set_grid_state(self, visible: bool, color, width: float, style: LineStyle):
        grid: BokehGrid = self._canvas()._plot.xgrid
        grid.visible = visible
        grid.grid_line_color = Color(color).hex
        grid.grid_line_width = width
        grid.grid_line_dash = to_bokeh_line_style(style)

    def _plt_flip(self) -> None:
        fig = self._canvas()._plot
        fig.x_range.flipped = not fig.x_range.flipped

    def _plt_get_limits(self) -> tuple[float, float]:
        limits: DataRange1d = self._canvas()._plot.x_range
        return limits.start, limits.end

    def _plt_set_limits(self, limits: tuple[float, float]):
        x_range: DataRange1d = self._canvas()._plot.x_range
        x_range.start, x_range.end = limits


class YAxis(Y, Axis):
    def _plt_set_grid_state(self, visible: bool, color, width: float, style: LineStyle):
        grid: BokehGrid = self._canvas()._plot.ygrid
        grid.visible = visible
        grid.grid_line_color = Color(color).hex
        grid.grid_line_width = width
        grid.grid_line_dash = to_bokeh_line_style(style)

    def _plt_flip(self) -> None:
        fig = self._canvas()._plot
        fig.y_range.flipped = not fig.y_range.flipped

    def _plt_get_limits(self) -> tuple[float, float]:
        limits: DataRange1d = self._canvas()._plot.y_range
        return limits.start, limits.end

    def _plt_set_limits(self, limits: tuple[float, float]):
        x_range: DataRange1d = self._canvas()._plot.y_range
        x_range.start, x_range.end = limits


class XLabel(X, Label):
    pass


class YLabel(Y, Label):
    pass


class XTicks(X, Ticks):
    pass


class YTicks(Y, Ticks):
    pass
