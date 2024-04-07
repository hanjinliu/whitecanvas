from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from bokeh.models import FactorRange
from cmap import Color

from whitecanvas.backend.bokeh._base import to_bokeh_line_style
from whitecanvas.types import AxisScale, LineStyle

if TYPE_CHECKING:
    from bokeh.models import Axis as BokehAxis
    from bokeh.models import DataRange1d
    from bokeh.models import Grid as BokehGrid

    from whitecanvas.backend.bokeh.canvas import Canvas


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
        return self._canvas()._get_xaxis()

    def _plt_get_range(self) -> DataRange1d:
        return self._canvas()._get_xrange()


class Y:
    def _plt_get_axis(self) -> BokehAxis:
        return self._canvas()._get_yaxis()

    def _plt_get_range(self) -> DataRange1d:
        return self._canvas()._get_yrange()


class Axis(_CanvasComponent):
    def __init__(self, canvas: Canvas):
        super().__init__(canvas)

    def _plt_get_axis(self) -> BokehAxis:
        raise NotImplementedError

    def _plt_get_visible(self) -> bool:
        return self._plt_get_axis().visible

    def _plt_set_visible(self, visible: bool):
        self._plt_get_axis().visible = visible

    def _plt_get_color(self):
        return np.array(Color(self._plt_get_axis().axis_label_text_color).rgba)

    def _plt_set_color(self, color):
        self._plt_get_axis().axis_label_text_color = Color(color).hex

    def _plt_set_scale(self, scale: AxisScale) -> None:
        raise NotImplementedError("Bokeh does not support dynamically changing scale.")


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
        if not visible:
            self._plt_get_axis().axis_label = None
        else:
            self._plt_get_axis().axis_label = self._text

    def _plt_get_text(self) -> str:
        return self._text

    def _plt_set_text(self, text: str):
        self._text = text
        if self._plt_get_visible():
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
        self._font_size = None
        self._tick_color = None

    def _plt_get_axis(self) -> BokehAxis:
        raise NotImplementedError

    def _plt_get_tick_labels(self) -> tuple[list[float], list[str]]:
        return [], []  # Not implemented on bokeh side

    def _plt_override_labels(self, pos: list[float], labels: list[str]):
        self._plt_get_axis().ticker = pos
        self._plt_get_axis().major_label_overrides = dict(zip(pos, labels))

    def _plt_reset_override(self):
        self._plt_get_axis().ticker = []
        self._plt_get_axis().major_label_overrides = {}

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible
        if not visible:
            self._plt_get_axis().major_label_text_font_size = "0pt"
            self._plt_get_axis().major_tick_line_color = None
            self._plt_get_axis().minor_tick_line_color = None
        else:
            self._plt_get_axis().major_label_text_font_size = self._font_size
            self._plt_get_axis().major_tick_line_color = self._tick_color
            self._plt_get_axis().minor_tick_line_color = self._tick_color

    def _plt_get_size(self) -> float:
        return self._font_size

    def _plt_set_size(self, size: int):
        self._font_size = f"{size}pt"
        if self._visible:
            self._plt_get_axis().major_label_text_font_size = self._font_size

    def _plt_get_fontfamily(self) -> str:
        return self._plt_get_axis().major_label_text_font

    def _plt_set_fontfamily(self, font):
        self._plt_get_axis().major_label_text_font = font

    def _plt_get_color(self):
        return np.array(Color(self._tick_color).rgba)

    def _plt_set_color(self, color):
        self._tick_color = Color(color).hex
        if self._visible:
            self._plt_get_axis().major_label_text_color = self._tick_color
            self._plt_get_axis().major_tick_line_color = self._tick_color
            self._plt_get_axis().minor_tick_line_color = self._tick_color

    def _plt_get_text_rotation(self) -> float:
        return self._plt_get_axis().major_label_orientation

    def _plt_set_text_rotation(self, rotation: float):
        self._plt_get_axis().major_label_orientation = rotation


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
        _range = self._plt_get_range()
        return _range.start, _range.end

    def _plt_set_limits(self, limits: tuple[float, float]):
        _range = self._plt_get_range()
        _range.start, _range.end = limits


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
        limits = self._plt_get_range()
        return limits.start, limits.end

    def _plt_set_limits(self, limits: tuple[float, float]):
        x_range = self._plt_get_range()
        x_range.start, x_range.end = limits


class XLabel(X, Label):
    pass


class YLabel(Y, Label):
    pass


class XTicks(X, Ticks):
    def _plt_set_multilevel_text(self, text: list[tuple[str, ...]]):
        self._canvas()._plot.x_range = FactorRange(*text)


class YTicks(Y, Ticks):
    def _plt_set_multilevel_text(self, text: list[tuple[str, ...]]):
        self._canvas()._plot.y_range = FactorRange(*text)
