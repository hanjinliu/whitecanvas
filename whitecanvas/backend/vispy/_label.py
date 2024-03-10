from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from vispy import scene
from vispy.visuals.axis import AxisVisual, Ticker

from whitecanvas.types import LineStyle

if TYPE_CHECKING:
    from vispy.visuals import LineVisual, TextVisual

    from whitecanvas.backend.vispy.canvas import Camera, Canvas

FONT_SIZE_FACTOR = 2.0


class TextLabel(scene.Label):
    _text_visual: TextVisual

    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    def _plt_get_text(self) -> str:
        return self.text

    def _plt_set_text(self, text: str):
        self.text = text

    def _plt_get_color(self):
        return np.array(self._text_visual.color).ravel()

    def _plt_set_color(self, color):
        self._text_visual.color = color

    def _plt_get_size(self) -> int:
        return self._text_visual.font_size * FONT_SIZE_FACTOR

    def _plt_set_size(self, size: int):
        self._text_visual.font_size = size / FONT_SIZE_FACTOR

    def _plt_get_fontfamily(self) -> str:
        return self._text_visual.face

    def _plt_set_fontfamily(self, family: str):
        self._text_visual.face = family


class Axis(scene.AxisWidget):
    axis: AxisVisual

    def __init__(self, canvas: Canvas, dim: int, **kwargs):
        kwargs.setdefault("axis_width", 2)
        kwargs.setdefault("tick_width", 1)
        super().__init__(**kwargs)
        self.unfreeze()
        self._dim = dim
        self._canvas_ref = weakref.ref(canvas)
        self.freeze()

    def _plt_viewbox(self) -> scene.ViewBox:
        return self._canvas_ref()._viewbox

    def _plt_camera(self) -> Camera:
        return self._plt_viewbox().camera

    def _plt_get_limits(self) -> tuple[float, float]:
        rect = self._plt_camera().rect
        if self._dim == 0:  # y
            return rect.bottom, rect.top
        else:
            return rect.left, rect.right

    def _plt_set_limits(self, limits: tuple[float, float]):
        camera = self._plt_camera()
        # NOTE: margin = 0 is ignored in the current implementation of vispy.
        # use a very small margin instead.
        margin = (limits[1] - limits[0]) * 1e-8
        with camera.changed.blocked():
            if self._dim == 0:  # y
                xlim = self._canvas_ref()._xaxis._plt_get_limits()
                camera.set_range(x=xlim, y=limits, margin=margin)
            else:
                ylim = self._canvas_ref()._yaxis._plt_get_limits()
                camera.set_range(x=limits, y=ylim, margin=margin)
        camera.set_default_state()
        camera.reset()

    def _plt_get_color(self):
        return np.array(self.axis.axis_color).ravel()

    def _plt_set_color(self, color):
        self.axis.axis_color = color

    def _plt_flip(self) -> None:
        camera = self._plt_camera()
        flipped = list(camera.flip)
        axis = 1 - self._dim
        flipped[axis] = not flipped[axis]
        camera.flip = tuple(flipped)

    def _plt_set_grid_state(self, visible: bool, color, width: float, style: LineStyle):
        # if visible:
        #     self._canvas()._gridlines.visible = True
        #     self._canvas()._gridlines._grid_color_fn['color'] = color
        # else:
        #     self._canvas()._gridlines.visible = False
        pass  # TODO: implement this


class Ticks:
    def __init__(self, axis: Axis):
        self._axis = weakref.ref(axis)
        axis.axis.ticker = VispyTicker(axis.axis)

    def _get_ticks(self) -> LineVisual:
        return self._axis().axis._ticks

    def _get_ticker(self) -> VispyTicker:
        return self._axis().axis.ticker

    @property
    def _text(self) -> TextVisual:
        return self._axis().axis._text

    def _plt_get_tick_labels(self) -> tuple[list[float], list[str]]:
        pos = self._get_ticks().pos
        if pos is None:
            return [], []
        return list(pos), self._text.text

    def _plt_override_labels(self, pos: list[float], labels: list[str]):
        self._get_ticker()._categorical_labels = (pos, labels)

    def _plt_reset_override(self):
        self._get_ticker()._categorical_labels = None

    def _plt_get_color(self):
        return np.array(self._text.color)

    def _plt_set_color(self, color):
        self._text.color = color
        self._axis().axis.tick_color = color

    def _plt_get_visible(self) -> bool:
        return self._get_ticker().visible

    def _plt_set_visible(self, visible: bool):
        self._get_ticker().visible = visible

    def _plt_get_size(self) -> float:
        return self._text.font_size * FONT_SIZE_FACTOR

    def _plt_set_size(self, size: float):
        self._text.font_size = size / FONT_SIZE_FACTOR

    def _plt_get_fontfamily(self) -> str:
        return self._text.face

    def _plt_set_fontfamily(self, font):
        self._text.face = font

    def _plt_get_text_rotation(self) -> float:
        return self._text.rotation

    def _plt_set_text_rotation(self, rotation: float):
        self._text.rotation = rotation


class VispyTicker(Ticker):
    """Ticker that supports categorical axes"""

    axis: AxisVisual

    def __init__(self, axis: AxisVisual):
        anchors = axis.ticker._anchors
        super().__init__(axis, anchors)
        self._categorical_labels: tuple[list[float], list[str]] | None = None
        self._visible = True

    def _get_tick_frac_labels(self):
        if not self._visible:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        if self._categorical_labels is None:
            return super()._get_tick_frac_labels()
        pos, labels = self._categorical_labels
        domain = self.axis.domain
        scale = domain[1] - domain[0]
        major_tick_fractions = (np.asarray(pos) - domain[0]) / scale
        minor_tick_fractions = np.zeros(0)
        ok = (0 <= major_tick_fractions) & (major_tick_fractions <= 1)
        tick_labels = np.asarray(labels)[ok]
        return major_tick_fractions[ok], minor_tick_fractions, tick_labels

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        self._visible = visible
        self.axis.update()
        self.axis._update_subvisuals()
