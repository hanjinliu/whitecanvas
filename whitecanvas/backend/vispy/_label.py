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
        return np.array(self._text_visual.color.rgba).ravel()

    def _plt_set_color(self, color):
        self._text_visual.color = color

    def _plt_get_size(self) -> int:
        return self._text_visual.font_size

    def _plt_set_size(self, size: int):
        self._text_visual.font_size = size

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
        grid_lines = self._canvas_ref()._grid_lines
        if self._dim == 0:  # y
            grid_lines.set_y_grid_lines(visible, color, width, style)
        else:
            grid_lines.set_x_grid_lines(visible, color, width, style)


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
        return np.array(self._text.color.rgba).ravel()

    def _plt_set_color(self, color):
        self._text.color = color
        self._axis().axis.tick_color = color

    def _plt_get_visible(self) -> bool:
        return self._get_ticker().visible

    def _plt_set_visible(self, visible: bool):
        self._get_ticker().visible = visible
        # axis = self._axis()
        # if axis._dim == 0:  # y
        #     axis.width_min = axis.width_max = 40 if visible else 0
        # else:
        #     axis.height_min = axis.height_max = 50 if visible else 0

    def _plt_get_size(self) -> float:
        return self._text.font_size

    def _plt_set_size(self, size: float):
        self._text.font_size = size

    def _plt_get_fontfamily(self) -> str:
        return self._text.face

    def _plt_set_fontfamily(self, font):
        self._text.face = font

    def _plt_get_text_rotation(self) -> float:
        return -self._text.rotation

    def _plt_set_text_rotation(self, rotation: float):
        self._text.rotation = -rotation


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
            major, minor, labels = np.zeros(0), np.zeros(0), np.zeros(0)
        elif self._categorical_labels is None:
            major, minor, labels = super()._get_tick_frac_labels()
        else:
            pos, labels = self._categorical_labels
            domain = self.axis.domain
            scale = domain[1] - domain[0]
            major_tick_fractions = (np.asarray(pos) - domain[0]) / scale
            minor_tick_fractions = np.zeros(0)
            ok = (0 <= major_tick_fractions) & (major_tick_fractions <= 1)
            tick_labels = np.asarray(labels)[ok]
            major = major_tick_fractions[ok]
            minor = minor_tick_fractions
            labels = tick_labels
        return major, minor, labels

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        self._visible = visible
        self.axis.update()
        self.axis._update_subvisuals()
