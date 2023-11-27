from __future__ import annotations
from typing import TYPE_CHECKING
import weakref

from vispy import scene
from vispy.visuals.axis import AxisVisual, Ticker
import numpy as np
from whitecanvas.types import LineStyle

if TYPE_CHECKING:
    from vispy.scene.cameras import PanZoomCamera
    from vispy.visuals import TextVisual
    from .canvas import Canvas


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
        return self._text_visual.color

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
    def __init__(self, canvas: Canvas, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.unfreeze()
        self._dim = dim
        self.freeze()
        self._canvas = weakref.ref(canvas)

    def _plt_viewbox(self) -> scene.ViewBox:
        return self._linked_view

    def _plt_camera(self) -> PanZoomCamera:
        return self._plt_viewbox().camera

    def _plt_get_limits(self) -> tuple[float, float]:
        rect = self._plt_camera().rect
        if self._dim == 0:  # y
            return rect.bottom, rect.top
        else:
            return rect.left, rect.right

    def _plt_set_limits(self, limits: tuple[float, float]):
        camera = self._plt_camera()
        rect = camera.rect
        if self._dim == 0:  # y
            camera.set_range(x=(rect.left, rect.right), y=limits, margin=0)
        else:
            camera.set_range(x=limits, y=(rect.bottom, rect.top), margin=0)
        camera.set_default_state()
        camera.reset()

    def _plt_get_color(self):
        return self.axis.axis_color

    def _plt_set_color(self, color):
        self.axis.axis_color = color

    def _plt_flip(self) -> None:
        camera = self._plt_camera()
        flipped = list(camera.flip)
        flipped[self._dim] = not flipped[self._dim]
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

    @property
    def _text(self) -> TextVisual:
        return self._axis().axis._text

    def _plt_get_text(self) -> list[tuple[float, str]]:
        axis = self._axis().axis
        return list(axis._ticks.pos), self._text.text

    def _plt_set_text(self, ticks: tuple[list[float], list[str]]):
        self._axis().axis.ticker._categorical_labels = ticks

    def _plt_reset_text(self):
        self._axis().axis.ticker._categorical_labels = None

    def _plt_get_color(self):
        return np.array(self._text.color)

    def _plt_set_color(self, color):
        self._text.color = color
        self._axis().axis.tick_color = color

    def _plt_get_visible(self) -> bool:
        return self._axis().visible

    def _plt_set_visible(self, visible: bool):
        self._axis().visible = visible

    def _plt_get_size(self) -> float:
        return self._text.font_size

    def _plt_set_size(self, size: str):
        self._text.font_size = size

    def _plt_get_fontfamily(self) -> str:
        return self._text.face

    def _plt_set_fontfamily(self, font):
        self._text.face = font


class VispyTicker(Ticker):
    """Ticker that supports categorical axes"""

    axis: AxisVisual

    def __init__(self, axis: AxisVisual):
        anchors = axis.ticker._anchors
        super().__init__(axis, anchors)
        self._categorical_labels: tuple[list[float], list[str]] | None = None

    def _get_tick_frac_labels(self):
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
