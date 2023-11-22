from __future__ import annotations
from typing import TYPE_CHECKING
import weakref

from vispy import scene
import numpy as np

if TYPE_CHECKING:
    from vispy.scene.cameras import PanZoomCamera
    from vispy.visuals import TextVisual


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
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.unfreeze()
        self._dim = dim
        self.freeze()

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


class Ticks:
    def __init__(self, axis: Axis):
        self._axis = weakref.ref(axis)

    @property
    def _text(self) -> TextVisual:
        return self._axis().axis._text

    def _plt_get_text(self) -> list[tuple[float, str]]:
        axis = self._axis().axis
        return list(axis._ticks.pos), self._text.text

    def _plt_set_text(self, ticks: list[tuple[float, str]]):
        raise NotImplementedError

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
