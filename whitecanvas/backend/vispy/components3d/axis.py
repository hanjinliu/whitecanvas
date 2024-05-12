from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from vispy import scene
from vispy.scene.visuals import Compound, Line, Text

if TYPE_CHECKING:
    from whitecanvas.backend.vispy.components3d.canvas import Camera, Canvas3D


class Axis3D(Compound):
    def __init__(self, canvas: Canvas3D, dim: int):
        name = "xyz"[dim]
        self._dim = dim
        self._line = Line(method="gl", width=2)
        self._text = Text(
            text=name,
            font_size=10,
            anchor_x="center",
            anchor_y="center",
        )
        self._canvas_ref = weakref.ref(canvas)
        super().__init__([self._line, self._text])
        self._update_pos([0, 0, 0], 1)
        self.parent = canvas._viewbox.scene

    def _plt_viewbox(self) -> scene.ViewBox:
        return self._canvas_ref()._viewbox

    def _plt_camera(self) -> Camera:
        return self._plt_viewbox().camera

    def _update_pos(
        self,
        origin: tuple[float, float, float],
        length: float | None = None,
    ):
        origin = np.asarray(origin)
        dr = np.zeros(3)
        if length is None:
            length = self._length()
        dr[self._dim] = length
        pos = np.stack([origin, origin + dr], axis=0)
        self._line.set_data(pos)
        self._text.pos = origin + dr * 1.1

    def _plt_get_limits(self) -> tuple[float, float]:
        start = self._line.pos[0]
        end = self._line.pos[1]
        return start[self._dim], end[self._dim]

    def _length(self) -> float:
        start = self._line.pos[0]
        end = self._line.pos[1]
        return np.sqrt(np.sum((end - start) ** 2))

    def _plt_set_limits(self, limits: tuple[float, float]):
        new_origin = self._line.pos[0]
        new_origin[self._dim] = limits[0]
        self._update_pos(new_origin, limits[1] - limits[0])
        canvas = self._canvas_ref()
        canvas._xaxis._update_pos(new_origin)
        canvas._yaxis._update_pos(new_origin)
        canvas._zaxis._update_pos(new_origin)

        start = self._line.pos[0]
        vecx = canvas._xaxis._line.pos[1] - start
        vecy = canvas._yaxis._line.pos[1] - start
        vecz = canvas._zaxis._line.pos[1] - start
        sizex = np.sqrt(np.dot(vecx, vecx))
        sizey = np.sqrt(np.dot(vecy, vecy))
        sizez = np.sqrt(np.dot(vecz, vecz))
        center = start + (vecx + vecy + vecz) / 2
        canvas._viewbox.camera.center = center
        canvas._viewbox.camera.scale_factor = max(sizex, sizey, sizez) * 1.5

    def _plt_get_color(self):
        return self._line.color

    def _plt_set_color(self, color):
        self._line.color = color

    def _plt_flip(self) -> None:
        camera = self._plt_camera()
        flipped = list(camera.flip)
        axis = 1 - self._dim
        flipped[axis] = not flipped[axis]
        camera.flip = tuple(flipped)


class AxisLabel3D:
    def __init__(self, axis: Axis3D):
        self._axis = axis

    def _plt_get_visible(self) -> bool:
        return self._axis._text.visible

    def _plt_set_visible(self, visible: bool):
        self._axis._text.visible = visible

    def _plt_get_text(self) -> str:
        return self._axis._text.text

    def _plt_set_text(self, text: str):
        self._axis._text.text = text

    def _plt_get_color(self):
        return np.array(self._axis._text.color).ravel()

    def _plt_set_color(self, color):
        self._axis._text.color = color

    def _plt_get_size(self) -> int:
        return self._axis._text.font_size

    def _plt_set_size(self, size: int):
        self._axis._text.font_size = size

    def _plt_get_fontfamily(self) -> str:
        return self._axis._text.face

    def _plt_set_fontfamily(self, family: str):
        self._axis._text.face = family
