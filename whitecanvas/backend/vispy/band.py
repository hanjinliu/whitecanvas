from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from vispy.color import Color as vispyColor
from vispy.gloo import set_state
from vispy.scene import visuals

from whitecanvas.backend import _not_implemented
from whitecanvas.protocols import BandProtocol, check_protocol
from whitecanvas.types import LineStyle, Orientation


# vispy's Polygon is not well implemented. Use custom class.
class VispyBand(visuals.Compound):
    def __init__(
        self,
        pos: NDArray[np.floating],
        color="black",
        border_color=None,
        border_width=1,
        **kwargs,
    ):
        self._mesh = visuals.Mesh()
        self._border = visuals.Line(method="gl")
        self._pos = pos
        self._color = vispyColor(color)
        self._border_width = border_width
        self._border_color = vispyColor(border_color)

        self._update()
        visuals.Compound.__init__(self, [self._mesh, self._border], **kwargs)
        self._mesh.set_gl_state(
            polygon_offset_fill=True,
            polygon_offset=(1, 1),
            cull_face=False,
        )
        self.freeze()

    def _update(self):
        set_state(polygon_offset_fill=False)
        npos = self._pos.shape[0]
        _index = np.arange(npos // 2 - 1, dtype=np.uint32)
        _index_inv = npos - _index
        faces0 = np.stack([_index, _index + 1, _index_inv - 1], axis=1)
        faces1 = np.stack([_index + 1, _index_inv - 1, _index_inv - 2], axis=1)
        faces = np.concatenate([faces0, faces1], axis=0)
        self._mesh.set_data(vertices=self._pos, faces=faces, color=self._color.rgba)
        if not self._border_color.is_blank:
            # Close border if it is not already.
            border_pos = self._pos
            if np.any(border_pos[0] != border_pos[-1]):
                border_pos = np.concatenate([border_pos, border_pos[:1]], axis=0)
            self._border.set_data(
                pos=border_pos, color=self._border_color.rgba, width=self._border_width
            )

            self._border.update()

    def set_pos(self, pos: NDArray[np.floating]):
        self._pos = pos
        self._update()

    @property
    def color(self):
        """The color of the polygon."""
        return self._color

    @color.setter
    def color(self, color):
        self._color = vispyColor(color, clip=True)
        self._mesh.set_data(
            vertices=self._mesh.mesh_data.get_vertices(),
            faces=self._mesh.mesh_data.get_faces(),
            color=self._color.rgba,
        )

    @property
    def border_color(self):
        """The border color of the polygon."""
        return self._border_color

    @border_color.setter
    def border_color(self, border_color):
        self._border_color = vispyColor(border_color)
        self._update()

    @property
    def border_width(self) -> float:
        return self._border.width

    @border_width.setter
    def border_width(self, width: float):
        self._border_width = width
        self._update()


@check_protocol(BandProtocol)
class Band(VispyBand):
    def __init__(self, t, ydata0, ydata1, orient: Orientation):
        if orient.is_vertical:
            fw = np.stack([t, ydata0], axis=1)
            bw = np.stack([t[::-1], ydata1[::-1]], axis=1)
        else:
            fw = np.stack([ydata0, t], axis=1)
            bw = np.stack([ydata1[::-1], t[::-1]], axis=1)
        verts = np.concatenate([fw, bw], axis=0)
        self._edge_style = LineStyle.SOLID
        super().__init__(verts, border_width=0)
        self.unfreeze()
        self._t = t
        self._y0 = ydata0
        self._y1 = ydata1

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_vertical_data(self):
        return self._t, self._y0, self._y1

    def _plt_get_horizontal_data(self):
        return self._t, self._y0, self._y1

    def _plt_set_vertical_data(self, t, ydata0, ydata1):
        verts = np.concatenate(
            [
                np.stack([t, ydata0], axis=1),
                np.stack([t[::-1], ydata1[::-1]], axis=1),
            ],
            axis=0,
        )
        self.set_pos(verts)
        self._t = t
        self._y0 = ydata0
        self._y1 = ydata1

    def _plt_set_horizontal_data(self, t, ydata0, ydata1):
        verts = np.concatenate(
            [
                np.stack([ydata0, t], axis=1),
                np.stack([ydata1[::-1], t[::-1]], axis=1),
            ],
            axis=0,
        )
        self.set_pos(verts)
        self._t = t
        self._y0 = ydata0
        self._y1 = ydata1

    ##### HasFace protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return np.array(self.color, dtype=np.float32)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self.color = color

    _plt_get_face_hatch, _plt_set_face_hatch = _not_implemented.face_pattern()

    ##### HasEdges protocol #####
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(self.border_color, dtype=np.float32)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.border_color = color

    def _plt_get_edge_width(self) -> float:
        return self.border_width

    def _plt_set_edge_width(self, width: float):
        self.border_width = width

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_style()

    def _plt_set_hover_text(self, text: list[str]):
        # TODO: not used yet
        self._hover_texts = text

    def _plt_connect_pick_event(self, callback):
        pass
