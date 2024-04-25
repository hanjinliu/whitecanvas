import numpy as np
from mpl_toolkits.mplot3d import art3d

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.types import Hatch


class Mesh3D(art3d.Poly3DCollection, MplLayer):
    def __init__(self, verts, faces):
        super().__init__(verts[faces])
        self._verts = verts
        self._faces = faces

    def _plt_get_data(self):
        return self._verts, self._faces

    def _plt_set_data(self, verts, faces):
        self._verts = verts
        self._faces = faces
        self.set_verts(verts[faces])

    def _plt_get_face_color(self):
        if self.axes is None:
            return self._facecolor3d
        return self.get_facecolor()

    def _plt_set_face_color(self, color):
        if self.axes is None:
            self._facecolor3d = color
        else:
            self.set_facecolor(color)

    def _plt_get_face_hatch(self):
        return self.get_hatch()

    def _plt_set_face_hatch(self, hatch):
        if isinstance(hatch, Hatch):
            self.set_hatch(hatch.value)
        else:
            self.set_hatch(hatch[0])

    def _plt_get_edge_color(self):
        if self.axes is None:
            return np.array([0, 0, 0, 0])
        return self.get_edgecolor()[0]

    def _plt_set_edge_color(self, color):
        if self.axes is None:
            self._edgecolor3d = color
        else:
            self.set_edgecolor(color)

    def _plt_get_edge_width(self):
        return self.get_linewidth()[0]

    def _plt_set_edge_width(self, width):
        self.set_linewidth(width)

    def _plt_get_edge_style(self):
        return self.get_linestyle()

    def _plt_set_edge_style(self, style):
        self.set_linestyle(style)

    def post_add(self, canvas):
        self._plt_set_face_color(self._facecolor3d)
        self._plt_set_edge_color(self._edgecolor3d)
