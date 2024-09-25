from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from vispy.geometry import MeshData
from vispy.scene import visuals
from vispy.visuals.filters import WireframeFilter

from whitecanvas.backend import _not_implemented


class Mesh3D(visuals.Mesh):
    mesh_data: MeshData

    def __init__(self, verts: NDArray[np.floating], faces: NDArray[np.intp]):
        self.wireframe_filter = WireframeFilter()
        super().__init__(vertices=verts, faces=faces)
        self.attach(self.wireframe_filter)
        self.unfreeze()

    def _plt_get_data(self):
        return self.mesh_data.get_vertices(), self.mesh_data.get_faces()

    def _plt_set_data(self, verts, faces):
        self.mesh_data.set_vertices(verts)
        self.mesh_data.set_faces(faces)
        self._update()

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self.mesh_data.get_face_colors()

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self.mesh_data.set_face_colors(color)
        self._update()

    _plt_get_face_hatch, _plt_set_face_hatch = _not_implemented.face_hatches()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.asarray(self.wireframe_filter.color.rgba, dtype=np.float32)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.wireframe_filter.color = color
        self.update()

    def _plt_get_edge_width(self) -> float:
        return self.wireframe_filter.width

    def _plt_set_edge_width(self, width: float):
        self.wireframe_filter.width = width
        self.update()

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_style()

    def _update(self):
        self._bounds = self.mesh_data.get_bounds()
        self.mesh_data_changed()

    def _plt_get_ndata(self) -> int:
        return self.mesh_data.get_faces().shape[0]
