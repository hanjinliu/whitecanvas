from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from plotly import graph_objects as go

from whitecanvas.backend import _not_implemented
from whitecanvas.backend.plotly._base import PlotlyHoverableLayer
from whitecanvas.utils.normalize import arr_color, as_color_array, rgba_str_color


class Mesh3D(PlotlyHoverableLayer[go.Mesh3d]):
    def __init__(self, verts: NDArray[np.floating], faces: NDArray[np.intp]):
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        nfaces = faces.shape[0]
        self._props = {
            "x": verts[:, 0],
            "y": verts[:, 1],
            "z": verts[:, 2],
            "i": faces[:, 0],
            "j": faces[:, 1],
            "k": faces[:, 2],
            "type": "mesh3d",
            "showlegend": False,
            "visible": True,
            "facecolor": ["blue"] * nfaces,
            "vertexcolor": (0, 0, 255, 255),
        }
        self._verts = verts
        self._faces = faces
        PlotlyHoverableLayer.__init__(self)

    def _plt_get_data(self):
        return self._verts, self._faces

    def _plt_set_data(self, verts, faces):
        self._verts = verts
        self._faces = faces
        self._props["x"] = verts[:, 0]
        self._props["y"] = verts[:, 1]
        self._props["z"] = verts[:, 2]
        self._props["i"] = faces[:, 0]
        self._props["j"] = faces[:, 1]
        self._props["k"] = faces[:, 2]

    def _plt_get_ndata(self) -> int:
        return self._faces.shape[0]

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        color = self._props["facecolor"]
        if len(color) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack([arr_color(c) for c in color], axis=0)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, self._faces.shape[0])
        self._props["facecolor"] = [rgba_str_color(c) for c in color]

    _plt_get_face_hatch, _plt_set_face_hatch = _not_implemented.face_hatches()
    _plt_get_edge_color, _plt_set_edge_color = _not_implemented.edge_color()
    _plt_get_edge_width, _plt_set_edge_width = _not_implemented.edge_width()
    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_style()
