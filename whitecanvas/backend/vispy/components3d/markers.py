from __future__ import annotations

import numpy as np
from vispy.scene import visuals

from whitecanvas.backend.vispy.markers import Markers as Markers2D


class Markers3D(Markers2D):
    def __init__(self, xdata, ydata, zdata):
        pos = np.stack([xdata, ydata, zdata], axis=1)
        visuals.Markers.__init__(self, pos=pos, edge_width=0, face_color="blue")
        self.unfreeze()
        self._hover_texts: list[str] | None = None

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        data = self._data["a_position"]
        return data[:, 0], data[:, 1], data[:, 2]

    def _plt_set_data(self, xdata, ydata, zdata):
        ndata = self._plt_get_ndata()
        size = self._plt_get_symbol_size()
        edge_width = self._plt_get_edge_width()
        face_color = self._plt_get_face_color()
        edge_color = self._plt_get_edge_color()
        if xdata.size > ndata:
            size = np.concatenate([size, np.full(xdata.size - ndata, size[-1])])
            edge_width = np.concatenate(
                [edge_width, np.full(xdata.size - ndata, edge_width[-1])]
            )
            face_color = np.concatenate(
                [face_color, np.full((xdata.size - ndata, 4), face_color[-1])]
            )
            edge_color = np.concatenate(
                [edge_color, np.full((xdata.size - ndata, 4), edge_color[-1])]
            )
        elif xdata.size < ndata:
            size = size[: xdata.size]
            edge_width = edge_width[: xdata.size]
            face_color = face_color[: xdata.size]
            edge_color = edge_color[: xdata.size]
        self.set_data(
            pos=np.stack([xdata, ydata, zdata], axis=1),
            size=size,
            edge_width=edge_width,
            face_color=face_color,
            edge_color=edge_color,
            symbol=self.symbol[0],
        )

    def _plt_set_hover_text(self, text: list[str]):
        # TODO: not used yet
        self._hover_texts = text

    def _compute_bounds(self, axis, view):
        # override to fix the bounds computation
        pos = self._data["a_position"]
        if pos is None or pos.size == 0:
            return None
        if pos.shape[1] > axis:
            return (pos[:, axis].min(), pos[:, axis].max())
        else:
            return (0, 0)
