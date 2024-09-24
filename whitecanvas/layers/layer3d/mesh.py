from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers._mixin import AbstractFaceEdgeMixin, MonoEdge, MultiFace
from whitecanvas.layers._primitive.line import LineLayerEvents
from whitecanvas.layers.layer3d._base import DataBoundLayer3D
from whitecanvas.protocols import MeshProtocol
from whitecanvas.types import ColorType, Hatch, LineStyle, MeshData


class MeshFace(MultiFace):
    def _ndata(self) -> int:
        return self._layer.data.faces.shape[0]


class Mesh3D(
    DataBoundLayer3D[MeshProtocol, MeshData], AbstractFaceEdgeMixin[MeshFace, MonoEdge]
):
    _backend_class_name = "components3d.Mesh3D"
    events: LineLayerEvents
    _events_class = LineLayerEvents

    def __init__(
        self,
        verts: NDArray[np.floating],
        faces: NDArray[np.intp],
        *,
        name: str | None = None,
        color: ColorType = "blue",
        hatch: Hatch = Hatch.SOLID,
        alpha: float = 1.0,
        backend: Backend | str | None = None,
    ):
        super().__init__(name=name)
        AbstractFaceEdgeMixin.__init__(self, MeshFace(self), MonoEdge(self))
        self._backend = self._create_backend(Backend(backend), verts, faces)
        self._x_hint, self._y_hint, self._z_hint = _verts_to_hints(verts)
        # self._backend._plt_connect_pick_event(self.events.clicked.emit)
        self.face.update(color=color, hatch=hatch, alpha=alpha)

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], backend: Backend | str | None = None
    ) -> Mesh3D:
        return cls(
            np.asarray(d["data"]["vertices"], dtype=np.float32),
            np.asarray(d["data"]["faces"], dtype=np.float32), name=d["name"],
            color=d["face"]["color"], hatch=d["face"]["hatch"], backend=backend,
        ).with_edge(
            color=d["edge"]["color"], width=d["edge"]["width"], style=d["edge"]["style"]
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "data": self.data.to_dict(),
            "name": self.name,
            "face": self.face.to_dict(),
            "edge": self.edge.to_dict(),
        }

    def _get_layer_data(self) -> MeshData:
        return MeshData(*self._backend._plt_get_data())

    def _norm_layer_data(self, data: tuple[ArrayLike, ArrayLike]) -> MeshData:
        verts, faces = data
        verts = np.atleast_2d(verts)
        faces = np.atleast_2d(faces)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"Expected verts to be (N, 3), got {verts.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Expected faces to be (N, 3), got {faces.shape}")
        return MeshData(verts, faces)

    def _set_layer_data(self, data: MeshData):
        verts, faces = data
        self._backend._plt_set_data(verts, faces)
        self._x_hint, self._y_hint, self._z_hint = _verts_to_hints(verts)

    def set_data(self, verts: NDArray[np.floating], faces: NDArray[np.intp]):
        self.data = MeshData(verts, faces)

    def as_edge_only(
        self,
        *,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Mesh3D:
        """
        Convert the surface to edge-only mode.

        This method will set the face color to transparent and the edge color to the
        current face color.

        Parameters
        ----------
        width : float, default 3.0
            Width of the edge.
        style : str or LineStyle, default LineStyle.SOLID
            Line style of the edge.
        """
        color = self.face.color
        if color.size == 0:
            ec = self.edge.color
        else:
            ec = np.mean(color, axis=0)
        self.with_edge(color=ec, width=width, style=style)
        self.face.update(alpha=0.0)
        return self


def _verts_to_hints(verts: NDArray[np.floating]):
    xmin, ymin, zmin = verts.min(axis=0)
    xmax, ymax, zmax = verts.max(axis=0)
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)
