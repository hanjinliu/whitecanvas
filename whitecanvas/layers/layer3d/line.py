from __future__ import annotations

from typing import Any

import numpy as np

from whitecanvas.backend import Backend
from whitecanvas.layers._primitive.line import LineLayerEvents, LineMixin
from whitecanvas.layers._sizehint import xyz_size_hint
from whitecanvas.layers.layer3d._base import DataBoundLayer3D
from whitecanvas.protocols import LineProtocol
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    LineStyle,
    XYZData,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xyz


class Line3D(LineMixin[LineProtocol], DataBoundLayer3D[LineProtocol, XYZData]):
    _backend_class_name = "components3d.MonoLine3D"
    events: LineLayerEvents
    _events_class = LineLayerEvents

    def __init__(
        self,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        zdata: ArrayLike1D,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        width: float = 1,
        alpha: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        xdata, ydata, zdata = normalize_xyz(xdata, ydata, zdata)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), xdata, ydata, zdata)
        self.update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )
        self._x_hint, self._y_hint, self._z_hint = xyz_size_hint(xdata, ydata, zdata)
        # self._backend._plt_connect_pick_event(self.events.clicked.emit)

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], backend: Backend | str | None = None
    ) -> Line3D:
        return cls(
            d["data"]["x"],
            d["data"]["y"],
            d["data"]["z"],
            name=d["name"],
            color=d["color"],
            width=d["width"],
            style=d["style"],
            antialias=d["antialias"],
            backend=backend,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "data": self.data.to_dict(),
            "name": self.name,
            "color": self.color,
            "width": self.width,
            "style": self.style,
            "antialias": self.antialias,
        }

    def _get_layer_data(self) -> XYZData:
        return XYZData(*self._backend._plt_get_data())

    def _norm_layer_data(self, data: Any) -> XYZData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 3:
                raise ValueError(f"Expected data to be (N, 3), got {data.shape}")
            xdata, ydata, zdata = data[:, 0], data[:, 1], data[:, 2]
        else:
            xdata, ydata, zdata = data
            if xdata is None:
                xdata = self.data.x
            else:
                xdata = as_array_1d(xdata)
            if ydata is None:
                ydata = self.data.y
            else:
                ydata = as_array_1d(ydata)
            if zdata is None:
                zdata = as_array_1d(zdata)
        if xdata.size != ydata.size or xdata.size != zdata.size:
            raise ValueError(
                "Expected xdata, ydata and zdata to have the same size, "
                f"got {xdata.size}, {ydata.size} and {zdata.size}."
            )
        return XYZData(xdata, ydata)

    def _set_layer_data(self, data: XYZData):
        x0, y0, z0 = data
        self._backend._plt_set_data(x0, y0, z0)
        self._x_hint, self._y_hint, self._z_hint = xyz_size_hint(x0, y0, z0)

    def set_data(
        self,
        xdata: ArrayLike1D | None = None,
        ydata: ArrayLike1D | None = None,
        zdata: ArrayLike1D | None = None,
    ):
        self.data = xdata, ydata, zdata

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data.x.size

    # def with_markers(
    #     self,
    #     symbol: Symbol | str = Symbol.CIRCLE,
    #     size: float | None = None,
    #     color: ColorType | _Void = _void,
    #     alpha: float = 1.0,
    #     hatch: str | Hatch = Hatch.SOLID,
    # ) -> _lg.Plot:
