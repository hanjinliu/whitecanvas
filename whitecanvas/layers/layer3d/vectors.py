from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers._primitive.vectors import VectorsLayerEvents
from whitecanvas.layers._sizehint import xyz_size_hint
from whitecanvas.layers.layer3d._base import DataBoundLayer3D
from whitecanvas.protocols import VectorsProtocol
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    LineStyle,
    XYZVectorData,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d, as_color_array
from whitecanvas.utils.type_check import is_real_number

_void = _Void()
_V = TypeVar("_V", bound=VectorsProtocol)


class Vectors3D(DataBoundLayer3D[_V, XYZVectorData]):
    _backend_class_name = "components3d.Vectors3D"
    events: VectorsLayerEvents
    _events_class = VectorsLayerEvents

    def __init__(
        self,
        x: ArrayLike1D,
        y: ArrayLike1D,
        z: ArrayLike1D,
        vx: ArrayLike1D,
        vy: ArrayLike1D,
        vz: ArrayLike1D,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        width: float = 1,
        alpha: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), x, vx, y, vy, z, vz)
        color = as_color_array(color, x.size)
        self.update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )
        self._x_hint, self._y_hint, self._z_hint = xyz_size_hint(
            np.concatenate([x, x + vx]),
            np.concatenate([y, y + vy]),
            np.concatenate([z, z + vz]),
        )

    def _get_layer_data(self) -> XYZVectorData:
        x, vx, y, vy, z, vz = self._backend._plt_get_data()
        return XYZVectorData(x, y, z, vx, vy, vz)

    def _norm_layer_data(self, data: Any) -> XYZVectorData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 6:
                raise ValueError(f"Expected data to be (N, 4), got {data.shape}")
            xdata, ydata, zdata = data[:, 0], data[:, 1], data[:, 2]
            xvec, yvec, zvec = data[:, 2], data[:, 3], data[:, 4]
        else:
            xdata, ydata, zdata, xvec, yvec, zvec = data
            if xdata is None:
                xdata = self.data.x
            else:
                xdata = as_array_1d(xdata)
            if ydata is None:
                ydata = self.data.y
            else:
                ydata = as_array_1d(ydata)
            if zdata is None:
                zdata = self.data.z
            else:
                zdata = as_array_1d(zdata)
        if xdata.size != ydata.size or xdata.size != zdata.size:
            raise ValueError(
                "Expected xdata, ydata and zdata to have the same size, "
                f"got {xdata.size}, {ydata.size} and {zdata.size}."
            )
        return XYZVectorData(xdata, ydata, zdata, xvec, yvec, zvec)

    def _set_layer_data(self, data: XYZVectorData):
        x0, y0, z0, vx, vy, vz = data
        self._backend._plt_set_data(x0, y0, vx, vy)
        self._x_hint, self._y_hint, self._z_hint = xyz_size_hint(
            np.concatenate([x0, x0 + vx]),
            np.concatenate([y0, y0 + vy]),
            np.concatenate([z0, z0 + vz]),
        )

    def set_data(
        self,
        xdata: ArrayLike1D | None = None,
        ydata: ArrayLike1D | None = None,
        zdata: ArrayLike1D | None = None,
        xvec: ArrayLike1D | None = None,
        yvec: ArrayLike1D | None = None,
        zvec: ArrayLike1D | None = None,
    ):
        self.data = xdata, ydata, zdata, xvec, yvec, zvec

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data.x.size

    @property
    def color(self) -> NDArray[np.floating]:
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        col = as_color_array(color, self.ndata)
        self._backend._plt_set_edge_color(col)
        self.events.color.emit(col)

    @property
    def width(self) -> float:
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float):
        if not is_real_number(width):
            raise TypeError(f"Width must be a number, got {type(width)}")
        if width < 0:
            raise ValueError(f"Width must be non-negative, got {width!r}")
        w = float(width)
        self._backend._plt_set_edge_width(w)
        self.events.width.emit(w)

    @property
    def style(self) -> LineStyle:
        """Style of the line."""
        return LineStyle(self._backend._plt_get_edge_style())

    @style.setter
    def style(self, style: str | LineStyle):
        s = LineStyle(style)
        self._backend._plt_set_edge_style(s)
        self.events.style.emit(s.value)

    @property
    def alpha(self) -> float:
        return float(self.color[:, 3])

    @alpha.setter
    def alpha(self, value: float):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    @property
    def antialias(self) -> bool:
        """Whether to use antialiasing."""
        return self._backend._plt_get_antialias()

    @antialias.setter
    def antialias(self, antialias: bool) -> None:
        if not isinstance(antialias, bool):
            raise TypeError(f"Expected antialias to be bool, got {type(antialias)}")
        self._backend._plt_set_antialias(antialias)
        self.events.antialias.emit(antialias)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if width is not _void:
            self.width = width
        if style is not _void:
            self.style = style
        if alpha is not _void:
            self.alpha = alpha
        if antialias is not _void:
            self.antialias = antialias
        return self

    def _as_legend_item(self) -> _legend.LineLegendItem:
        return _legend.LineLegendItem(self.color[0], self.width, self.style)
