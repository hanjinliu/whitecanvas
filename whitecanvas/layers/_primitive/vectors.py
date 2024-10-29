from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers._base import (
    DataBoundLayer,
    LayerEvents,
)
from whitecanvas.layers._sizehint import xy_size_hint
from whitecanvas.protocols import VectorsProtocol
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    LineStyle,
    XYVectorData,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d, as_color_array
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

_void = _Void()
_V = TypeVar("_V", bound=VectorsProtocol)


class VectorsLayerEvents(LayerEvents):
    color = Signal(np.ndarray)
    width = Signal(float)
    style = Signal(str)
    antialias = Signal(bool)
    # clicked = Signal(int)


class Vectors(DataBoundLayer[_V, XYVectorData]):
    events: VectorsLayerEvents
    _events_class = VectorsLayerEvents

    def __init__(
        self,
        x: ArrayLike1D,
        y: ArrayLike1D,
        vx: ArrayLike1D,
        vy: ArrayLike1D,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        width: float = 1,
        alpha: float | _Void = _void,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), x, vx, y, vy)
        color = as_color_array(color, len(x))
        self.update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )
        self._x_hint, self._y_hint = xy_size_hint(
            np.concatenate([x, x + vx]), np.concatenate([y, y + vy])
        )

    def _get_layer_data(self) -> XYVectorData:
        x, vx, y, vy = self._backend._plt_get_data()
        return XYVectorData(x, y, vx, vy)

    def _norm_layer_data(self, data: Any) -> XYVectorData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 4:
                raise ValueError(f"Expected data to be (N, 4), got {data.shape}")
            xdata, ydata = data[:, 0], data[:, 1]
            xvec, yvec = data[:, 2], data[:, 3]
        else:
            xdata, ydata, xvec, yvec = data
            if xdata is None:
                xdata = self.data.x
            else:
                xdata = as_array_1d(xdata)
            if ydata is None:
                ydata = self.data.y
            else:
                ydata = as_array_1d(ydata)
        if xdata.size != ydata.size:
            raise ValueError(
                "Expected xdata and ydata to have the same size, "
                f"got {xdata.size} and {ydata.size}"
            )
        return XYVectorData(xdata, ydata, xvec, yvec)

    def _set_layer_data(self, data: XYVectorData):
        x0, y0, vx, vy = data
        self._backend._plt_set_data(x0, vx, y0, vy)
        self._x_hint, self._y_hint = xy_size_hint(
            np.concatenate([x0, x0 + vx]), np.concatenate([y0, y0 + vy])
        )

    def set_data(
        self,
        xdata: ArrayLike1D | None = None,
        ydata: ArrayLike1D | None = None,
        xvec: ArrayLike1D | None = None,
        yvec: ArrayLike1D | None = None,
    ):
        self.data = xdata, ydata, xvec, yvec

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
            raise TypeError(f"Width must be a number, got {width!r}")
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
    def alpha(self) -> NDArray[np.floating]:
        return self.color[:, 3]

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

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a Line from a dictionary."""
        return cls(
            d["data"]["x"], d["data"]["y"], d["data"]["vx"], d["data"]["vy"],
            name=d["name"], color=d["color"], width=d["width"],
            style=d["style"], antialias=d["antialias"], backend=backend,
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "data": self._get_layer_data().to_dict(),
            "name": self.name,
            "visible": self.visible,
            "color": self.color,
            "width": self.width,
            "style": self.style,
            "antialias": self.antialias,
        }

    def _as_legend_item(self) -> _legend.LineLegendItem:
        return _legend.LineLegendItem(self.color[0], self.width, self.style)
