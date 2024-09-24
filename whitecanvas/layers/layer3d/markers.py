from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

import numpy as np
from cmap import Colormap
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers._mixin import (
    EdgeNamespace,
    FaceEdgeMixinEvents,
    FaceNamespace,
    MultiFaceEdgeMixin,
)
from whitecanvas.layers._sizehint import xyz_size_hint
from whitecanvas.layers.layer3d._base import DataBoundLayer3D
from whitecanvas.protocols import MarkersProtocol
from whitecanvas.types import (
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Symbol,
    XYZData,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xyz

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers._mixin import ConstEdge, ConstFace, MultiEdge, MultiFace

_void = _Void()
_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)
_Size = TypeVar("_Size", float, NDArray[np.floating])


class MarkersLayerEvents(FaceEdgeMixinEvents):
    clicked = Signal(list)
    symbol = Signal(Symbol)
    size = Signal(float)


class Markers3D(
    DataBoundLayer3D[MarkersProtocol, XYZData],
    MultiFaceEdgeMixin[_Face, _Edge],
    Generic[_Face, _Edge, _Size],
):
    events: MarkersLayerEvents
    _events_class = MarkersLayerEvents
    _backend_class_name = "components3d.Markers3D"

    if TYPE_CHECKING:

        def __new__(cls, *args, **kwargs) -> Markers3D[ConstFace, ConstEdge, float]: ...

    def __init__(
        self,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        zdata: ArrayLike1D,
        *,
        name: str | None = None,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 12.0,
        color: ColorType = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
        MultiFaceEdgeMixin.__init__(self)
        xdata, ydata, zdata = normalize_xyz(xdata, ydata, zdata)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), xdata, ydata, zdata)
        self._size_is_array = False
        self.update(symbol=symbol, size=size, color=color, hatch=hatch, alpha=alpha)
        self.edge.color = color
        if not self.symbol.has_face():
            self.edge.update(width=2.0, color=color)
        pad_r = size / 400
        self._x_hint, self._y_hint, self._z_hint = xyz_size_hint(
            xdata, ydata, zdata, pad_r, pad_r, pad_r
        )

        # self._backend._plt_connect_pick_event(self.events.clicked.emit)
        self._init_events()

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data.x.size

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], backend: Backend | str | None = None
    ) -> Markers3D:
        return cls(
            d["data"]["x"], d["data"]["y"], d["data"]["z"], name=d["name"],
            symbol=d["symbol"], size=d["size"], color=d["face"]["color"],
            hatch=d["face"]["hatch"], backend=backend,
        ).with_edge(
            color=d["edge"]["color"], width=d["edge"]["width"], style=d["edge"]["style"]
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "line3d",
            "data": self.data.to_dict(),
            "name": self.name,
            "symbol": self.symbol,
            "size": self.size,
            "face": self.face.to_dict(),
            "edge": self.edge.to_dict(),
        }

    def _get_layer_data(self) -> XYZData:
        return XYZData(*self._backend._plt_get_data())

    def _norm_layer_data(self, data: Any) -> XYZData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 3:
                raise ValueError("Expected a (N, 3) array")
            xdata, ydata, zdata = data[:, 0], data[:, 1], data[:, 2]
        else:
            xdata, ydata, zdata = data
        x0, y0, z0 = self.data
        if xdata is not None:
            x0 = as_array_1d(xdata)
        if ydata is not None:
            y0 = as_array_1d(ydata)
        if zdata is not None:
            z0 = as_array_1d(zdata)
        if x0.size != y0.size or x0.size != z0.size:
            raise ValueError(
                "Expected xdata, ydata and zdata to have the same size, "
                f"got {x0.size}, {y0.size} and {z0.size}."
            )
        return XYZData(x0, y0, z0)

    def _set_layer_data(self, data: XYZData):
        x0, y0, z0 = data
        self._backend._plt_set_data(x0, y0, z0)
        if self._size_is_array:
            pad_r = self.size.mean() / 400
        else:
            pad_r = self.size / 400
        self._x_hint, self._y_hint, self._z_hint = xyz_size_hint(
            x0, y0, z0, pad_r, pad_r, pad_r
        )

    def set_data(
        self,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
        zdata: ArrayLike | None = None,
    ):
        if xdata is None:
            xdata = self.data.x
        if ydata is None:
            ydata = self.data.y
        if zdata is None:
            zdata = self.data.z
        self.data = XYZData(xdata, ydata, zdata)

    @property
    def symbol(self) -> Symbol:
        """Symbol used to mark the data points."""
        return self._backend._plt_get_symbol()

    @symbol.setter
    def symbol(self, symbol: str | Symbol):
        sym = Symbol(symbol)
        if self.ndata > 0:
            self._backend._plt_set_symbol(sym)
        self.events.symbol.emit(sym)

    @property
    def size(self) -> _Size:
        """Size of the symbol."""
        size = self._backend._plt_get_symbol_size()
        if self._size_is_array:
            return size
        elif size.size > 0:
            return size[0]
        else:
            return 10.0  # default size

    @size.setter
    def size(self, size: _Size):
        """Set marker size"""
        ndata = self.ndata
        if not isinstance(size, (float, int, np.number)):
            if not self._size_is_array:
                raise ValueError(
                    "Expected size to be a scalar. Use with_size_multi() to "
                    "set multiple sizes."
                )
            size = as_array_1d(size)
            if size.size != ndata:
                raise ValueError(
                    f"Expected `size` to have the same size as the layer data size "
                    f"({self.ndata}), got {size.size}."
                )
        if ndata > 0:
            self._backend._plt_set_symbol_size(size)
        self.events.size.emit(size)

    def update(
        self,
        *,
        symbol: Symbol | str | _Void = _void,
        size: float | _Void = _void,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        hatch: str | Hatch | _Void = _void,
    ) -> Self:
        """Update the properties of the markers."""
        if symbol is not _void:
            self.symbol = symbol
        if size is not _void:
            self.size = size
        if color is not _void:
            self.face.color = color
        if hatch is not _void:
            self.face.hatch = hatch
        if alpha is not _void:
            self.face.alpha = alpha
        return self

    def color_by_density(
        self,
        cmap: ColormapType = "jet",
        *,
        width: float = 0.0,
    ) -> Self:
        """
        Set the color of the markers by density.

        Parameters
        ----------
        cmap : ColormapType, optional
            Colormap used to map the density to colors.
        """
        from whitecanvas.utils.kde import gaussian_kde

        xyzdata = self.data
        xyz = np.vstack([xyzdata.x, xyzdata.y, xyzdata.z])
        density = gaussian_kde(xyz)(xyz)
        normed = density / density.max()
        self.with_face_multi(color=Colormap(cmap)(normed))
        if width is not None:
            self.width = width
        return self

    def as_edge_only(
        self,
        *,
        width: float = 3.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Self:
        """
        Convert the markers to edge-only mode.

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
        if color.ndim == 0:
            pass
        elif color.ndim == 1:
            self.with_edge(color=color, width=width, style=style)
        elif color.ndim == 2:
            self.with_edge_multi(color=color, width=width, style=style)
        else:
            raise RuntimeError("Unreachable error.")
        self.face.update(alpha=0.0)
        return self

    def with_face(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1.0,
    ) -> Markers3D[ConstFace, _Edge, _Size]:
        """
        Return a markers layer with constant face properties.

        In most cases, this function is not needed because the face properties
        can be set directly on construction. Examples below are equivalent:

        >>> markers = canvas.add_markers(x, y).with_face(color="red")
        >>> markers = canvas.add_markers(x, y, color="red")

        This function will be useful when you want to change the markers from
        multi-face state to constant-face state:

        >>> markers = canvas.add_markers(x, y).with_face_multi(color=colors)
        >>> markers = markers.with_face(color="red")

        Parameters
        ----------
        color : color-like, optional
            Color of the marker faces.
        hatch : str or FacePattern, optional
            Hatch pattern of the faces.
        alpha : float, optional
            Alpha channel of the faces.

        Returns
        -------
        Markers3D
            The updated markers layer.
        """
        super().with_face(color, hatch, alpha)
        if not self.symbol.has_face():
            width = self.edge.width
            if isinstance(width, (float, int, np.number)):
                self.edge.update(width=width or 1.0, color=color)
            else:
                self.edge.update(width=width[0] or 1.0, color=color)
        return self

    def with_face_multi(
        self,
        *,
        color: ColorType | Sequence[ColorType] | _Void = _void,
        hatch: str | Hatch | Sequence[str | Hatch] | _Void = _void,
        alpha: float = 1,
    ) -> Markers3D[MultiFace, _Edge, _Size]:
        """
        Return a markers layer with multi-face properties.

        This function is used to create a markers layer with multiple face
        properties, such as colorful markers.

        >>> markers = canvas.add_markers(x, y).with_face_multi(color=colors)

        Parameters
        ----------
        color : color-like or sequence of color-like, optional
            Color(s) of the marker faces.
        hatch : str or Hatch or sequence of it, optional
            Pattern(s) of the faces.
        alpha : float or sequence of float, optional
            Alpha channel(s) of the faces.

        Returns
        -------
        Markers3D
            The updated markers layer.
        """
        super().with_face_multi(color, hatch, alpha)
        if not self.symbol.has_face():
            width = self.edge.width
            if isinstance(width, (float, int, np.number)):
                self.edge.update(width=width or 1.0, color=color)
            else:
                self.edge.update(width=width[0] or 1.0, color=color)
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Markers3D[_Face, ConstEdge, _Size]:
        return super().with_edge(color, width, style, alpha)

    def with_edge_multi(
        self,
        *,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1.0,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Markers3D[_Face, MultiEdge, _Size]:
        """
        Return a markers layer with multi-edge properties.

        This function is used to create a markers layer with multiple edge
        properties, such as colorful markers.

        >>> markers = canvas.add_markers(x, y).with_edge_multi(color=colors)

        Parameters
        ----------
        color : color-like or sequence of color-like, optional
            Color(s) of the marker faces.
        width : float or array of float, optional
            Width(s) of the edges.
        style : str, LineStyle or sequence of them, optional
            Line style(s) of the edges.
        alpha : float or sequence of float, optional
            Alpha channel(s) of the faces.

        Returns
        -------
        Markers3D
            The updated markers layer.
        """
        return super().with_edge_multi(color, width, style, alpha)

    def with_size_multi(
        self,
        size: float | Sequence[float],
    ) -> Markers3D[_Face, _Edge, NDArray[np.float32]]:
        if isinstance(size, (float, int, np.number)):
            _size = np.full(self.ndata, size, dtype=np.float32)
        else:
            _size = as_array_1d(size, dtype=np.float32)
            if _size.size != self.ndata:
                raise ValueError(
                    "Expected size to have the same size as the data, "
                    f"got {_size.size} and {self.ndata}"
                )
        self._size_is_array = True
        self.size = _size
        return self
