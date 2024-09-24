from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

import numpy as np
from cmap import Colormap
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend, _text_utils
from whitecanvas.layers._base import HoverableDataBoundLayer
from whitecanvas.layers._mixin import (
    EdgeNamespace,
    FaceEdgeMixinEvents,
    FaceNamespace,
    MultiFaceEdgeMixin,
)
from whitecanvas.layers._primitive.text import Texts
from whitecanvas.layers._sizehint import xy_size_hint
from whitecanvas.protocols import MarkersProtocol
from whitecanvas.types import (
    Alignment,
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    OrientationLike,
    Symbol,
    XYData,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xy, parse_texts

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers import group as _lg
    from whitecanvas.layers._mixin import ConstEdge, ConstFace, MultiEdge, MultiFace

_void = _Void()
_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)
_Size = TypeVar("_Size", float, NDArray[np.floating])


class MarkersLayerEvents(FaceEdgeMixinEvents):
    clicked = Signal(list)
    symbol = Signal(Symbol)
    size = Signal(float)


class Markers(
    HoverableDataBoundLayer[MarkersProtocol, XYData],
    MultiFaceEdgeMixin[_Face, _Edge],
    Generic[_Face, _Edge, _Size],
):
    events: MarkersLayerEvents
    _events_class = MarkersLayerEvents

    if TYPE_CHECKING:

        def __new__(cls, *args, **kwargs) -> Markers[ConstFace, ConstEdge, float]: ...

    def __init__(
        self,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        *,
        name: str | None = None,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 12.0,
        color: ColorType = "blue",
        alpha: float | _Void = _void,
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
        MultiFaceEdgeMixin.__init__(self)
        xdata, ydata = normalize_xy(xdata, ydata)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self._size_is_array = False
        self.update(symbol=symbol, size=size, color=color, hatch=hatch, alpha=alpha)
        self.edge.color = color
        if not self.symbol.has_face():
            self.edge.update(width=2.0, color=color)
        pad_r = size / 400
        self._x_hint, self._y_hint = xy_size_hint(xdata, ydata, pad_r, pad_r)

        self._backend._plt_connect_pick_event(self.events.clicked.emit)
        self._init_events()

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data.x.size

    def _get_layer_data(self) -> XYData:
        return XYData(*self._backend._plt_get_data())

    def _norm_layer_data(self, data: Any) -> XYData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("Expected a (N, 2) array")
            xdata, ydata = data[:, 0], data[:, 1]
        else:
            xdata, ydata = data
        x0, y0 = self.data
        if xdata is not None:
            x0 = as_array_1d(xdata)
        if ydata is not None:
            y0 = as_array_1d(ydata)
        if x0.size != y0.size:
            raise ValueError(
                "Expected xdata and ydata to have the same size, "
                f"got {x0.size} and {y0.size}"
            )
        return XYData(x0, y0)

    def _set_layer_data(self, data: XYData):
        x0, y0 = data
        self._backend._plt_set_data(x0, y0)
        if self._size_is_array:
            pad_r = self.size.mean() / 400
        else:
            pad_r = self.size / 400
        self._x_hint, self._y_hint = xy_size_hint(x0, y0, pad_r, pad_r)

    def set_data(
        self,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
    ):
        if xdata is None:
            xdata = self.data.x
        if ydata is None:
            ydata = self.data.y
        self.data = XYData(xdata, ydata)

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a Band from a dictionary."""
        return cls(
            d["data"]["x"], d["data"]["y"], size=d["size"], symbol=d["symbol"],
            name=d["name"], color=d["face"]["color"],
            hatch=d["face"]["hatch"], backend=backend,
        ).with_edge(
            color=d["edge"]["color"], width=d["edge"]["width"], style=d["edge"]["style"]
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": "markers",
            "data": self._get_layer_data().to_dict(),
            "name": self.name,
            "size": self.size,
            "symbol": self.symbol,
            "face": self.face.to_dict(),
            "edge": self.edge.to_dict(),
        }

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

        xydata = self.data
        xy = np.vstack([xydata.x, xydata.y])
        density = gaussian_kde(xy)(xy)
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

    def with_hover_template(
        self,
        template: str,
        extra: Any | None = None,
    ) -> Self:
        """Add hover template to the markers."""
        xs, ys = self.data
        if self._backend_name in ("plotly", "bokeh"):  # conversion for HTML
            template = template.replace("\n", "<br>")
        params = parse_texts(template, xs.size, extra)
        # set default format keys
        params.setdefault("x", xs)
        params.setdefault("y", ys)
        if "i" not in params:
            params["i"] = np.arange(xs.size)
        texts = [
            template.format(**{k: v[i] for k, v in params.items()})
            for i in range(xs.size)
        ]
        self._backend._plt_set_hover_text(texts)
        return self

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.LabeledMarkers:
        """
        Add horizontal error bars to the markers.

        Parameters
        ----------
        err : ArrayLike
            Error values. If `err_high` is not specified, the error bars are
            symmetric.
        err_high : array-like, optional
            Upper error values.
        color : color-like, optional
            Line color of the error bars.
        width : float
            Width of the error bars.
        style : str or LineStyle
            Line style of the error bars.
        antialias : bool
            Antialiasing of the error bars.
        capsize : float, optional
            Size of the caps at the end of the error bars.

        Returns
        -------
        LabeledMarkers
            Layer group containing the markers and the error bars as children.
        """
        from whitecanvas.layers._primitive import Errorbars
        from whitecanvas.layers.group import LabeledMarkers

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge.color
        if width is _void:
            width = self.edge.width
        if style is _void:
            style = self.edge.style
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            orient=Orientation.HORIZONTAL, backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars.empty_v(f"xerr-of-{self.name}", backend=self._backend_name)
        return LabeledMarkers(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
    ) -> _lg.LabeledMarkers:
        """
        Add vertical error bars to the markers.

        Parameters
        ----------
        err : ArrayLike
            Error values. If `err_high` is not specified, the error bars are
            symmetric.
        err_high : array-like, optional
            Upper error values.
        color : color-like, optional
            Line color of the error bars.
        width : float
            Width of the error bars.
        style : str or LineStyle
            Line style of the error bars.
        antialias : bool
            Antialiasing of the error bars.
        capsize : float, optional
            Size of the caps at the end of the error bars.

        Returns
        -------
        LabeledMarkers
            Layer group containing the markers and the error bars as children.
        """
        from whitecanvas.layers._primitive import Errorbars
        from whitecanvas.layers.group import LabeledMarkers

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge.color
        if width is _void:
            width = self.edge.width
        if style is _void:
            style = self.edge.style
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            orient=Orientation.VERTICAL, backend=self._backend_name
        )  # fmt: skip
        xerr = Errorbars.empty_h(f"yerr-of-{self.name}", backend=self._backend_name)
        return LabeledMarkers(self, xerr, yerr, name=self.name)

    def with_text(
        self,
        strings: str | list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
    ) -> _lg.LabeledMarkers:
        """
        Add texts at the positions of the data points.

        Parameters
        ----------
        strings : str, sequence of str
            String values to be added. If single str is given, same string will be added
            to all of the positions.
        color : ColorType, optional
            Color of the text string.
        size : float, default 12
            Point size of the text.
        rotation : float, default 0.0
            Rotation angle of the text in degrees.
        anchor : str or Alignment, default Alignment.BOTTOM_LEFT
            Anchor position of the text. The anchor position will be the coordinate
            given by (x, y).
        family : str, optional
            Font family of the text.

        Returns
        -------
        LabeledMarkers
            Layer group containing the markers.
        """
        from whitecanvas.layers import Errorbars
        from whitecanvas.layers.group import LabeledMarkers

        old_name = self.name
        strings = _text_utils.norm_label_text(strings, self.data)
        texts = Texts(
            *self.data, strings, color=color, size=size, rotation=rotation,
            name=f"text-of-{old_name}", anchor=anchor, family=fontfamily,
            backend=self._backend_name,
        )  # fmt: skip
        self.name = f"markers-of-{old_name}"
        return LabeledMarkers(
            self,
            Errorbars.empty_h(f"xerr-of-{old_name}", backend=self._backend_name),
            Errorbars.empty_v(f"yerr-of-{old_name}", backend=self._backend_name),
            texts=texts,
            name=old_name,
        )

    def with_network(
        self,
        connections: NDArray[np.intp],
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool = True,
    ) -> _lg.Graph:
        """
        Add network edges to the markers to create a graph.

        Parameters
        ----------
        connections : (N, 2) array of int
            Integer array that defines the connections between nodes.
        color : color-like, optional
            Color of the lines.
        width : float, optional
            Width of the line.
        style : str, optional
            Line style of the line.
        antialias : bool, optional
            Antialiasing of the line.

        Returns
        -------
        Graph
            A Graph layer that contains the markers and the edges as children.
        """
        from whitecanvas.layers._primitive import MultiLine
        from whitecanvas.layers.group import Graph

        edges = np.asarray(connections, dtype=np.intp)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be a (N, 2) array")

        if color is _void:
            color = self.edge.color
        if width is _void:
            width = self.edge.width
        if style is _void:
            style = self.edge.style
        segs = []
        nodes = self.data.stack()
        for i0, i1 in edges:
            segs.append(np.stack([nodes[i0], nodes[i1]], axis=0))
        edges_layer = MultiLine(
            segs, name="edges", color=color, width=width, style=style,
            antialias=antialias, backend=self._backend_name
        )  # fmt: skip
        texts = Texts(
            nodes[:, 0], nodes[:, 1], [""] * nodes.shape[0], name="texts",
            backend=self._backend_name,
        )  # fmt: skip
        return Graph(self, edges_layer, texts, name=self.name)

    def with_stem(
        self,
        orient: OrientationLike = "vertical",
        *,
        bottom: NDArray[np.floating] | float | None = None,
        color: ColorType | _Void = _void,
        alpha: float = 1.0,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool = True,
    ) -> _lg.StemPlot:
        """
        Grow stems from the markers.

        Parameters
        ----------
        orient : str or Orientation, default "vertical"
            Orientation to grow stems.
        bottom : float or array-like, optional
            Bottom of the stems. If not specified, the bottom is set to 0.
        color : color-like, optional
            Color of the lines.
        alpha : float, optional
            Alpha channel of the lines.
        width : float, optional
            Width of the lines.
        style : str or LineStyle
            Line style used to draw the stems.
        antialias : bool, optional
            Line antialiasing.

        Returns
        -------
        StemPlot
            StemPlot layer containing the markers and the stems as children.
        """
        from whitecanvas.layers._primitive import MultiLine
        from whitecanvas.layers.group import StemPlot

        ori = Orientation.parse(orient)
        xdata, ydata = self.data
        if bottom is None:
            bottom = np.zeros_like(ydata)
        elif isinstance(bottom, (float, int, np.number)):
            bottom = np.full((ydata.size,), bottom)
        else:
            bottom = as_array_1d(bottom)
            if bottom.shape != ydata.shape:
                raise ValueError(
                    "Expected bottom to have the same size as ydata, "
                    f"got {bottom.shape} and {ydata.shape}"
                )
        if ori.is_vertical:
            root = np.stack([xdata, bottom], axis=1)
            leaf = np.stack([xdata, ydata], axis=1)
        else:
            root = np.stack([bottom, ydata], axis=1)
            leaf = np.stack([xdata, ydata], axis=1)
        segs = np.stack([root, leaf], axis=1)
        if color is _void:
            color = self.edge.color
        if width is _void:
            width = self.edge.width
        if style is _void:
            style = self.edge.style
        mline = MultiLine(
            segs, name="stems", color=color, width=width, style=style,
            antialias=antialias, alpha = alpha, backend=self._backend_name,
        )  # fmt: skip
        return StemPlot(self, mline, orient=orient, name=self.name)

    def with_face(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1.0,
    ) -> Markers[ConstFace, _Edge, _Size]:
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
        Markers
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
    ) -> Markers[MultiFace, _Edge, _Size]:
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
        Markers
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
    ) -> Markers[_Face, ConstEdge, _Size]:
        return super().with_edge(color, width, style, alpha)

    def with_edge_multi(
        self,
        *,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1.0,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Markers[_Face, MultiEdge, _Size]:
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
        Markers
            The updated markers layer.
        """
        return super().with_edge_multi(color, width, style, alpha)

    def with_size_multi(
        self,
        size: float | Sequence[float],
    ) -> Markers[_Face, _Edge, NDArray[np.float32]]:
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

    def _as_all_multi(self) -> Markers[MultiFace, MultiEdge, NDArray[np.float32]]:
        return (
            self.with_face_multi().with_edge_multi(width=0.0).with_size_multi(self.size)
        )

    def _as_legend_item(self) -> _legend.MarkersLegendItem:
        if self._size_is_array:
            size = np.mean(self.size)
        else:
            size = self.size
        return _legend.MarkersLegendItem(
            self.symbol, size, self.face._as_legend_info(), self.edge._as_legend_info()
        )
