from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Iterable, Sequence, TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas.layers._primitive.text import Texts
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.protocols import MarkersProtocol
from whitecanvas.layers._sizehint import xy_size_hint
from whitecanvas.layers._mixin import (
    MultiFaceEdgeMixin,
    FaceNamespace,
    EdgeNamespace,
    FaceEdgeMixinEvents,
)
from whitecanvas.backend import Backend
from whitecanvas.types import (
    LineStyle,
    Symbol,
    ColorType,
    FacePattern,
    _Void,
    Alignment,
    Orientation,
    XYData,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xy

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg
    from whitecanvas.layers._mixin import (
        ConstFace,
        ConstEdge,
        MultiFace,
        MultiEdge,
    )

_void = _Void()
_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)
_Size = TypeVar("_Size", float, NDArray[np.floating])


class MarkersLayerEvents(FaceEdgeMixinEvents):
    picked = Signal(list)
    symbol = Signal(Symbol)
    size = Signal(float)


class Markers(
    MultiFaceEdgeMixin[MarkersProtocol, _Face, _Edge],
    DataBoundLayer[MarkersProtocol, XYData],
    Generic[_Face, _Edge, _Size],
):
    events: MarkersLayerEvents
    _events_class = MarkersLayerEvents

    def __init__(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 15.0,
        color: ColorType = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.update(symbol=symbol, size=size, color=color, pattern=pattern, alpha=alpha)
        self._size_is_array = False
        self.edge.color = color
        if not self.symbol.has_face():
            self.edge.update(width=1.0, color=color)
        pad_r = size / 400
        self._x_hint, self._y_hint = xy_size_hint(xdata, ydata, pad_r, pad_r)

        self._backend._plt_connect_pick_event(self.events.picked.emit)

    @classmethod
    def empty(
        cls, backend: Backend | str | None = None
    ) -> Markers[ConstFace, ConstEdge, float]:
        """Return an empty markers layer."""
        return cls([], [], backend=backend)

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
        pad_r = self.size / 400
        self._x_hint, self._y_hint = xy_size_hint(x0, y0, pad_r, pad_r)

    def set_data(
        self,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
    ):
        self.data = XYData(xdata, ydata)

    @property
    def symbol(self) -> Symbol:
        """Symbol used to mark the data points."""
        return self._backend._plt_get_symbol()

    @symbol.setter
    def symbol(self, symbol: str | Symbol):
        sym = Symbol(symbol)
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
        if not isinstance(size, (float, int, np.number)):
            if not self._size_is_array:
                raise ValueError("Expected size to be a scalar")
            size = as_array_1d(size)
            if size.size != self.ndata:
                raise ValueError(
                    "Expected size to have the same size as the data, "
                    f"got {size.size} and {self.ndata}"
                )
        self._backend._plt_set_symbol_size(size)
        self.events.size.emit(size)

    def update(
        self,
        *,
        symbol: Symbol | str | _Void = _void,
        size: float | _Void = _void,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        pattern: str | FacePattern | _Void = _void,
    ) -> Markers[_Face, _Edge, _Size]:
        """Update the properties of the markers."""
        if symbol is not _void:
            self.symbol = symbol
        if size is not _void:
            self.size = size
        if color is not _void:
            self.face.color = color
        if pattern is not _void:
            self.face.pattern = pattern
        if alpha is not _void:
            self.face.alpha = alpha
        return self

    def with_hover_text(self, text: Iterable[Any]) -> Markers[_Face, _Edge, _Size]:
        """Add hover text to the markers."""
        texts = [str(t) for t in text]
        if len(texts) != self.ndata:
            raise ValueError(
                "Expected text to have the same size as the data, "
                f"got {len(texts)} and {self.ndata}"
            )
        self._backend._plt_set_hover_text(texts)
        return self

    def with_hover_template(
        self, template: str, **kwargs
    ) -> Markers[_Face, _Edge, _Size]:
        """Add hover template to the markers."""
        xs, ys = self.data
        custom_keys = list(kwargs.keys())
        custom_values = [kwargs[k] for k in custom_keys]
        if "x" in custom_keys or "y" in custom_keys or "i" in custom_keys:
            raise ValueError("x, y and i are reserved formats.")
        texts = []
        for i in range(xs.size):
            others = {k: v[i] for k, v in zip(custom_keys, custom_values)}
            texts.append(template.format(x=xs[i], y=ys[i], i=i, **others))
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
        from whitecanvas.layers.group import LabeledMarkers
        from whitecanvas.layers._primitive import Errorbars

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
            backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name)
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
        from whitecanvas.layers.group import LabeledMarkers
        from whitecanvas.layers._primitive import Errorbars

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
            backend=self._backend_name
        )  # fmt: skip
        xerr = Errorbars.empty(Orientation.HORIZONTAL, backend=self._backend_name)
        return LabeledMarkers(self, xerr, yerr, name=self.name)

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
    ) -> _lg.LabeledMarkers:
        from whitecanvas.layers import Errorbars
        from whitecanvas.layers.group import LabeledMarkers

        if isinstance(strings, str):
            strings = [strings] * self.data.x.size
        else:
            strings = list(strings)
            if len(strings) != self.data.x.size:
                raise ValueError(
                    f"Number of strings ({len(strings)}) does not match the "
                    f"number of data ({self.data.x.size})."
                )
        texts = Texts(
            *self.data, strings, color=color, size=size, rotation=rotation,
            anchor=anchor, family=fontfamily, backend=self._backend_name,
        )  # fmt: skip
        return LabeledMarkers(
            self,
            Errorbars.empty(Orientation.HORIZONTAL, backend=self._backend_name),
            Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name),
            texts=texts,
            name=self.name,
        )

    def with_network(
        self,
        connections: NDArray[np.intp],
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
            nodes[:, 0],
            nodes[:, 1],
            [""] * nodes.shape[0],
            name="texts",
            backend=self._backend_name,
        )
        return Graph(self, edges_layer, texts, edges, name=self.name)

    def with_stem(
        self,
        orient: str | Orientation = Orientation.VERTICAL,
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
        orient : str or Orientation, default is vertical
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
        from whitecanvas.layers.group import StemPlot
        from whitecanvas.layers._primitive import MultiLine

        ori = Orientation.parse(orient)
        xdata, ydata = self.data
        if bottom is None:
            bottom = np.zeros_like(ydata)
        elif isinstance(bottom, (float, int, np.number)):
            bottom = np.full_like(ydata, bottom)
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
        color: ColorType | None = None,
        pattern: FacePattern | str = FacePattern.SOLID,
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
        pattern : str or FacePattern, optional
            Pattern (hatch) of the faces.
        alpha : float, optional
            Alpha channel of the faces.

        Returns
        -------
        Markers
            The updated markers layer.
        """
        super().with_face(color, pattern, alpha)
        if not self.symbol.has_face():
            width = self.edge.width
            if isinstance(width, (float, int, np.number)):
                self.edge.update(width=width or 1.0, color=color)
            else:
                self.edge.update(width=width[0] or 1.0, color=color)
        return self

    def with_face_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        pattern: str | FacePattern | Sequence[str | FacePattern] = FacePattern.SOLID,
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
        pattern : str or FacePattern or sequence of it, optional
            Pattern(s) of the faces.
        alpha : float or sequence of float, optional
            Alpha channel(s) of the faces.

        Returns
        -------
        Markers
            The updated markers layer.
        """
        super().with_face_multi(color, pattern, alpha)
        if not self.symbol.has_face():
            width = self.edge.width
            if isinstance(width, (float, int, np.number)):
                self.edge.update(width=width or 1.0, color=color)
            else:
                self.edge.update(width=width[0] or 1.0, color=color)
        return self

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Markers[_Face, ConstEdge, _Size]:
        return super().with_edge(color, width, style, alpha)

    def with_edge_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1.0,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Markers[_Face, MultiEdge, _Size]:
        return super().with_edge_multi(color, width, style, alpha)

    def with_size_multi(
        self,
        size: float | Sequence[float],
    ) -> Markers[_Face, _Edge, NDArray[np.float32]]:
        if isinstance(size, (float, int, np.number)):
            size = np.full(self.ndata, size, dtype=np.float32)
        else:
            size = as_array_1d(size, dtype=np.float32)
            if size.size != self.ndata:
                raise ValueError(
                    "Expected size to have the same size as the data, "
                    f"got {size.size} and {self.ndata}"
                )
        self._size_is_array = True
        self.size = size
        return self
