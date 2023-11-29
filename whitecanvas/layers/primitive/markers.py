from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.protocols import MarkersProtocol, HeteroMarkersProtocol
from whitecanvas.layers._base import XYData, PrimitiveLayer
from whitecanvas.layers._sizehint import xy_size_hint
from whitecanvas.layers._mixin import FaceEdgeMixin, HeteroFaceEdgeMixin
from whitecanvas.backend import Backend
from whitecanvas.types import (
    Symbol,
    ColorType,
    FacePattern,
    _Void,
    Alignment,
    Orientation,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xy

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg
    from typing_extensions import Self
    from whitecanvas.layers._mixin import (
        FaceNamespace,
        EdgeNamespace,
        MultiFaceNamespace,
        MultiEdgeNamespace,
    )

_void = _Void()


class MarkersBase(PrimitiveLayer[MarkersProtocol | HeteroMarkersProtocol]):
    face: FaceNamespace | MultiFaceNamespace
    edge: EdgeNamespace | MultiEdgeNamespace

    def __init__(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        color: ColorType = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.name = name if name is not None else "Line"
        self.update(
            symbol=symbol, size=size, color=color, pattern=pattern, alpha=alpha
        )  # fmt: skip
        self.edge.color = color
        if not self.symbol.has_face():
            self.edge.width = 1.0
        pad_r = size / 400
        self._x_hint, self._y_hint = xy_size_hint(xdata, ydata, pad_r, pad_r)

    @classmethod
    def empty(cls, backend: Backend | str | None = None) -> Self:
        """Return an empty markers layer."""
        # TODO: not works with size 0
        return cls([], [], backend=backend)

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data.x.size

    @property
    def data(self) -> XYData:
        """Current data of the layer."""
        return XYData(*self._backend._plt_get_data())

    def set_data(
        self,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
    ):
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
        self._backend._plt_set_data(x0, y0)
        pad_r = self.size / 400
        self._x_hint, self._y_hint = xy_size_hint(x0, y0, pad_r, pad_r)

    @property
    def symbol(self) -> Symbol:
        """Symbol used to mark the data points."""
        return self._backend._plt_get_symbol()

    @symbol.setter
    def symbol(self, symbol: str | Symbol):
        self._backend._plt_set_symbol(Symbol(symbol))

    def update(
        self,
        *,
        symbol: Symbol | str | _Void = _void,
        size: float | _Void = _void,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        pattern: str | FacePattern | _Void = _void,
    ) -> Self:
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
        from whitecanvas.layers.group import LabeledMarkers
        from whitecanvas.layers.primitive import Errorbars

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
        from whitecanvas.layers.group import LabeledMarkers
        from whitecanvas.layers.primitive import Errorbars

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
        from whitecanvas.layers.group import TextGroup, LabeledMarkers

        if isinstance(strings, str):
            strings = [strings] * self.data.x.size
        else:
            strings = list(strings)
            if len(strings) != self.data.x.size:
                raise ValueError(
                    f"Number of strings ({len(strings)}) does not match the "
                    f"number of data ({self.data.x.size})."
                )
        texts = TextGroup.from_strings(
            *self.data, strings, color=color, size=size, rotation=rotation,
            anchor=anchor, fontfamily=fontfamily, backend=self._backend_name,
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
        from whitecanvas.layers.primitive import MultiLine
        from whitecanvas.layers.group import Graph, TextGroup

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
        texts = TextGroup.from_strings(
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
        from whitecanvas.layers.primitive import MultiLine

        ori = Orientation.parse(orient)
        xdata, ydata = self.data
        if ori.is_vertical:
            root = np.stack([xdata, np.zeros_like(ydata)], axis=1)
            leaf = np.stack([xdata, ydata], axis=1)
        else:
            root = np.stack([np.zeros_like(xdata), ydata], axis=1)
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


class Markers(MarkersBase, FaceEdgeMixin[MarkersProtocol]):
    if TYPE_CHECKING:

        def __init__(
            self,
            xdata: ArrayLike,
            ydata: ArrayLike,
            *,
            name: str | None = None,
            symbol: Symbol | str = Symbol.CIRCLE,
            size: float = 10,
            color: ColorType = "blue",
            alpha: float = 1.0,
            pattern: str | FacePattern = FacePattern.SOLID,
            backend: Backend | str | None = None,
        ):
            ...

    @property
    def size(self) -> float:
        """Size of the symbol."""
        return self._backend._plt_get_symbol_size()

    @size.setter
    def size(self, size: float):
        """Set marker size"""
        self._backend._plt_set_symbol_size(size)


class HeteroMarkers(MarkersBase, HeteroFaceEdgeMixin[HeteroMarkersProtocol]):
    if TYPE_CHECKING:

        def __init__(
            self,
            xdata: ArrayLike,
            ydata: ArrayLike,
            *,
            name: str | None = None,
            symbol: Symbol | str = Symbol.CIRCLE,
            size: float | Sequence[float] = 10,
            color: ColorType | Sequence[ColorType] = "blue",
            alpha: float = 1.0,
            pattern: str
            | FacePattern
            | Sequence[str | FacePattern] = FacePattern.SOLID,
            backend: Backend | str | None = None,
        ):
            ...

    @property
    def size(self) -> float:
        """Size of the symbol."""
        return self._backend._plt_get_symbol_size()

    @size.setter
    def size(self, size: float | NDArray[np.floating]):
        """Set marker size"""
        if not isinstance(size, (float, int, np.number)):
            size = np.asarray(size)
        self._backend._plt_set_symbol_size(size)
