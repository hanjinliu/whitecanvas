from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers import Layer, _legend
from whitecanvas.layers._mixin import (
    EnumArray,
    MultiEdge,
    MultiFace,
    SinglePropertyEdgeBase,
    SinglePropertyFaceBase,
)
from whitecanvas.layers._primitive import Markers
from whitecanvas.layers.group._collections import (
    LayerCollectionBase,
    LayerContainerEvents,
)
from whitecanvas.theme import get_theme
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    Hatch,
    LineStyle,
    Symbol,
    XYData,
    _Void,
)
from whitecanvas.utils.normalize import (
    as_any_1d_array,
    as_array_1d,
    as_color_array,
    parse_texts,
)
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

_void = _Void()
_Markers = Markers[MultiFace, MultiEdge, NDArray[np.floating]]


class MarkerCollectionEvents(LayerContainerEvents):
    size = Signal(object)
    symbol = Signal(object)


class MarkerCollectionFace(SinglePropertyFaceBase):
    _layer: MarkerCollection

    def _iter_markers(self) -> Iterator[tuple[NDArray[np.bool_], _Markers]]:
        for i, layer in enumerate(self._layer):
            yield self._layer._slices[i], layer

    @property
    def color(self) -> NDArray[np.floating]:
        """Face color of the markers."""
        out = np.empty((self._layer.ndata, 4), dtype=np.float32)
        for sl, layer in self._iter_markers():
            out[sl] = layer.face.color
        return out

    @color.setter
    def color(self, color):
        colors = as_color_array(color, self._layer.ndata)
        for sl, layer in self._iter_markers():
            layer.face.color = colors[sl]
        self.events.color.emit(colors)

    @property
    def hatch(self) -> EnumArray[Hatch]:
        """Face fill hatch."""
        out = np.empty(self._layer.ndata, dtype=object)
        for sl, layer in self._iter_markers():
            out[sl] = layer.face.hatch
        return out

    @hatch.setter
    def hatch(self, hatch: str | Hatch | Iterable[str | Hatch]):
        if isinstance(hatch, (str, Hatch)):
            h = Hatch(hatch)
            for layer in self._layer:
                layer.face.hatch = h
            self.events.hatch.emit(h)
        else:
            _hatches = np.array([Hatch(h) for h in hatch], dtype=object)
            markers = list(self._layer)
            for i, layer in enumerate(markers):
                sl = self._layer._slices[i]
                _hat = _hatches[sl]
                _hat_unique = list(set(_hat))
                if len(_hat_unique) == 1:
                    layer.face.hatch = _hat_unique[0]
                else:
                    new_mks = self._layer._split_markers(i, _hat, _hat_unique)
                    for mk, each in zip(new_mks, _hat_unique):
                        mk.face.hatch = each
            self.events.hatch.emit(_hatches)

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> MarkerCollection:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        hatch : FacePattern, optional
            Fill hatch of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if hatch is not _void:
            self.hatch = hatch
        if alpha is not _void:
            self.alpha = alpha
        return self._layer

    def _as_legend_info(self):
        if self.color.size == 0:
            return _legend.EmptyLegendItem()
        color = self.color[0]
        hatch = self.hatch[0]
        return _legend.FaceInfo(color, hatch)


class MarkerCollectionEdge(SinglePropertyEdgeBase):
    _layer: MarkerCollection

    def _iter_markers(self) -> Iterator[tuple[NDArray[np.bool_], _Markers]]:
        for i, layer in enumerate(self._layer):
            yield self._layer._slices[i], layer

    @property
    def color(self) -> NDArray[np.floating]:
        """Edge color of the markers."""
        out = np.empty((self._layer.ndata, 4), dtype=np.float32)
        for sl, layer in self._iter_markers():
            out[sl] = layer.edge.color
        return out

    @color.setter
    def color(self, color):
        colors = as_color_array(color, self._layer.ndata)
        for sl, layer in self._iter_markers():
            layer.edge.color = colors[sl]
        self.events.color.emit(colors)

    @property
    def width(self) -> NDArray[np.float32]:
        """Edge widths."""
        out = np.empty(self._layer.ndata, dtype=np.float32)
        for sl, layer in self._iter_markers():
            out[sl] = layer.edge.width
        return out

    @width.setter
    def width(self, width: float | Iterable[float]):
        if is_real_number(width):
            for _, layer in self._iter_markers():
                layer.edge.width = width
            self.events.width.emit(width)
        else:
            widths = as_array_1d(width)
            for sl, layer in self._iter_markers():
                layer.edge.width = widths[sl]
            self.events.width.emit(widths)

    @property
    def style(self) -> EnumArray[LineStyle]:
        """Edge line style."""
        out = np.empty(self._layer.ndata, dtype=object)
        for sl, layer in self._iter_markers():
            out[sl] = layer.edge.style
        return out

    @style.setter
    def style(self, style: str | Hatch | Iterable[str | Hatch]):
        styles = as_any_1d_array(style, self._layer.ndata, dtype=object)
        for sl, layer in self._iter_markers():
            layer.edge.style = styles[sl]
        self.events.style.emit(styles)

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer

    def _as_legend_info(self):
        if self.color.size == 0:
            return _legend.EmptyLegendItem()
        color = self.color[0]
        width = self.width[0]
        style = self.style[0]
        return _legend.EdgeInfo(color, width, style)


class MarkerCollection(LayerCollectionBase[_Markers]):
    events: MarkerCollectionEvents
    _events_class = MarkerCollectionEvents

    def __init__(
        self,
        layers: list[_Markers],
        *more_layers: _Markers,
        name: str | None = None,
    ):
        if isinstance(layers, Layer):
            layers = [layers]
        layers = list(layers) + list(more_layers)
        super().__init__(layers, name=name)
        self._face_namespace = MarkerCollectionFace(self)
        self._edge_namespace = MarkerCollectionEdge(self)
        sizes = [layer.ndata for layer in layers]
        cum_size = np.cumsum([0, *sizes])
        ndata = cum_size[-1]
        self._slices = [np.zeros(ndata, dtype=np.bool_) for _ in sizes]
        for i, sl in enumerate(self._slices):
            sl[cum_size[i] : cum_size[i + 1]] = True

    @classmethod
    def from_arrays(
        cls,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        name: str | None = None,
        backend: str | Backend | None = None,
    ) -> Self:
        markers = Markers(xdata, ydata, backend=backend)._as_all_multi()
        return cls([markers], name=name)

    @property
    def face(self) -> MarkerCollectionFace:
        return self._face_namespace

    @property
    def edge(self) -> MarkerCollectionEdge:
        return self._edge_namespace

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return sum(layer.ndata for layer in self)

    @property
    def data(self) -> XYData:
        """All the data of the markers."""
        ndata = self.ndata
        xs = np.empty(ndata, dtype=np.float32)
        ys = np.empty(ndata, dtype=np.float32)
        for i, layer in enumerate(self):
            data = layer.data
            sl = self._slices[i]
            xs[sl] = data.x
            ys[sl] = data.y
        return XYData(xs, ys)

    @data.setter
    def data(self, data: XYData):
        data = self._norm_layer_data(data)
        for i, layer in enumerate(self):
            sl = self._slices[i]
            x0 = data.x[sl]
            y0 = data.y[sl]
            layer.data = XYData(x0, y0)

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

    @property
    def symbol(self) -> EnumArray[Symbol]:
        """Symbol of the markers."""
        out = np.empty(self.ndata, dtype=object)
        for i, layer in enumerate(self):
            sl = self._slices[i]
            out[sl] = layer.symbol
        return out

    @symbol.setter
    def symbol(self, symbol: str | Symbol | Iterable[str | Symbol]):
        if isinstance(symbol, (str, Symbol)):
            sym = Symbol(symbol)
            for layer in self:
                layer.symbol = sym
            self.events.symbol.emit(sym)
        else:
            _symbols = np.array([Symbol(s) for s in symbol], dtype=object)
            markers = list(self)
            for i, layer in enumerate(markers):
                sl = self._slices[i]
                _sym = _symbols[sl]
                _sym_unique = list(set(_sym))
                if len(_sym_unique) == 1:
                    layer.symbol = _sym_unique[0]
                else:
                    new_mks = self._split_markers(i, _sym, _sym_unique)
                    for mk, each in zip(new_mks, _sym_unique):
                        mk.symbol = each
            self.events.symbol.emit(_symbols)

    @property
    def size(self) -> NDArray[np.float32]:
        out = np.empty(self.ndata, dtype=np.float32)
        for i, layer in enumerate(self):
            sl = self._slices[i]
            out[sl] = layer.size
        return out

    @size.setter
    def size(self, size: float | Iterable[float]):
        if is_real_number(size):
            for i, layer in enumerate(self):
                sl = self._slices[i]
                layer.size = size
            self.events.size.emit(size)
        else:
            size = as_array_1d(size)
            for i, layer in enumerate(self):
                sl = self._slices[i]
                layer.size = size[sl]
            self.events.size.emit(size)

    def with_face(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the face properties."""
        self.face.update(color=color, hatch=hatch, alpha=alpha)
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = get_theme().foreground_color
        self.edge.update(color=color, style=style, width=width, alpha=alpha)
        return self

    def with_hover_text(self, text: str | Iterable[Any]) -> Self:
        """Add hover text to the markers."""
        if isinstance(text, str):
            texts = np.full(self.ndata, text, dtype=object)
        else:
            texts = np.array([str(t) for t in text], dtype=object)
        if len(texts) != self.ndata:
            raise ValueError(
                "Expected text to have the same size as the data, "
                f"got {len(texts)} and {self.ndata}"
            )
        return self._set_hover_text_unsafe(texts)

    def with_hover_template(self, template: str, extra: Any | None = None) -> Self:
        """Add hover template to the markers."""
        xs, ys = self.data
        if xs.size == 0:  # empty layer
            return self
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
        texts = np.array(texts, dtype=object)
        return self._set_hover_text_unsafe(texts)

    def _set_hover_text_unsafe(self, texts: NDArray[np.object_]):
        for i, layer in enumerate(self):
            sl = self._slices[i]
            layer._backend._plt_set_hover_text(texts[sl])
        return self

    def _split_markers(
        self,
        idx: int,
        spec: NDArray[Any],
        spec_unique: NDArray[Any] | None = None,
    ) -> list[_Markers]:
        """Split the marker at index i."""
        layer = self[idx]
        all_data = layer.data
        if spec_unique is None:
            spec_unique = set(spec)
        new_markers: list[_Markers] = []
        new_slices = self._slices.copy()
        new_slices.pop(idx)
        for each in spec_unique:
            mask = spec == each
            markers = Markers(
                all_data.x[mask],
                all_data.y[mask],
                symbol=layer.symbol,
                backend=self._backend_name,
            )
            markers.with_size_multi(
                layer.size[mask],
            ).with_face_multi(
                color=layer.face.color[mask],
                hatch=layer.face.hatch[mask],
            ).with_edge_multi(
                color=layer.edge.color[mask],
                width=layer.edge.width[mask],
                style=layer.edge.style[mask],
            )
            new_markers.append(markers)
            new_slices.append(mask)

        self.pop(idx)
        self.extend(new_markers)
        self._slices = new_slices
        return new_markers

    def _as_legend_item(self):
        if self._children:
            return self._children[0]._as_legend_item()
        return _legend.EmptyLegendItem()
