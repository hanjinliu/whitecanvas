from __future__ import annotations
from typing import Any, Generic, Iterator, Literal, overload, TypeVar
from abc import ABC, abstractmethod

from cmap import Color

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas import protocols
from whitecanvas import layers as _l
from whitecanvas.types import LineStyle, Symbol, ColorType
from whitecanvas.canvas import canvas_namespace as _ns, layerlist as _ll
from whitecanvas.utils.normalize import as_array_1d, normalize_xy
from whitecanvas.backend import Backend


_T = TypeVar("_T", bound=protocols.HasVisibility)
_L = TypeVar("_L", bound=_l.Layer)


class CanvasBase(ABC, Generic[_T]):
    """Base class for any canvas object."""

    title = _ns.TitleNamespace()
    x = _ns.XAxisNamespace()
    y = _ns.YAxisNamespace()
    layers = _ll.LayerList()
    mouse = _ns.MouseNamespace()

    def __init__(self, backend: str | None = None):
        self._backend_installer = Backend(backend)
        self._backend = self._create_backend()

        # default colors
        self.x.color = "black"
        self.y.color = "black"
        self.background_color = "white"

        # connect layer events
        self.layers.events.inserted.connect(self._cb_inserted)
        self.layers.events.removed.connect(self._cb_removed)
        self.layers.events.reordered.connect(self._cb_reordered)
        self._is_grouping = False

        canvas = self._canvas()
        canvas._plt_connect_xlim_changed(self.x.lim_changed.emit)
        canvas._plt_connect_ylim_changed(self.y.lim_changed.emit)
        canvas._plt_connect_mouse_click(self.mouse.clicked.emit)
        canvas._plt_connect_mouse_click(self.mouse.moved.emit)
        canvas._plt_connect_mouse_drag(self.mouse.moved.emit)
        canvas._plt_connect_mouse_double_click(self.mouse.double_clicked.emit)
        canvas._plt_connect_mouse_double_click(self.mouse.moved.emit)

    @abstractmethod
    def _create_backend(self) -> _T:
        """Create a backend object."""

    @abstractmethod
    def _canvas(self) -> protocols.CanvasProtocol:
        """Return the canvas object."""

    @property
    def background_color(self):
        return self._canvas()._plt_get_background_color()

    @background_color.setter
    def background_color(self, color):
        self._canvas()._plt_set_background_color(np.array(Color(color)))

    def show(self):
        """Show the canvas."""
        self._backend._plt_set_visible(True)

    def hide(self):
        """Hide the canvas."""
        self._backend._plt_set_visible(False)

    def screenshot(self):
        return self._canvas()._plt_screenshot()

    @overload
    def add_line(
        self, ydata: ArrayLike, *, name: str | None = None,
        color: ColorType = "blue", width: float = 1.0, style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0, antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    @overload
    def add_line(
        self, xdata: ArrayLike, ydata: ArrayLike, *, name: str | None = None,
        color="blue", width: float = 1.0, style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0, antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    def add_line(self, *args, name=None, color="blue", width=1.0, style=LineStyle.SOLID, antialias=True):
        """
        Add a Line layer to the canvas.

        >>> canvas.add_line(y, ...)
        >>> canvas.add_line(x, y, ...)

        Parameters
        ----------
        name : str, optional
            Name of the layer.

        Returns
        -------
        Line
            The line layer.
        """
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(_l.Line, name)
        layer = _l.Line(
            xdata, ydata, name=name, color=color, line_width=width, line_style=style,
            antialias=antialias, backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_markers(
        ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, face_color: Any = "blue", edge_color: Any = "black",
        edge_width: float =0, edge_style: LineStyle | str = LineStyle.SOLID,
    ):  # fmt: skip
        ...

    @overload
    def add_markers(
        xdata: ArrayLike, ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, face_color: Any = "blue", edge_color: Any = "black",
        edge_width: float =0, edge_style: LineStyle | str = LineStyle.SOLID,
    ):  # fmt: skip
        ...

    def add_markers(
        self,
        *args,
        name=None,
        symbol=Symbol.CIRCLE,
        size=6,
        face_color="blue",
        edge_color="black",
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ):
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(_l.Markers, name)
        layer = _l.Markers(
            xdata, ydata, name=name, symbol=symbol, size=size, face_color=face_color,
            edge_color=edge_color, edge_width=edge_width, edge_style=edge_style,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_bars(
        self,
        center: ArrayLike,
        top: ArrayLike,
        bottom: ArrayLike | None = None,
        *,
        name=None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        bar_width: float = 0.8,
        face_color="blue",
        edge_color="black",
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ):
        name = self._coerce_name(_l.Bars, name)
        layer = _l.Bars(
            center, top, bottom, bar_width=bar_width, name=name, orient=orient,
            face_color=face_color, edge_color=edge_color, edge_width=edge_width,
            edge_style=edge_style, backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_hist(
        self,
        data: ArrayLike,
        *,
        bins: int | ArrayLike | None = None,
        range: tuple[float, float] | None = None,
        density: bool = False,
        name: str | None = None,
        face_color="blue",
        edge_color="black",
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ):
        data = as_array_1d(data)
        name = self._coerce_name("histogram", name)
        counts, edges = np.histogram(data, bins, density=density, range=range)
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]
        layer = _l.Bars(
            centers, counts, bar_width=width, name=name, face_color=face_color,
            edge_color=edge_color, edge_width=edge_width, edge_style=edge_style,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_infcurve(
        self,
        model,
        params: dict[str, Any] = {},
        bounds: tuple[float, float] = (-np.inf, np.inf),
        name=None,
        color="blue",
        width=1.0,
        style=LineStyle.SOLID,
        antialias=True,
    ):
        name = self._coerce_name(_l.InfCurve, name)
        layer = _l.InfCurve(
            model, params=params, bounds=bounds, name=name, color=color,
            line_width=width, line_style=style, antialias=antialias,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_fillbetween(
        self,
        xdata: ArrayLike,
        ydata0: ArrayLike,
        ydata1: ArrayLike,
        *,
        name: str | None = None,
        face_color="blue",
        edge_color="black",
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ):
        name = self._coerce_name(_l.Band, name)
        layer = _l.Band(
            xdata, ydata0, ydata1, name=name, face_color=face_color,
            edge_color=edge_color, edge_width=edge_width, edge_style=edge_style,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_errorbars(
        self,
        xdata: ArrayLike,
        ylow: ArrayLike,
        yhigh: ArrayLike,
        *,
        name: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        color: ColorType = "blue",
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        capsize: float = 0.0,
    ):
        name = self._coerce_name(_l.Errorbars, name)
        layer = _l.Errorbars(
            xdata, ylow, yhigh, name=name, color=color, line_width=width,
            line_style=style, antialias=antialias, capsize=capsize,
            orient=orient, backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_layer(self, layer: _L) -> _L:
        """Add a layer to the canvas."""
        self.layers.append(layer)
        return layer

    def _coerce_name(self, layer_type: type[_l.Layer] | str, name: str | None) -> str:
        if name is None:
            if isinstance(layer_type, str):
                name = layer_type
            else:
                name = layer_type.__name__.lower()
        basename = name
        i = 0
        _exists = {layer.name for layer in self.layers}
        while name in _exists:
            name = f"{basename}-{i}"
            i += 1
        return name

    def _cb_inserted(self, idx: int, layer: _l.Layer):
        if self._is_grouping:
            return
        if idx < 0:
            idx = len(self.layers) + idx
        _canvas = self._canvas()
        for l in _iter_layers(layer):
            _canvas._plt_insert_layer(idx, l._backend)
            l._connect_canvas(self)

    def _cb_removed(self, idx: int, layer: _l.Layer):
        if self._is_grouping:
            return
        _canvas = self._canvas()
        for l in _iter_layers(layer):
            _canvas._plt_remove_layer(l._backend)
            l._disconnect_canvas(self)

    def _cb_reordered(self):
        zorder = 0
        for layer in self.layers:
            if isinstance(layer, _l.PrimitiveLayer):
                layer._backend._plt_set_zorder(zorder)
            elif isinstance(layer, _l.LayerGroup):
                for child in layer._iter_children():
                    child._backend._plt_set_zorder(zorder)

    def _group_layers(self, group: _l.LayerGroup):
        # TODO: do not remove the backend objects!
        indices: list[int] = []
        not_found: list[_l.PrimitiveLayer] = []
        for layer in group._iter_children():
            try:
                idx = self.layers.index(layer)
                indices.append(idx)
            except ValueError:
                not_found.extend(_iter_layers(layer))
        if len(indices) > 0:
            self._is_grouping = True
            try:
                for idx in reversed(indices):
                    self.layers.pop(idx)
                self.layers.append(group)
                _canvas = self._canvas()
                for child in not_found:
                    _canvas._plt_insert_layer(idx, child._backend)
                    child._connect_canvas(self)
            finally:
                self._is_grouping = False


def _iter_layers(layer: _l.Layer) -> Iterator[_l.PrimitiveLayer[protocols.BaseProtocol]]:
    if isinstance(layer, _l.PrimitiveLayer):
        yield layer
    elif isinstance(layer, _l.LayerGroup):
        for child in layer._iter_children():
            yield from _iter_layers(child)
    else:
        raise TypeError(f"Unknown layer type: {type(layer).__name__}")
