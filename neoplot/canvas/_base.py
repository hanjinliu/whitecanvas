from __future__ import annotations
from typing import Any, Generic, overload, TypeVar
from abc import ABC, abstractmethod

from cmap import Color

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neoplot import protocols
from neoplot import layers
from neoplot.layers import Layer
from neoplot.types import LineStyle, Symbol
from neoplot.canvas import canvas_namespace as _ns, layerlist as _ll
from neoplot.utils.normalize import normalize_xy
from neoplot.backend import Backend


_T = TypeVar("_T", bound=protocols.HasVisibility)
_L = TypeVar("_L", bound=Layer)


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
        color="blue", width: float = 1.0, style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0, antialias: bool = True,
    ) -> layers.Line:  # fmt: skip
        ...

    @overload
    def add_line(
        self, xdata: ArrayLike, ydata: ArrayLike, *, name: str | None = None,
        color="blue", width: float = 1.0, style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0, antialias: bool = True,
    ) -> layers.Line:  # fmt: skip
        ...

    def add_line(
        self, *args, name=None, color="blue", width=1.0, style=LineStyle.SOLID, alpha: float = 1.0, antialias=True
    ):
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
        name = self._coerce_name(layers.Line, name)
        layer = layers.Line(
            xdata, ydata, name=name, color=color, width=width, style=style,
            antialias=antialias, backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_scatter(
        ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, face_color: Any = "blue", edge_color: Any = "black",
        edge_width: float =0, edge_style: LineStyle | str = LineStyle.SOLID,
    ):  # fmt: skip
        ...

    @overload
    def add_scatter(
        xdata: ArrayLike, ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, face_color: Any = "blue", edge_color: Any = "black",
        edge_width: float =0, edge_style: LineStyle | str = LineStyle.SOLID,
    ):  # fmt: skip
        ...

    def add_scatter(
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
        name = self._coerce_name(layers.Scatter, name)
        layer = layers.Scatter(
            xdata, ydata, name=name, symbol=symbol, size=size, face_color=face_color,
            edge_color=edge_color, edge_width=edge_width, edge_style=edge_style,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_bar(
        self,
        *args,
        name=None,
        width: float = 0.8,
        face_color="blue",
        edge_color="black",
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ):
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(layers.Bar, name)
        layer = layers.Bar(
            xdata, ydata, width=width, name=name, face_color=face_color,
            edge_color=edge_color, edge_width=edge_width, edge_style=edge_style,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_layer(self, layer: _L) -> _L:
        """Add a layer to the canvas."""
        self.layers.append(layer)
        return layer

    def _coerce_name(self, layer_type: type[Layer], name: str | None) -> str:
        if name is None:
            name = layer_type.__name__
        basename = name
        i = 0
        _exists = {layer.name for layer in self.layers}
        while name in _exists:
            name = f"{basename}-{i}"
            i += 1
        return name

    def _cb_inserted(self, idx: int, layer: Layer):
        if idx < 0:
            idx = len(self.layers) + idx
        self._canvas()._plt_insert_layer(idx, layer._backend)
        layer._connect_canvas(self)

    def _cb_removed(self, idx: int, layer: Layer):
        self._canvas()._plt_remove_layer(layer._backend)
        layer._disconnect_canvas(self)

    def _cb_reordered(self):
        for i, layer in enumerate(self.layers):
            layer._backend._plt_set_zorder(i)


def _as_array_1d(x: ArrayLike) -> NDArray[np.number]:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got {x.ndim}D array")
    if x.dtype.kind not in "iuf":
        raise ValueError(f"Input {x!r} did not return a numeric array")
    return x
