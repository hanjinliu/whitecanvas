from __future__ import annotations
from typing import Any, Callable, Generic, Iterator, Literal, overload, TypeVar
from abc import ABC, abstractmethod

from cmap import Color, Colormap

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas import protocols
from whitecanvas import layers as _l
from whitecanvas.layers import group as _lg
from whitecanvas.types import LineStyle, Symbol, ColorType, Alignment, ColormapType
from whitecanvas.canvas import canvas_namespace as _ns, layerlist as _ll
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.utils.normalize import as_array_1d, norm_color, normalize_xy
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

    def __init__(
        self,
        backend: str | None = None,
        *,
        palette: ColormapType = "tab10",
    ):
        self._backend_installer = Backend(backend)
        self._backend = self._create_backend()
        self._color_palette = ColorPalette(palette)

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
    def background_color(self) -> NDArray[np.floating]:
        """Background color of the canvas."""
        return norm_color(self._canvas()._plt_get_background_color())

    @background_color.setter
    def background_color(self, color):
        self._canvas()._plt_set_background_color(np.array(Color(color)))

    @property
    def aspect_ratio(self) -> float | None:
        return self._canvas()._plt_get_aspect_ratio()

    @aspect_ratio.setter
    def aspect_ratio(self, ratio: float | None):
        if ratio is not None:
            ratio = float(ratio)
        self._canvas()._plt_set_aspect_ratio(ratio)

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
        self, ydata: ArrayLike, *, name: str | None = None, color: ColorType | None = None,
        line_width: float = 1.0, line_style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    @overload
    def add_line(
        self, xdata: ArrayLike, ydata: ArrayLike, *, name: str | None = None,
        color: ColorType | None = None, line_width: float = 1.0,
        line_style: LineStyle | str = LineStyle.SOLID, antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    def add_line(
        self,
        *args,
        name=None,
        color=None,
        line_width=1.0,
        line_style=LineStyle.SOLID,
        antialias=True,
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
        name = self._coerce_name(_l.Line, name)
        (color,) = self._generate_colors(color)
        layer = _l.Line(
            xdata, ydata, name=name, color=color, line_width=line_width, line_style=line_style,
            antialias=antialias, backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_markers(
        ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, face_color: ColorType | None = None, edge_color: ColorType | None = None,
        edge_width: float =0, edge_style: LineStyle | str = LineStyle.SOLID,
    ) -> _l.Markers:  # fmt: skip
        ...

    @overload
    def add_markers(
        xdata: ArrayLike, ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, face_color: ColorType | None = None, edge_color: ColorType | None = None,
        edge_width: float =0, edge_style: LineStyle | str = LineStyle.SOLID,
    ) -> _l.Markers:  # fmt: skip
        ...

    def add_markers(
        self,
        *args,
        name=None,
        symbol=Symbol.CIRCLE,
        size=6,
        face_color=None,
        edge_color=None,
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ):
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(_l.Markers, name)
        face_color, edge_color = self._generate_colors(face_color, edge_color)
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
        face_color: ColorType | None = None,
        edge_color: ColorType | None = None,
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ):
        name = self._coerce_name(_l.Bars, name)
        face_color, edge_color = self._generate_colors(face_color, edge_color)
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
        face_color: ColorType | None = None,
        edge_color: ColorType | None = None,
        edge_width: float = 0,
        edge_style: str | LineStyle = LineStyle.SOLID,
    ):
        data = as_array_1d(data)
        name = self._coerce_name("histogram", name)
        counts, edges = np.histogram(data, bins, density=density, range=range)
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]
        face_color, edge_color = self._generate_colors(face_color, edge_color)
        layer = _l.Bars(
            centers, counts, bar_width=width, name=name, face_color=face_color,
            edge_color=edge_color, edge_width=edge_width, edge_style=edge_style,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_infcurve(
        self,
        model: Callable[..., NDArray[np.floating]],
        params: dict[str, Any] = {},
        bounds: tuple[float, float] = (-np.inf, np.inf),
        name: str | None = None,
        color: ColorType | None = None,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        antialias: bool = True,
    ) -> _l.InfCurve:
        name = self._coerce_name(_l.InfCurve, name)
        (color,) = self._generate_colors(color)
        layer = _l.InfCurve(
            model, params=params, bounds=bounds, name=name, color=color,
            line_width=width, line_style=style, antialias=antialias,
            backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_band(
        self,
        xdata: ArrayLike,
        ydata0: ArrayLike,
        ydata1: ArrayLike,
        *,
        name: str | None = None,
        face_color: ColorType | None = None,
        edge_color: ColorType | None = None,
        edge_width=0,
        edge_style=LineStyle.SOLID,
    ) -> _l.Band:
        name = self._coerce_name(_l.Band, name)
        face_color, edge_color = self._generate_colors(face_color, edge_color)
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
    ) -> _l.Errorbars:
        name = self._coerce_name(_l.Errorbars, name)
        (color,) = self._generate_colors(color)
        layer = _l.Errorbars(
            xdata, ylow, yhigh, name=name, color=color, line_width=width,
            line_style=style, antialias=antialias, capsize=capsize,
            orient=orient, backend=self._backend_installer,
        )  # fmt: skip
        return self.add_layer(layer)

    def add_text(
        self,
        x: float,
        y: float,
        string: str,
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str = "sans-serif",
    ) -> _l.Text:
        """
        Add a text layer to the canvas.

        Parameters
        ----------
        x : float
            X position of the text.
        y : float
            Y position of the text.
        string : str
            Text string to display.
        color : ColorType, optional
            Color of the text string.
        size : float, default is 12
            Point size of the text.
        rotation : float, default is 0.0
            Rotation angle of the text in degrees.
        anchor : str or Alignment, default is Alignment.BOTTOM_LEFT
            Anchor position of the text. The anchor position will be the coordinate
            given by (x, y).
        fontfamily : str, default is "sans-serif"
            Font family of the text.

        Returns
        -------
        Text
            The text layer.
        """
        layer = _l.Text(
            x,
            y,
            string,
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            fontfamily=fontfamily,
            backend=self._backend_installer,
        )
        return self.add_layer(layer)

    def add_texts(
        self,
        x: ArrayLike,
        y: ArrayLike,
        texts: list[str],
        *,
        name: str | None = None,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str = "sans-serif",
    ) -> _lg.TextGroup:
        layer = _lg.TextGroup.from_strings(
            x,
            y,
            texts,
            name=name,
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            fontfamily=fontfamily,
            backend=self._backend_installer,
        )
        return self.add_layer(layer)

    def add_image(
        self,
        image: ArrayLike,
        *,
        cmap: ColormapType = "gray",
        clim: tuple[float | None, float | None] | None = None,
        flip_canvas: bool = True,
        lock_aspect: bool = True,
    ) -> _l.Image:
        """
        Add an image layer to the canvas.

        This method automatically flips the image vertically by default.

        Parameters
        ----------
        image : ArrayLike
            Image data. Must be 2D or 3D array. If 3D, the last dimension must be
            RGB(A). Note that the first dimension is the vertical axis.
        cmap : ColormapType, default is "gray"
            Colormap used for the image.
        clim : (float or None, float or None) or None
            Contrast limits. If None, the limits are automatically determined by
            min and max of the data. You can also pass None separately to either
            limit to use the default behavior.
        flip_canvas : bool, default is True
            If True, flip the canvas vertically so that the image looks normal.

        Returns
        -------
        Image
            The image layer.
        """
        layer = _l.Image(image, cmap=cmap, clim=clim, backend=self._backend_installer)
        if flip_canvas and not self.y.flipped:
            self.y.flipped = True
        if lock_aspect:
            self.aspect_ratio = 1.0
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

    def _generate_colors(self, color: ColorType | None, *colors: ColorType | None) -> tuple[Color, ...]:
        if color is None:
            color = self._color_palette.next()
        others = [color if c is None else c for c in colors]
        return color, *others

    def _repr_png_(self):
        """Return PNG representation of the widget for QtConsole."""
        from io import BytesIO

        try:
            from imageio import imwrite
        except ImportError:
            return None

        rendered = self.screenshot()
        if rendered is not None:
            with BytesIO() as file_obj:
                imwrite(file_obj, rendered, format="png")
                file_obj.seek(0)
                return file_obj.read()
        return None


def _iter_layers(layer: _l.Layer) -> Iterator[_l.PrimitiveLayer[protocols.BaseProtocol]]:
    if isinstance(layer, _l.PrimitiveLayer):
        yield layer
    elif isinstance(layer, _l.LayerGroup):
        for child in layer._iter_children():
            yield from _iter_layers(child)
    else:
        raise TypeError(f"Unknown layer type: {type(layer).__name__}")
