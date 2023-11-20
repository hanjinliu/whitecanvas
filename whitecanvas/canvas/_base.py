from __future__ import annotations
from typing import Any, Callable, Iterator, Literal, overload, TypeVar, TYPE_CHECKING
from abc import ABC, abstractmethod

from cmap import Color

import numpy as np
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas import protocols
from whitecanvas import layers as _l
from whitecanvas.layers import group as _lg
from whitecanvas.types import (
    LineStyle,
    Symbol,
    ColorType,
    Alignment,
    ColormapType,
    FacePattern,
)
from whitecanvas.canvas import canvas_namespace as _ns, layerlist as _ll
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.utils.normalize import as_array_1d, norm_color, normalize_xy
from whitecanvas.backend import Backend, patch_dummy_backend
from whitecanvas.theme import get_theme

if TYPE_CHECKING:
    from typing_extensions import Self

_L = TypeVar("_L", bound=_l.Layer)


class CanvasBase(ABC):
    """Base class for any canvas object."""

    lims_changed = Signal(tuple)
    title = _ns.TitleNamespace()
    x = _ns.XAxisNamespace()
    y = _ns.YAxisNamespace()
    layers = _ll.LayerList()
    mouse = _ns.MouseNamespace()

    def __init__(self, palette: ColormapType | None = None):
        if palette is None:
            palette = get_theme().palette
        self._color_palette = ColorPalette(palette)
        self._is_grouping = False
        if not self._get_backend().name.startswith("."):
            self._init_canvas()

    def _init_canvas(self):
        # default colors
        theme = get_theme()
        self.x.color = theme.foreground_color
        self.y.color = theme.foreground_color
        self.x.ticks.fontfamily = theme.fontfamily
        self.y.ticks.fontfamily = theme.fontfamily
        self.x.ticks.color = theme.foreground_color
        self.y.ticks.color = theme.foreground_color
        self.x.ticks.size = theme.fontsize
        self.y.ticks.size = theme.fontsize
        self.background_color = theme.background_color

        # connect layer events
        self.layers.events.inserted.connect(self._cb_inserted, unique=True)
        self.layers.events.removed.connect(self._cb_removed, unique=True)
        self.layers.events.reordered.connect(self._cb_reordered, unique=True)

        canvas = self._canvas()
        canvas._plt_connect_xlim_changed(self._emit_xlim_changed)
        canvas._plt_connect_ylim_changed(self._emit_ylim_changed)

    def _install_mouse_events(self):
        canvas = self._canvas()
        canvas._plt_connect_mouse_click(self.mouse.clicked.emit)
        canvas._plt_connect_mouse_click(self.mouse.moved.emit)
        canvas._plt_connect_mouse_drag(self.mouse.moved.emit)
        canvas._plt_connect_mouse_double_click(self.mouse.double_clicked.emit)
        canvas._plt_connect_mouse_double_click(self.mouse.moved.emit)

    def _emit_xlim_changed(self, lim):
        self.x.lim_changed.emit(lim)
        self.lims_changed.emit((lim, self.y.lim))

    def _emit_ylim_changed(self, lim):
        self.y.lim_changed.emit(lim)
        self.lims_changed.emit((self.x.lim, lim))

    @abstractmethod
    def _get_backend(self) -> Backend:
        """Return the backend."""

    @abstractmethod
    def _canvas(self) -> protocols.CanvasProtocol:
        """Return the canvas object."""

    @property
    def aspect_ratio(self) -> float | None:
        """Aspect ratio of the canvas (None if not locked)."""
        return self._canvas()._plt_get_aspect_ratio()

    @aspect_ratio.setter
    def aspect_ratio(self, ratio: float | None):
        if ratio is not None:
            ratio = float(ratio)
        self._canvas()._plt_set_aspect_ratio(ratio)

    @property
    def visible(self):
        """Show the canvas."""
        self._canvas()._plt_set_visible(True)

    @visible.setter
    def visible(self):
        """Hide the canvas."""
        self._canvas()._plt_set_visible(False)

    @property
    def lims(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the x/y limits of the canvas."""
        return self.x.lim, self.y.lim

    @lims.setter
    def lims(self, lims: tuple[tuple[float, float], tuple[float, float]]):
        xlim, ylim = lims
        self.x.lim = xlim
        self.y.lim = ylim

    @overload
    def add_line(
        self, ydata: ArrayLike, *, name: str | None = None, color: ColorType | None = None,
        width: float = 1.0, style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    @overload
    def add_line(
        self, xdata: ArrayLike, ydata: ArrayLike, *, name: str | None = None,
        color: ColorType | None = None, width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID, antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    def add_line(
        self,
        *args,
        name=None,
        color=None,
        width=1.0,
        style=LineStyle.SOLID,
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
        color = self._generate_colors(color)
        layer = _l.Line(
            xdata, ydata, name=name, color=color, width=width, style=style,
            antialias=antialias, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_markers(
        self, ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, color: ColorType | None = None, alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Markers:  # fmt: skip
        ...

    @overload
    def add_markers(
        self, xdata: ArrayLike, ydata: ArrayLike, *,
        name: str | None = None, symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 6, color: ColorType | None = None, alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Markers:  # fmt: skip
        ...

    def add_markers(
        self,
        *args,
        name=None,
        symbol=Symbol.CIRCLE,
        size=6,
        color=None,
        alpha=1.0,
        pattern=FacePattern.SOLID,
    ):
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(_l.Markers, name)
        color = self._generate_colors(color)
        layer = _l.Markers(
            xdata, ydata, name=name, symbol=symbol, size=size, color=color,
            alpha=alpha, pattern=pattern, backend=self._get_backend(),
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
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Bars:
        name = self._coerce_name(_l.Bars, name)
        color = self._generate_colors(color)
        layer = _l.Bars(
            center, top, bottom, bar_width=bar_width, name=name, orient=orient,
            color=color, alpha=alpha, pattern=pattern, backend=self._get_backend(),
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
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Bars:
        data = as_array_1d(data)
        name = self._coerce_name("histogram", name)
        counts, edges = np.histogram(data, bins, density=density, range=range)
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]
        color = self._generate_colors(color)
        layer = _l.Bars(
            centers, counts, bar_width=width, name=name, color=color, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
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
        color = self._generate_colors(color)
        layer = _l.InfCurve(
            model, params=params, bounds=bounds, name=name, color=color,
            width=width, style=style, antialias=antialias,
            backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_band(
        self,
        xdata: ArrayLike,
        ydata0: ArrayLike,
        ydata1: ArrayLike,
        *,
        name: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Band:
        name = self._coerce_name(_l.Band, name)
        color = self._generate_colors(color)
        layer = _l.Band(
            xdata, ydata0, ydata1, name=name, orient=orient, color=color,
            alpha=alpha, pattern=pattern, backend=self._get_backend(),
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
        color = self._generate_colors(color)
        layer = _l.Errorbars(
            xdata, ylow, yhigh, name=name, color=color, width=width,
            style=style, antialias=antialias, capsize=capsize,
            orient=orient, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_kde(
        self,
        data: ArrayLike,
        band_width: float | str = "scott",
        *,
        name: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ):
        from whitecanvas.utils.kde import gaussian_kde

        data = as_array_1d(data)
        name = self._coerce_name(_l.Band, name)
        color = self._generate_colors(color)
        kde = gaussian_kde(data, bw_method=band_width)

        sigma = np.sqrt(kde.covariance[0, 0])
        pad = sigma * 4
        x = np.linspace(data.min() - pad, data.max() + pad, 100)
        y1 = kde(x)
        y0 = np.zeros_like(y1)
        layer = _l.Band(
            x, y0, y1, name=name, orient=orient, color=color, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_violinplot(
        self,
        data: dict[str, ArrayLike],
        *,
        name: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        shape: Literal["both", "left", "right"] = "both",
        violin_width: float = 0.5,
        band_width: float | str = "scott",
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ):
        name = self._coerce_name(_lg.ViolinPlot, name)
        color = self._generate_colors(color)
        group = _lg.ViolinPlot.from_dict(
            data,
            name=name,
            shape=shape,
            violin_width=violin_width,
            orient=orient,
            band_width=band_width,
            color=color,
            alpha=alpha,
            pattern=pattern,
        )
        return self.add_layer(group)

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
        fontfamily: str | None = None,
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
        fontfamily : str, optional
            Font family of the text.

        Returns
        -------
        Text
            The text layer.
        """
        layer = _l.Text(
            x, y, string, color=color, size=size, rotation=rotation, anchor=anchor,
            fontfamily=fontfamily, backend=self._get_backend(),
        )  # fmt: skip
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
        fontfamily: str | None = None,
    ) -> _lg.TextGroup:
        layer = _lg.TextGroup.from_strings(
            x, y, texts, name=name, color=color, size=size, rotation=rotation,
            anchor=anchor, fontfamily=fontfamily, backend=self._get_backend(),
        )  # fmt: skip
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
        layer = _l.Image(image, cmap=cmap, clim=clim, backend=self._get_backend())
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
            # this happens when the grouped layer is inserted
            layer._connect_canvas(self)
            return
        if idx < 0:
            idx = len(self.layers) + idx
        _canvas = self._canvas()
        for l in _iter_layers(layer):
            _canvas._plt_add_layer(l._backend)
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
                zorder += 1
            elif isinstance(layer, _l.LayerGroup):
                for child in layer.iter_children_recursive():
                    child._backend._plt_set_zorder(zorder)
                    zorder += 1
            else:
                raise RuntimeError(f"type {type(layer)} not expected")

    def _group_layers(self, group: _l.LayerGroup):
        indices: list[int] = []  # layers to remove
        not_found: list[_l.PrimitiveLayer] = []  # primitive layers to add
        id_exists = set(map(id, self.layers.iter_primitives()))
        for layer in group.iter_children():
            try:
                idx = self.layers.index(layer)
                indices.append(idx)
            except ValueError:
                not_found.extend(_iter_layers(layer))
        self._is_grouping = True
        try:
            for idx in reversed(indices):
                # remove from the layer list since it is directly grouped
                self.layers.pop(idx)
            self.layers.append(group)
            _canvas = self._canvas()
            for child in not_found:
                if id(child) in id_exists:
                    # skip since it is already in the canvas
                    continue
                child._connect_canvas(self)
                _canvas._plt_add_layer(child._backend)
        finally:
            self._is_grouping = False
        self._cb_reordered()

    def _generate_colors(self, color: ColorType | None) -> Color:
        if color is None:
            color = self._color_palette.next()
        return color


class Canvas(CanvasBase):
    def __init__(
        self,
        backend: str | None = None,
        *,
        palette: ColormapType | None = None,
    ):
        self._backend_installer = Backend(backend)
        self._backend = self._create_backend_object()
        super().__init__(palette=palette)

    @classmethod
    def from_backend(
        cls,
        obj: protocols.CanvasProtocol,
        *,
        palette: ColormapType | None = None,
        backend: str | None = None,
    ) -> Self:
        """Create a canvas object from a backend object."""
        with patch_dummy_backend() as name:
            self = cls(backend=name, palette=palette)
        self._backend_installer = Backend(backend)
        self._backend = obj
        self._init_canvas()
        return self

    def _create_backend_object(self) -> protocols.CanvasProtocol:
        return self._backend_installer.get("Canvas")()

    def _get_backend(self):
        return self._backend_installer

    def _canvas(self) -> protocols.CanvasProtocol:
        return self._backend


def _iter_layers(
    layer: _l.Layer,
) -> Iterator[_l.PrimitiveLayer[protocols.BaseProtocol]]:
    if isinstance(layer, _l.PrimitiveLayer):
        yield layer
    elif isinstance(layer, _l.LayerGroup):
        yield from layer.iter_children_recursive()
    else:
        raise TypeError(f"Unknown layer type: {type(layer).__name__}")