from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, overload, TypeVar, TYPE_CHECKING
from abc import ABC, abstractmethod

from cmap import Color

import numpy as np
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal, SignalGroup

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
    Orientation,
    ArrayLike1D,
    Rect,
    _Void,
)
from whitecanvas.canvas import (
    _namespaces as _ns,
    layerlist as _ll,
    _categorical as _cat,
)
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.canvas._imageref import ImageRef
from whitecanvas.canvas._between import BetweenPlotter
from whitecanvas.canvas._stacked import StackPlotter
from whitecanvas.utils.normalize import as_array_1d, normalize_xy
from whitecanvas.backend import Backend, patch_dummy_backend
from whitecanvas.theme import get_theme
from whitecanvas._signal import MouseSignal, GeneratorSignal

if TYPE_CHECKING:
    from typing_extensions import Self

_L = TypeVar("_L", bound=_l.Layer)
_L0 = TypeVar("_L0", _l.Bars, _l.Band)
_void = _Void()


class CanvasEvents(SignalGroup):
    lims = Signal(Rect)
    mouse_clicked = MouseSignal(object)
    mouse_moved = GeneratorSignal()
    mouse_double_clicked = MouseSignal(object)


class CanvasBase(ABC):
    """Base class for any canvas object."""

    title = _ns.TitleNamespace()
    x = _ns.XAxisNamespace()
    y = _ns.YAxisNamespace()
    layers = _ll.LayerList()
    events: CanvasEvents

    def __init__(self, palette: ColormapType | None = None):
        if palette is None:
            palette = get_theme().palette
        self._color_palette = ColorPalette(palette)
        self.events = CanvasEvents()
        self._is_grouping = False
        self._autoscale_enabled = True
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
        canvas._plt_connect_mouse_click(self.events.mouse_clicked.emit)
        canvas._plt_connect_mouse_click(self.events.mouse_moved.emit)
        canvas._plt_connect_mouse_drag(self.events.mouse_moved.emit)
        canvas._plt_connect_mouse_double_click(self.events.mouse_double_clicked.emit)
        canvas._plt_connect_mouse_double_click(self.events.mouse_moved.emit)

    def _emit_xlim_changed(self, lim):
        self.x.events.lim.emit(lim)
        self.events.lims.emit(Rect(*lim, *self.y.lim))

    def _emit_ylim_changed(self, lim):
        self.y.events.lim.emit(lim)
        self.events.lims.emit(Rect(*self.x.lim, *lim))

    @abstractmethod
    def _get_backend(self) -> Backend:
        """Return the backend."""

    @abstractmethod
    def _canvas(self) -> protocols.CanvasProtocol:
        """Return the canvas object."""

    @property
    def native(self) -> Any:
        """Return the native canvas object."""
        return self._canvas()._plt_get_native()

    @property
    def aspect_ratio(self) -> float | None:
        """Aspect ratio of the canvas (None if not locked)."""
        return self._canvas()._plt_get_aspect_ratio()

    @aspect_ratio.setter
    def aspect_ratio(self, ratio: float | None):
        if ratio is not None:
            ratio = float(ratio)
        self._canvas()._plt_set_aspect_ratio(ratio)

    def autoscale(
        self,
        xpad: float | tuple[float, float] | None = None,
        ypad: float | tuple[float, float] | None = None,
    ):
        ar = np.stack([layer.bbox_hint() for layer in self.layers], axis=0)
        xmin = np.min(ar[:, 0])
        xmax = np.max(ar[:, 1])
        ymin = np.min(ar[:, 2])
        ymax = np.max(ar[:, 3])
        x0, x1 = self.x.lim
        y0, y1 = self.y.lim
        if np.isnan(xmin):
            xmin = x0
        if np.isnan(xmax):
            xmax = x1
        if np.isnan(ymin):
            ymin = y0
        if np.isnan(ymax):
            ymax = y1
        if xpad is not None:
            xrange = xmax - xmin
            if isinstance(xpad, (int, float, np.number)):
                dx0 = dx1 = xpad * xrange
            else:
                dx0, dx1 = xpad[0] * xrange, xpad[1] * xrange
            xmin -= dx0
            xmax += dx1
        if ypad is not None:
            yrange = ymax - ymin
            if isinstance(ypad, (int, float, np.number)):
                dy0 = dy1 = ypad * yrange
            else:
                dy0, dy1 = ypad[0] * yrange, ypad[1] * yrange
            ymin -= dy0
            ymax += dy1
        if xmax - xmin < 1e-6:
            xmin -= 0.05
            xmax += 0.05
        if ymax - ymin < 1e-6:
            ymin -= 0.05
            ymax += 0.05
        self.x.lim = xmin, xmax
        self.y.lim = ymin, ymax
        return xmin, xmax, ymin, ymax

    @property
    def visible(self):
        """Show the canvas."""
        return self._canvas()._plt_get_visible()

    @visible.setter
    def visible(self, visible):
        """Hide the canvas."""
        self._canvas()._plt_set_visible(visible)

    @property
    def lims(self) -> Rect:
        """Return the x/y limits of the canvas."""
        return Rect(*self.x.lim, *self.y.lim)

    @lims.setter
    def lims(self, lims: tuple[float, float, float, float]):
        xmin, xmax, ymin, ymax = lims
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid view rect: {Rect(*lims)}")
        with self.events.lims.blocked():
            self.x.lim = xmin, xmax
            self.y.lim = ymin, ymax
        self.events.lims.emit(Rect(xmin, xmax, ymin, ymax))

    def update_axes(
        self,
        visible: bool = _void,
        color: ColorType | None = _void,
    ):
        if visible is not _void:
            self.x.ticks.visible = visible
            self.y.ticks.visible = visible
        if color is not _void:
            self.x.color = color
            self.x.ticks.color = color
            self.x.label.color = color
            self.y.color = color
            self.y.ticks.color = color
            self.y.label.color = color
        return self

    def cat(
        self,
        data: Any,
        by: str | None = None,
        *,
        orient: str | Orientation = Orientation.VERTICAL,
        offsets: float | ArrayLike1D | None = None,
        palette: ColormapType | None = None,
        update_labels: bool = True,
    ) -> _cat.CategorizedDataPlotter[Self]:
        """
        Categorize input data for plotting.

        This method provides categorical plotting methods for the input data.
        Methods are very similar to `seaborn` and `plotly.express`.

        >>> df = sns.load_dataset("iris")
        >>> canvas.cat(df, by="species").to_violinplot(y="sepal_width)
        >>> canvas.cat(df, by="species").mean().to_line(y="sepal_width)

        Parameters
        ----------
        data : tabular data
            Any categorizable data. Currently, dict, pandas.DataFrame, and
            polars.DataFrame are supported.
        by : str, optional
            Which column to use for grouping.
        orient : str or Orientation, default is Orientation.VERTICAL
            Orientation of the plot.
        offsets : scalar or sequence, optional
            Offset for each category. If scalar, the same offset is used for all.
        palette : ColormapType, optional
            Color palette used for plotting the categories.
        update_labels : bool, default is True
            If True, update the x/y labels to the corresponding names.

        Returns
        -------
        CategorizedDataPlotter
            Plotter object.
        """
        orient = Orientation.parse(orient)
        plotter = _cat.CategorizedDataPlotter(
            self, data, by=by, orient=orient, offsets=offsets,
            update_label=update_labels, palette=palette
        )  # fmt: skip
        if update_labels:
            if orient.is_vertical:
                self.x.label.text = by
            else:
                self.y.label.text = by
        return plotter

    def colorize(
        self,
        data: Any,
        by: str | None = None,
        *,
        update_labels: bool = True,
        palette: ColormapType | None = None,
    ) -> _cat.ColorizedPlotter[Self]:
        if palette is None:
            palette = self._color_palette
        plotter = _cat.ColorizedPlotter(
            self, data, by, palette=palette, update_label=update_labels
        )
        return plotter

    def stack_over(self, layer: _L0) -> StackPlotter[Self, _L0]:
        """
        Stack new data over the existing layer.

        For example following code

        >>> bars_0 = canvas.add_bars(x, y0)
        >>> bars_1 = canvas.stack_over(bars_0).add(y1)
        >>> bars_2 = canvas.stack_over(bars_1).add(y2)

        will result in a bar plot like this

         ┌───┐
         ├───│┌───┐
         │   │├───│
         ├───│├───│
        ─┴───┴┴───┴─
        """
        if not isinstance(layer, (_l.Bars, _l.Band, _lg.StemPlot, _lg.LabeledBars)):
            raise TypeError(
                f"Only Bars and Band are supported as an input, "
                f"got {type(layer)!r}."
            )
        return StackPlotter(self, layer)

    # TODO
    # def annotate(self, layer, at: int):
    #     ...

    def refer_image(self, layer: _l.Image) -> ImageRef[Self]:
        return ImageRef(self, layer)

    def between(self, l0, l1) -> BetweenPlotter[Self]:
        return BetweenPlotter(self, l0, l1)

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

    @overload
    def add_line(
        self, xdata: ArrayLike, ydata: Callable[[ArrayLike], ArrayLike], *,
        name: str | None = None, color: ColorType | None = None, width: float = 1.0,
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

    @overload
    def add_bars(
        self, center: ArrayLike, height: ArrayLike, bottom: ArrayLike | None = None,
        *, name=None, orient: str | Orientation = Orientation.VERTICAL,
        bar_width: float = 0.8, color: ColorType | None = None,
        alpha: float = 1.0, pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Bars:  # fmt: skip
        ...

    @overload
    def add_bars(
        self, height: ArrayLike,
        *, name=None, orient: str | Orientation = Orientation.VERTICAL,
        bar_width: float = 0.8, color: ColorType | None = None,
        alpha: float = 1.0, pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Bars:  # fmt: skip
        ...

    def add_bars(
        self,
        *args,
        bottom=None,
        name=None,
        orient=Orientation.VERTICAL,
        bar_width=0.8,
        color=None,
        alpha=1.0,
        pattern=FacePattern.SOLID,
    ) -> _l.Bars:
        center, height = normalize_xy(*args)
        if bottom is not None:
            bottom = as_array_1d(bottom)
            if bottom.shape != height.shape:
                raise ValueError("Expected bottom to have the same shape as height")
        name = self._coerce_name(_l.Bars, name)
        color = self._generate_colors(color)
        layer = _l.Bars(
            center, height, bottom, bar_width=bar_width, name=name, orient=orient,
            color=color, alpha=alpha, pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_hist(
        self,
        data: ArrayLike,
        *,
        bins: int | ArrayLike = 10,
        range: tuple[float, float] | None = None,
        density: bool = False,
        name: str | None = None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Bars:
        name = self._coerce_name("histogram", name)
        color = self._generate_colors(color)
        layer = _l.Bars.from_histogram(
            data, bins=bins, range=range, density=density, name=name, color=color,
            alpha=alpha, pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_spans(
        self,
        spans: ArrayLike,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType = "blue",
        alpha: float = 0.2,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Spans:
        name = self._coerce_name("histogram", name)
        color = self._generate_colors(color)
        layer = _l.Spans(
            spans, name=name, orient=orient, color=color, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_infline(
        self,
        pos: tuple[float, float] = (0, 0),
        angle: float = 0.0,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
    ):
        name = self._coerce_name(_l.InfLine, name)
        color = self._generate_colors(color)
        layer = _l.InfLine(
            pos, angle, name=name, color=color,
            width=width, style=style, antialias=antialias,
            backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_infcurve(
        self,
        model: Callable[..., NDArray[np.floating]],
        params: dict[str, Any] = {},
        *,
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
        orient: str | Orientation = Orientation.VERTICAL,
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
        orient: str | Orientation = Orientation.VERTICAL,
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

    def add_rug(
        self,
        events: ArrayLike,
        *,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        color: ColorType = "black",
        alpha: float = 1.0,
        orient: str | Orientation = Orientation.VERTICAL,
    ) -> _l.Rug:
        name = self._coerce_name(_l.Errorbars, name)
        color = self._generate_colors(color)
        layer = _l.Rug(
            events, low=low, high=high, name=name, color=color,
            alpha=alpha, orient=orient, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_kde(
        self,
        data: ArrayLike,
        *,
        bottom: float = 0.0,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        band_width: float | str = "scott",
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
        y0 = np.full_like(y1, bottom)
        layer = _l.Band(
            x, y0, y1, name=name, orient=orient, color=color, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
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
        self.add_layer(layer)
        if flip_canvas and not self.y.flipped:
            self.y.flipped = True
        if lock_aspect:
            self.aspect_ratio = 1.0
        return layer

    def add_layer(
        self,
        layer: _L,
        *,
        over: _l.Layer | Iterable[_l.Layer] | None = None,
        under: _l.Layer | Iterable[_l.Layer] | None = None,
    ) -> _L:
        """Add a layer to the canvas."""
        if over is None and under is None:
            self.layers.append(layer)
        elif over is not None:
            if under is not None:
                raise ValueError("Cannot specify both `over` and `under`")
            if isinstance(over, _l.Layer):
                idx = self.layers.index(over)
            else:
                idx = max([self.layers.index(l) for l in over])
            self.layers.insert(idx + 1, layer)
        else:
            idx = self.layers.index(under)
            if isinstance(under, _l.Layer):
                idx = self.layers.index(under)
            else:
                idx = min([self.layers.index(l) for l in under])
            self.layers.insert(idx, layer)
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

    def _autoscale_for_layer(self, layer: _l.Layer, pad_rel: float = 0.025):
        if not self._autoscale_enabled:
            return
        xmin, xmax, ymin, ymax = layer.bbox_hint()
        if len(self.layers) > 1:
            # NOTE: if there was no layer, so backend may not have xlim/ylim,
            # or they may be set to a default value.
            _xmin, _xmax = self.x.lim
            _ymin, _ymax = self.y.lim
            _dx = (_xmax - _xmin) * pad_rel
            _dy = (_ymax - _ymin) * pad_rel
            xmin = np.min([xmin, _xmin + _dx])
            xmax = np.max([xmax, _xmax - _dx])
            ymin = np.min([ymin, _ymin + _dy])
            ymax = np.max([ymax, _ymax - _dy])

        # this happens when there is <= 1 data
        if np.isnan(xmax) or np.isnan(xmin):
            xmin, xmax = self.x.lim
        elif xmax - xmin < 1e-6:
            xmin -= 0.05
            xmax += 0.05
        else:
            dx = (xmax - xmin) * pad_rel
            xmin -= dx
            xmax += dx
        if np.isnan(ymax) or np.isnan(ymin):
            ymin, ymax = self.y.lim
        elif ymax - ymin < 1e-6:
            ymin -= 0.05
            ymax += 0.05
        else:
            dy = (ymax - ymin) * pad_rel
            ymin -= dy  # TODO: this causes bars/histogram to float
            ymax += dy  #       over the x-axis.
        self.lims = xmin, xmax, ymin, ymax

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
        # autoscale
        if isinstance(layer, _l.Image):
            pad_rel = 0
        else:
            pad_rel = 0.025
        self._autoscale_for_layer(layer, pad_rel=pad_rel)

    def _cb_removed(self, idx: int, layer: _l.Layer):
        if self._is_grouping:
            return
        _canvas = self._canvas()
        for l in _iter_layers(layer):
            _canvas._plt_remove_layer(l._backend)
            l._disconnect_canvas(self)

    def _cb_reordered(self):
        layer_backends = []
        for layer in self.layers:
            if isinstance(layer, _l.PrimitiveLayer):
                layer_backends.append(layer._backend)
            elif isinstance(layer, _l.LayerGroup):
                for child in layer.iter_children_recursive():
                    layer_backends.append(child._backend)
            else:
                raise RuntimeError(f"type {type(layer)} not expected")
        self._canvas()._plt_reorder_layers(layer_backends)

    def _cb_layer_grouped(self, group: _l.LayerGroup):
        indices: list[int] = []  # layers to remove
        not_found: list[_l.PrimitiveLayer] = []  # primitive layers to add
        id_exists = set(map(id, self.layers.iter_primitives()))
        for layer in group.iter_children():
            try:
                idx = self.layers.index(layer)
                indices.append(idx)
            except ValueError:
                not_found.extend(_iter_layers(layer))
        if not indices:
            return
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
        self._autoscale_for_layer(group)

    def _generate_colors(self, color: ColorType | None) -> Color:
        if color is None:
            color = self._color_palette.next()
        return color


class Canvas(CanvasBase):
    _CURRENT_INSTANCE: Canvas | None = None

    def __init__(
        self,
        backend: str | None = None,
        *,
        palette: ColormapType | None = None,
    ):
        self._backend = Backend(backend)
        self._backend_object = self._create_backend_object()
        super().__init__(palette=palette)
        self.__class__._CURRENT_INSTANCE = self

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
        self._backend = Backend(backend)
        self._backend_object = obj
        self._init_canvas()
        return self

    def _create_backend_object(self) -> protocols.CanvasProtocol:
        return self._backend.get("Canvas")()

    def _get_backend(self):
        return self._backend

    def _canvas(self) -> protocols.CanvasProtocol:
        return self._backend_object


def _iter_layers(
    layer: _l.Layer,
) -> Iterator[_l.PrimitiveLayer[protocols.BaseProtocol]]:
    if isinstance(layer, _l.PrimitiveLayer):
        yield layer
    elif isinstance(layer, _l.LayerGroup):
        yield from layer.iter_children_recursive()
    else:
        raise TypeError(f"Unknown layer type: {type(layer).__name__}")
