from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
from cmap import Color
from numpy.typing import ArrayLike
from psygnal import Signal, SignalGroup

from whitecanvas import layers as _l
from whitecanvas import protocols, theme
from whitecanvas._signal import MouseMoveSignal, MouseSignal
from whitecanvas.backend import Backend, patch_dummy_backend
from whitecanvas.canvas import (
    _namespaces as _ns,
)
from whitecanvas.canvas import (
    dataframe as _df,
)
from whitecanvas.canvas import (
    layerlist as _ll,
)
from whitecanvas.canvas._between import BetweenPlotter
from whitecanvas.canvas._dims import Dims
from whitecanvas.canvas._fit import FitPlotter
from whitecanvas.canvas._imageref import ImageRef
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.canvas._stacked import StackOverPlotter
from whitecanvas.layers import _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.types import (
    Alignment,
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    Rect,
    Symbol,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xy

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec, Self

    _P = ParamSpec("_P")
    _DF = TypeVar("_DF")

_L = TypeVar("_L", bound=_l.Layer)
_L0 = TypeVar("_L0", _l.Bars, _l.Band)
_void = _Void()


class CanvasEvents(SignalGroup):
    lims = Signal(Rect)
    drawn = Signal()
    mouse_clicked = MouseSignal(object)
    mouse_moved = MouseMoveSignal()
    mouse_double_clicked = MouseSignal(object)


class CanvasBase(ABC):
    """Base class for any canvas object."""

    title = _ns.TitleNamespace()
    x = _ns.XAxisNamespace()
    y = _ns.YAxisNamespace()
    dims = Dims()
    layers = _ll.LayerList()
    overlays = _ll.LayerList()
    events: CanvasEvents

    def __init__(self, palette: ColormapType | None = None):
        if palette is None:
            palette = theme.get_theme().palette
        self._color_palette = ColorPalette(palette)
        self.events = CanvasEvents()
        self._is_grouping = False
        self._autoscale_enabled = True
        if not self._get_backend().is_dummy():
            self._init_canvas()

    def _init_canvas(self):
        # default colors and font
        _t = theme.get_theme()
        _ft = _t.font
        self.x.color = _t.foreground_color
        self.y.color = _t.foreground_color
        self.x.label.update(family=_ft.family, color=_ft.color, size=_ft.size)
        self.y.label.update(family=_ft.family, color=_ft.color, size=_ft.size)
        self.title.update(family=_ft.family, color=_ft.color, size=_ft.size)
        self.x.ticks.update(family=_ft.family, color=_ft.color, size=_ft.size)
        self.y.ticks.update(family=_ft.family, color=_ft.color, size=_ft.size)

        # connect layer events
        self.layers.events.inserted.connect(self._cb_inserted, unique=True)
        self.layers.events.removed.connect(self._cb_removed, unique=True)
        self.layers.events.reordered.connect(self._cb_reordered, unique=True)
        self.layers.events.connect(self._draw_canvas, unique=True)

        self.overlays.events.inserted.connect(self._cb_inserted_overlay, unique=True)
        self.overlays.events.removed.connect(self._cb_removed, unique=True)
        self.overlays.events.connect(self._draw_canvas, unique=True)

        canvas = self._canvas()
        canvas._plt_connect_xlim_changed(self._emit_xlim_changed)
        canvas._plt_connect_ylim_changed(self._emit_ylim_changed)

    def _install_mouse_events(self):
        canvas = self._canvas()
        canvas._plt_connect_mouse_click(self.events.mouse_clicked.emit)
        canvas._plt_connect_mouse_click(self.events.mouse_moved.emit)
        canvas._plt_connect_mouse_drag(self.events.mouse_moved.emit)
        canvas._plt_connect_mouse_release(self.events.mouse_moved.emit)
        canvas._plt_connect_mouse_double_click(self.events.mouse_double_clicked.emit)
        canvas._plt_connect_mouse_double_click(self.events.mouse_moved.emit)

    def _emit_xlim_changed(self, lim):
        self.x.events.lim.emit(lim)
        self.events.lims.emit(Rect(*lim, *self.y.lim))

    def _emit_ylim_changed(self, lim):
        self.y.events.lim.emit(lim)
        self.events.lims.emit(Rect(*self.x.lim, *lim))

    def _emit_mouse_moved(self, ev):
        """Emit mouse moved event with autoscaling blocked"""
        _was_enabled = self._autoscale_enabled
        # If new layers are added during the mouse move event, the canvas
        # should not be autoscaled, otherwise unexpected values will be
        # passed to the callback functions.
        self._autoscale_enabled = False
        try:
            self.events.mouse_moved.emit(ev)
        finally:
            self._autoscale_enabled = _was_enabled

    @abstractmethod
    def _get_backend(self) -> Backend:
        """Return the backend."""

    @abstractmethod
    def _canvas(self) -> protocols.CanvasProtocol:
        """Return the canvas object."""

    def _draw_canvas(self):
        self._canvas()._plt_draw()
        self.events.drawn.emit()

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
    ) -> tuple[float, float, float, float]:
        """
        Autoscale the canvas to fit the contents.

        Parameters
        ----------
        xpad : float or (float, float), optional
            Padding in the x direction.
        ypad : float or (float, float), optional
            Padding in the y direction.
        """
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
        small_diff = 1e-6
        if xmax - xmin < small_diff:
            xmin -= 0.05
            xmax += 0.05
        if ymax - ymin < small_diff:
            ymin -= 0.05
            ymax += 0.05
        self.x.lim = xmin, xmax
        self.y.lim = ymin, ymax
        return xmin, xmax, ymin, ymax

    def install_second_y(
        self,
        *,
        palette: ColormapType | None = None,
    ) -> Canvas:
        """Create a twin canvas that share one of the axis."""
        try:
            new = self._canvas()._plt_twinx()
        except AttributeError:
            raise NotImplementedError(
                f"Backend {self._get_backend()} does not support `install_second_y`."
            )
        canvas = Canvas.from_backend(new, palette=palette, backend=self._get_backend())
        canvas._init_canvas()
        return canvas

    def install_inset(
        self,
        rect: Rect | tuple[float, float, float, float],
        *,
        palette: ColormapType | None = None,
    ) -> Canvas:
        if not isinstance(rect, Rect):
            rect = Rect(*rect)
        try:
            new = self._canvas()._plt_inset(rect)
        except AttributeError:
            raise NotImplementedError(
                f"Backend {self._get_backend()} does not support `install_inset`"
            )
        canvas = Canvas.from_backend(new, palette=palette, backend=self._get_backend())
        canvas._init_canvas()
        return canvas

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

    def update_labels(
        self,
        title: str | None = None,
        x: str | None = None,
        y: str | None = None,
    ) -> Self:
        """
        Helper function to update the title, x, and y labels.

        >>> from whitecanvas import new_canvas
        >>> canvas = new_canvas("matplotlib").update_labels("Title", "X", "Y")
        """
        if title is not None:
            self.title.text = title
        if x is not None:
            self.x.label.text = x
        if y is not None:
            self.y.label.text = y
        return self

    def cat(
        self,
        data: _DF,
        x: str | None = None,
        y: str | None = None,
        *,
        update_labels: bool = True,
    ) -> _df.CatPlotter[Self, _DF]:
        """
        Categorize input data for plotting.

        This method provides categorical plotting methods for the input data.
        Methods are very similar to `seaborn` and `plotly.express`.

        Parameters
        ----------
        data : tabular data
            Any categorizable data. Currently, dict, pandas.DataFrame, and
            polars.DataFrame are supported.

        update_labels : bool, default True
            If True, update the x/y labels to the corresponding names.

        Returns
        -------
        CategorizedPlot
            Plotter object.
        """
        plotter = _df.CatPlotter(self, data, x, y, update_label=update_labels)
        return plotter

    def cat_x(
        self,
        data: _DF,
        x: str | Sequence[str] | None = None,
        y: str | None = None,
        *,
        update_labels: bool = True,
    ) -> _df.XCatPlotter[Self, _DF]:
        return _df.XCatPlotter(self, data, x, y, update_labels)

    def cat_y(
        self,
        data: _DF,
        x: str | None = None,
        y: str | Sequence[str] | None = None,
        *,
        update_labels: bool = True,
    ) -> _df.YCatPlotter[Self, _DF]:
        return _df.YCatPlotter(self, data, y, x, update_labels)

    def cat_xy(
        self,
        data: _DF,
        x: str | Sequence[str],
        y: str | Sequence[str],
        *,
        update_labels: bool = True,
    ) -> _df.XYCatPlotter[Self, _DF]:
        return _df.XYCatPlotter(self, data, x, y, update_labels)

    def stack_over(self, layer: _L0) -> StackOverPlotter[Self, _L0]:
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
        return StackOverPlotter(self, layer)

    # TODO
    # def annotate(self, layer, at: int):
    #     ...

    def between(self, l0, l1) -> BetweenPlotter[Self]:
        return BetweenPlotter(self, l0, l1)

    def imref(self, layer: _l.Image) -> ImageRef[Self]:
        """The Image reference namespace."""
        while isinstance(layer, _l.LayerWrapper):
            layer = layer._base_layer
        if not isinstance(layer, _l.Image):
            raise TypeError(
                f"Expected an Image layer or its wrapper, got {type(layer)}."
            )
        return ImageRef(self, layer)

    def fit(self, layer: _l.DataBoundLayer[_P]) -> FitPlotter[Self, _P]:
        """The fit plotter namespace."""
        return FitPlotter(self, layer)

    @overload
    def add_line(
        self, ydata: ArrayLike1D, *, name: str | None = None,
        color: ColorType | None = None, width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID, alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    @overload
    def add_line(
        self, xdata: ArrayLike1D, ydata: ArrayLike1D, *, name: str | None = None,
        color: ColorType | None = None, width: float | None = None,
        style: LineStyle | str | None = None, alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    @overload
    def add_line(
        self, xdata: ArrayLike1D, ydata: Callable[[ArrayLike1D], ArrayLike1D], *,
        name: str | None = None, color: ColorType | None = None,
        width: float | None = None, style: LineStyle | str | None = None,
        alpha: float = 1.0, antialias: bool = True,
    ) -> _l.Line:  # fmt: skip
        ...

    def add_line(
        self,
        *args,
        name=None,
        color=None,
        width=None,
        style=None,
        alpha=1.0,
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
        color : color-like, optional
            Color of the bars.
        width : float, optional
            Line width. Use the theme default if not specified.
        style : str or LineStyle, optional
            Line style. Use the theme default if not specified.
        alpha : float, default 1.0
            Alpha channel of the line.
        antialias : bool, default True
            Antialiasing of the line.

        Returns
        -------
        Line
            The line layer.
        """
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(_l.Line, name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.Line(
            xdata, ydata, name=name, color=color, width=width, style=style,
            alpha=alpha, antialias=antialias, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_markers(
        self, xdata: ArrayLike1D, ydata: ArrayLike1D, *,
        name: str | None = None, symbol: Symbol | str | None = None,
        size: float | None = None, color: ColorType | None = None, alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> _l.Markers[_mixin.ConstFace, _mixin.ConstEdge, float]:  # fmt: skip
        ...

    @overload
    def add_markers(
        self, ydata: ArrayLike1D, *,
        name: str | None = None, symbol: Symbol | str | None = None,
        size: float | None = None, color: ColorType | None = None, alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> _l.Markers[_mixin.ConstFace, _mixin.ConstEdge, float]:  # fmt: skip
        ...

    def add_markers(
        self,
        *args,
        name=None,
        symbol=None,
        size=None,
        color=None,
        alpha=1.0,
        hatch=None,
    ):
        """
        Add markers (scatter plot).

        >>> canvas.add_markers(x, y)  # standard usage
        >>> canvas.add_markers(y)  # use 0, 1, ... for the x values

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        symbol : str or Symbol, optional
            Marker symbols. Use the theme default if not specified.
        size : float, optional
            Marker size. Use the theme default if not specified.
        color : color-like, optional
            Color of the marker faces.
        alpha : float, default 1.0
            Alpha channel of the marker faces.
        hatch : str or FacePattern, optional
            Pattern of the marker faces. Use the theme default if not specified.

        Returns
        -------
        Markers
            The markers layer.
        """
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(_l.Markers, name)
        color = self._generate_colors(color)
        symbol = theme._default("markers.symbol", symbol)
        size = theme._default("markers.size", size)
        hatch = theme._default("markers.hatch", hatch)
        layer = _l.Markers(
            xdata, ydata, name=name, symbol=symbol, size=size, color=color,
            alpha=alpha, hatch=hatch, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_bars(
        self, center: ArrayLike1D, height: ArrayLike1D, *,
        bottom: ArrayLike1D | None = None, name=None,
        orient: str | Orientation = Orientation.VERTICAL, extent: float | None = None,
        color: ColorType | None = None, alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> _l.Bars[_mixin.ConstFace, _mixin.ConstEdge]:  # fmt: skip
        ...

    @overload
    def add_bars(
        self, height: ArrayLike1D, *, bottom: ArrayLike1D | None = None,
        name=None, orient: str | Orientation = Orientation.VERTICAL,
        extent: float | None = None, color: ColorType | None = None,
        alpha: float = 1.0, hatch: str | Hatch | None = None,
    ) -> _l.Bars[_mixin.ConstFace, _mixin.ConstEdge]:  # fmt: skip
        ...

    def add_bars(
        self,
        *args,
        bottom=None,
        name=None,
        orient=Orientation.VERTICAL,
        extent=None,
        color=None,
        alpha=1.0,
        hatch=None,
    ):
        """
        Add a bar plot.

        >>> canvas.add_bars(x, heights)  # standard usage
        >>> canvas.add_bars(heights)  # use 0, 1, ... for the x values
        >>> canvas.add_bars(..., orient="horizontal")  # horizontal bars

        Parameters
        ----------
        bottom : float or array-like, optional
            Bottom level of the bars.
        name : str, optional
            Name of the layer.
        orient : str or Orientation, default Orientation.VERTICAL
            Orientation of the bars.
        extent : float, default 0.8
            Bar width in the canvas coordinate
        color : color-like, optional
            Color of the bars.
        alpha : float, default 1.0
            Alpha channel of the bars.
        hatch : str or FacePattern, default FacePattern.SOLID
            Pattern of the bar faces.

        Returns
        -------
        Bars
            The bars layer.
        """
        center, height = normalize_xy(*args)
        if bottom is not None:
            bottom = as_array_1d(bottom)
            if bottom.shape != height.shape:
                raise ValueError("Expected bottom to have the same shape as height")
        name = self._coerce_name(_l.Bars, name)
        color = self._generate_colors(color)
        extent = theme._default("bars.extent", extent)
        hatch = theme._default("bars.hatch", hatch)
        layer = _l.Bars(
            center, height, bottom, extent=extent, name=name, orient=orient,
            color=color, alpha=alpha, hatch=hatch, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_hist(
        self,
        data: ArrayLike1D,
        *,
        bins: int | ArrayLike1D = 10,
        limits: tuple[float, float] | None = None,
        name: str | None = None,
        shape: Literal["step", "polygon", "bars"] = "bars",
        kind: Literal["count", "density", "frequency", "percent"] = "count",
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
    ) -> _lg.Histogram:
        """
        Add data as a histogram.

        >>> canvas.add_hist(np.random.normal(size=100), bins=12)

        Parameters
        ----------
        data : array-like
            1D Array of data.
        bins : int or 1D array-like, default 10
            Bins of the histogram. This parameter will directly be passed
            to `np.histogram`.
        limits : (float, float), optional
            Limits in which histogram will be built. This parameter will equivalent to
            the `range` paraneter of `np.histogram`.
        density : bool, default False
            If True, heights of bars will be normalized so that the total
            area of the histogram will be 1. This parameter will directly
            be passed to `np.histogram`.
        name : str, optional
            Name of the layer.
        orient : str or Orientation, default Orientation.VERTICAL
            Orientation of the bars.
        color : color-like, optional
            Color of the bars.

        Returns
        -------
        Bars
            The bars layer that represents the histogram.
        """
        name = self._coerce_name("histogram", name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _lg.Histogram.from_array(
            data, bins=bins, limits=limits, shape=shape, kind=kind, name=name,
            color=color, width=width, style=style, orient=orient,
            backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_hist2d(
        self,
        x: ArrayLike1D,
        y: ArrayLike1D,
        *,
        cmap: ColormapType = "inferno",
        name: str | None = None,
        bins: int | tuple[int, int] = 10,
        rangex: tuple[float, float] | None = None,
        rangey: tuple[float, float] | None = None,
        density: bool = False,
    ) -> _l.Image:
        """
        Add a 2D histogram of given X/Y data.

        >>> x = np.random.normal(size=100)
        >>> y = np.random.normal(size=200)
        >>> canvas.add_hist2d(x, y)

        Note that unlike `add_image()` method, this method does not lock the aspect
        ratio and flip the canvas by default.

        Parameters
        ----------
        x : array-like
            1D Array of X data.
        y : array-like
            1D Array of Y data.
        cmap : ColormapType, default "gray"
            Colormap used for the image.
        name : str, optional
            Name of the layer.
        bins : int or tuple[int, int], optional
            Bins of the histogram of X/Y dimension respectively. If an integer is given,
            it will be used for both dimensions.
        rangex : (float, float), optional
            Range of x values in which histogram will be built.
        rangey : (float, float), optional
            Range of y values in which histogram will be built.
        density : bool, default False
            If True, values of the histogram will be normalized so that the total
            intensity of the histogram will be 1.

        Returns
        -------
        Image
            Image layer representing the 2D histogram.
        """
        layer = _l.Image.build_hist(
            x, y, bins=bins, range=(rangex, rangey), density=density, name=name,
            cmap=cmap, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_cdf(
        self,
        data: ArrayLike1D,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.Line:
        """
        Add a empirical cumulative distribution function (CDF) plot.

        >>> canvas.add_cdf(np.random.normal(size=100))

        Parameters
        ----------
        data : array-like
            1D Array of data.
        name : str, optional
            Name of the layer.
        orient : str or Orientation, default Orientation.VERTICAL
            Orientation of the bars.
        color : color-like, optional
            Color of the bars.
        width : float, optional
            Line width. Use the theme default if not specified.
        style : str or LineStyle, optional
            Line style. Use the theme default if not specified.
        alpha : float, default 1.0
            Alpha channel of the line.
        antialias : bool, default True
            Antialiasing of the line.

        Returns
        -------
        Line
            The line layer that represents the CDF.
        """
        name = self._coerce_name("histogram", name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.Line.build_cdf(
            data, orient=orient, name=name, color=color, width=width, style=style,
            alpha=alpha, antialias=antialias, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_spans(
        self,
        spans: ArrayLike,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType = "blue",
        alpha: float = 0.4,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _l.Spans:
        """
        Add spans that extends infinitely.

        >>> canvas.add_spans([[5, 10], [15, 20]])

           |::::|     |::::|
           |::::|     |::::|
        ───5────10────15───20─────>
           |::::|     |::::|
           |::::|     |::::|

        Parameters
        ----------
        spans : (N, 2) array-like
            Array that contains the start and end points of the spans.
        name : str, optional
            Name of the layer.
        orient : str or Orientation, default Orientation.VERTICAL
            Orientation of the bars.
        color : color-like, optional
            Color of the bars.
        alpha : float, default 0.4
            Alpha channel of the bars.
        hatch : str or FacePattern, default FacePattern.SOLID
            Pattern of the bar faces.

        Returns
        -------
        Spans
            The spans layer.
        """
        name = self._coerce_name("histogram", name)
        color = self._generate_colors(color)
        layer = _l.Spans(
            spans, name=name, orient=orient, color=color, alpha=alpha,
            hatch=hatch, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_infline(
        self,
        pos: tuple[float, float] = (0, 0),
        angle: float = 0.0,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.InfLine:
        """
        Add an infinitely long line to the canvas.

        >>> canvas.add_infline((0, 0), 45)  # y = x
        >>> canvas.add_infline((1, 0), 90)  # x = 1
        >>> canvas.add_infline((0, -1), 0)  # y = -1

        Parameters
        ----------
        pos : (float, float), default (0, 0)
            One of the points this line passes.
        angle : float, default 0.0
            Angle of the line in degree, defined by the counter-clockwise
            rotation from the x axis.
        name : str, optional
            Name of the layer.
        color : color-like, optional
            Color of the bars.
        width : float, optional
            Line width. Use the theme default if not specified.
        style : str or LineStyle, optional
            Line style. Use the theme default if not specified.
        alpha : float, default 1.0
            Alpha channel of the line.
        antialias : bool, default True
            Antialiasing of the line.

        Returns
        -------
        InfLine
            The infline layer.
        """
        name = self._coerce_name(_l.InfLine, name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.InfLine(
            pos, angle, name=name, color=color, alpha=alpha,
            width=width, style=style, antialias=antialias,
            backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_infcurve(
        self,
        model: Callable[Concatenate[Any, _P], Any],
        *,
        bounds: tuple[float, float] = (-float("inf"), float("inf")),
        name: str | None = None,
        color: ColorType | None = None,
        width: float | None = None,
        style: str | LineStyle | None = None,
        antialias: bool = True,
    ) -> _l.InfCurve[_P]:
        """
        Add an infinite curve to the canvas.

        >>> canvas.add_infcurve(lambda x: x ** 2)  # parabola
        >>> canvas.add_infcurve(lambda x, a: np.sin(a*x)).with_params(2)  # parametric

        Parameters
        ----------
        model : callable
            The model function. The first argument must be the x coordinates. Same
            signature as `scipy.optimize.curve_fit`.
        bounds : (float, float), default (-inf, inf)
            Lower and upper bounds that the function is defined.
        name : str, optional
            Name of the layer.
        color : color-like, optional
            Color of the bars.
        width : float, optional
            Line width. Use the theme default if not specified.
        style : str or LineStyle, optional
            Line style. Use the theme default if not specified.
        alpha : float, default 1.0
            Alpha channel of the line.
        antialias : bool, default True
            Antialiasing of the line.

        Returns
        -------
        InfCurve
            The infcurve layer.
        """
        name = self._coerce_name(_l.InfCurve, name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.InfCurve(
            model, bounds=bounds, name=name, color=color, width=width,
            style=style, antialias=antialias, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_hline(
        self,
        y: float,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.InfLine:
        return self.add_infline(
            (0, y), 0, name=name, color=color, width=width, style=style, alpha=alpha,
            antialias=antialias
        )  # fmt: skip

    def add_vline(
        self,
        x: float,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.InfLine:
        return self.add_infline(
            (x, 0), 90, name=name, color=color, width=width, style=style, alpha=alpha,
            antialias=antialias,
        )  # fmt: skip

    def add_band(
        self,
        xdata: ArrayLike1D,
        ylow: ArrayLike1D,
        yhigh: ArrayLike1D,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType | None = None,
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _l.Band:
        """
        Add a band (fill-between) layer to the canvas.

        Parameters
        ----------
        xdata : array-like
            X coordinates of the band.
        ylow : array-like
            Either lower or upper y coordinates of the band.
        yhigh : array-like
            The other y coordinates of the band.
        name : str, optional
            Name of the layer, by default None
        orient : str, Orientation, default Orientation.VERTICAL
            Orientation of the band. If vertical, band will be filled between
            vertical orientation.,
        color : color-like, default None
            Color of the band face.,
        alpha : float, default 1.0
            Alpha channel of the band face.
        hatch : str, FacePattern, default FacePattern.SOLID
            Hatch of the band face.

        Returns
        -------
        Band
            The band layer.
        """
        name = self._coerce_name(_l.Band, name)
        color = self._generate_colors(color)
        layer = _l.Band(
            xdata, ylow, yhigh, name=name, orient=orient, color=color,
            alpha=alpha, hatch=hatch, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_errorbars(
        self,
        xdata: ArrayLike1D,
        ylow: ArrayLike1D,
        yhigh: ArrayLike1D,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
        antialias: bool = False,
        capsize: float = 0.0,
    ) -> _l.Errorbars:
        """
        Add parallel lines as errorbars.

        Parameters
        ----------
        xdata : array-like
            X coordinates of the errorbars.
        ylow : array-like
            Lower bound of the errorbars.
        yhigh : array-like
            Upper bound of the errorbars.
        name : str, optional
            Name of the layer.
        orient : str or Orientation, default Orientation.VERTICAL
            Orientation of the errorbars. If vertical, errorbars will be parallel
            to the y axis.
        color : color-like, optional
            Color of the bars.
        width : float, optional
            Line width. Use the theme default if not specified.
        style : str or LineStyle, optional
            Line style. Use the theme default if not specified.
        alpha : float, default 1.0
            Alpha channel of the line.
        antialias : bool, default True
            Antialiasing of the line.
        capsize : float, default 0.0
            Size of the caps of the error indicators

        Returns
        -------
        Errorbars
            The errorbars layer.
        """
        name = self._coerce_name(_l.Errorbars, name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.Errorbars(
            xdata, ylow, yhigh, name=name, color=color, width=width,
            style=style, antialias=antialias, capsize=capsize,
            orient=orient, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_rug(
        self,
        events: ArrayLike1D,
        *,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType = "black",
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        alpha: float = 1.0,
    ) -> _l.Rug:
        """
        Add input data as a rug plot.

        >>> canvas.add_rug([2, 4, 5, 8, 11])

          │ ││  │   │
        ──┴─┴┴──┴───┴──> x
          2 45  8   11

        Parameters
        ----------
        events : array-like
            A 1D array of events.
        low : float, default 0.0
            The lower bound of the rug lines.
        high : float, default 1.0
            The upper bound of the rug lines.
        name : str, optional
            Name of the layer.
        orient : str or Orientation, default Orientation.VERTICAL
            Orientation of the errorbars. If vertical, rug lines will be parallel
            to the y axis.
        color : color-like, optional
            Color of the bars.
        width : float, default 1.0
            Line width.
        style : str or LineStyle, default LineStyle.SOLID
            Line style.
        alpha : float, default 1.0
            Alpha channel of the line.
        antialias : bool, default True
            Antialiasing of the line.

        Returns
        -------
        Rug
            The rug layer.
        """
        name = self._coerce_name(_l.Errorbars, name)
        color = self._generate_colors(color)
        layer = _l.Rug(
            events, low=low, high=high, name=name, color=color, alpha=alpha,
            width=width, style=style, antialias=antialias, orient=orient,
            backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_kde(
        self,
        data: ArrayLike1D,
        *,
        bottom: float = 0.0,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        band_width: float | Literal["scott", "silverman"] = "scott",
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
    ) -> _l.Band:
        """
        Add data as a band layer representing kernel density estimation (KDE).

        Parameters
        ----------
        data : array-like
            1D data to calculate the KDE.
        bottom : float, default 0.0
            Scalar value that define the height of the bottom line.
        name : str, optional
            Name of the layer, by default None
        orient : str, Orientation, default Orientation.VERTICAL
            Orientation of the KDE.
        band_width : float or str, default "scott"
            Band width parameter of KDE. Must be a number or a string as the
            method to automatic determination.
        color : color-like, default None
            Color of the band face.,
        alpha : float, default 1.0
            Alpha channel of the band face.
        hatch : str, FacePattern, default FacePattern.SOLID
            Hatch of the band face.

        Returns
        -------
        Kde
            The KDE layer.
        """
        name = self._coerce_name(_l.Band, name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)

        layer = _lg.Kde.from_array(
            data, bottom=bottom, scale=1, band_width=band_width, name=name,
            orient=orient, color=color, width=width, style=style,
            backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_text(
        self,
        x: ArrayLike1D,
        y: ArrayLike1D,
        string: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        family: str | None = None,
    ) -> _l.Texts[_mixin.ConstFace, _mixin.ConstEdge, _mixin.ConstFont]:
        ...

    @overload
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
        family: str | None = None,
    ) -> _l.Texts[_mixin.ConstFace, _mixin.ConstEdge, _mixin.ConstFont]:
        ...

    def add_text(
        self,
        x,
        y,
        string,
        *,
        color="black",
        size=12,
        rotation=0.0,
        anchor=Alignment.BOTTOM_LEFT,
        family=None,
    ):
        """
        Add a text layer to the canvas.

        >>> canvas.add_text([0, 0], [1, 1], ["text-0", "text-1])
        >>> canvas.add_text(...).with_face(color="red")  # with background
        >>> canvas.add_text(...).with_edge(color="red")  # with outline

        Parameters
        ----------
        x : float or array-like
            X position of the text.
        y : float or array-like
            Y position of the text.
        string : str or list[str]
            Text string to display.
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
        Text
            The text layer.
        """
        if (
            isinstance(x, (int, float, np.number))
            and isinstance(y, (int, float, np.number))
            and isinstance(string, str)
        ):
            x, y, string = [x], [y], [string]
        x_, y_ = normalize_xy(x, y)
        if isinstance(string, str):
            string = [string] * x_.size
        elif len(string) != x_.size:
            raise ValueError("Expected string to have the same size as x/y")
        layer = _l.Texts(
            x_, y_, string, color=color, size=size, rotation=rotation, anchor=anchor,
            family=family, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_image(
        self,
        image: ArrayLike,
        *,
        name: str | None = None,
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
        cmap : ColormapType, default "gray"
            Colormap used for the image.
        clim : (float or None, float or None) or None
            Contrast limits. If None, the limits are automatically determined by
            min and max of the data. You can also pass None separately to either
            limit to use the default behavior.
        flip_canvas : bool, default True
            If True, flip the canvas vertically so that the image looks normal.

        Returns
        -------
        Image
            The image layer.
        """
        layer = _l.Image(
            image, name=name, cmap=cmap, clim=clim, backend=self._get_backend()
        )
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

    @overload
    def group_layers(
        self,
        layers: Iterable[_l.Layer],
        name: str | None = None,
    ) -> _l.LayerGroup:
        ...

    @overload
    def group_layers(self, *layers: _l.Layer, name: str | None = None) -> _l.LayerGroup:
        ...

    def group_layers(self, layers, *more_layers, name=None):
        """
        Group layers.

        Parameters
        ----------
        layers : iterable of Layer
            Layers to group.

        Returns
        -------
        LayerGroup
            The grouped layer.
        """
        if more_layers:
            if not isinstance(layers, _l.Layer):
                raise TypeError("No overload matches the arguments")
            layers = [layers, *more_layers]
        return _lg.LayerTuple(layers, name=name)

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
        """This function will be called when a layer is inserted to the canvas."""
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
        small_diff = 1e-6
        if np.isnan(xmax) or np.isnan(xmin):
            xmin, xmax = self.x.lim
        elif xmax - xmin < small_diff:
            xmin -= 0.05
            xmax += 0.05
        else:
            dx = (xmax - xmin) * pad_rel
            xmin -= dx
            xmax += dx
        if np.isnan(ymax) or np.isnan(ymin):
            ymin, ymax = self.y.lim
        elif ymax - ymin < small_diff:
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

        _canvas = self._canvas()
        for l in _iter_layers(layer):
            _canvas._plt_add_layer(l._backend)
            l._connect_canvas(self)

        if isinstance(layer, _l.LayerWrapper):
            # TODO: check if connecting LayerGroup is necessary
            layer._connect_canvas(self)
        # autoscale
        if isinstance(layer, _l.Image):
            pad_rel = 0
        else:
            pad_rel = 0.025
        self._autoscale_for_layer(layer, pad_rel=pad_rel)
        if isinstance(layer, (_l.LayerGroup, _l.LayerWrapper)):
            self._cb_reordered()

    def _cb_inserted_overlay(self, idx: int, layer: _l.Layer):
        _canvas = self._canvas()
        fn = self._get_backend().get("as_overlay")
        for l in _iter_layers(layer):
            _canvas._plt_add_layer(l._backend)
            fn(l._backend, _canvas)
            l._connect_canvas(self)

        if isinstance(layer, _l.LayerWrapper):
            # TODO: check if connecting LayerGroup is necessary
            fn(l._backend, _canvas)
            layer._connect_canvas(self)

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
            elif isinstance(layer, _l.LayerWrapper):
                for child in _iter_layers(layer):
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
            # this patch will delay initialization by "_init_canvas" until the backend
            # objects are created.
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
    elif isinstance(layer, _l.LayerWrapper):
        yield from _iter_layers(layer._base_layer)
    else:
        raise TypeError(f"Unknown layer type: {type(layer).__name__}")
