from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
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
from whitecanvas.backend import Backend, patch_dummy_backend
from whitecanvas.canvas import _namespaces as _ns
from whitecanvas.canvas import dataframe as _df
from whitecanvas.canvas import layerlist as _ll
from whitecanvas.canvas._between import BetweenPlotter
from whitecanvas.canvas._dims import Dims
from whitecanvas.canvas._fit import FitPlotter
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.canvas._stacked import StackOverPlotter
from whitecanvas.layers import _legend, _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.types import (
    Alignment,
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    HistBinType,
    KdeBandWidthType,
    LineStyle,
    Location,
    LocationStr,
    Orientation,
    OrientationLike,
    Rect,
    StepStyle,
    StepStyleStr,
    Symbol,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xy
from whitecanvas.utils.predicate import not_starts_with_underscore
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec, Self

    _P = ParamSpec("_P")
    _DF = TypeVar("_DF")

_L = TypeVar("_L", bound=_l.Layer)
_L0 = TypeVar("_L0", _l.Bars, _l.Band)


class CanvasEvents(SignalGroup):
    lims = Signal(Rect)
    drawn = Signal()


class CanvasNDBase(ABC):
    title = _ns.TitleNamespace()
    x = _ns.XAxisNamespace()
    y = _ns.YAxisNamespace()
    layers = _ll.LayerList()
    events: CanvasEvents

    def __init__(self, palette: ColormapType | None = None):
        if palette is None:
            palette = theme.get_theme().palette
        self.events = CanvasEvents()
        self._color_palette = ColorPalette(palette)
        self._is_grouping = False

    @abstractmethod
    def _get_backend(self) -> Backend:
        """Return the backend."""

    @abstractmethod
    def _canvas(self) -> protocols.CanvasProtocol:
        """Return the canvas object."""

    @abstractmethod
    def _autoscale_for_layer(self, layer: _l.Layer):
        """Autoscale the canvas for the given layer."""

    def _draw_canvas(self):
        self._canvas()._plt_draw()
        self.events.drawn.emit()

    def _coerce_name(self, name: str | None, default: str = "_data") -> str:
        if name is None:
            basename = default
            name = f"{default}-0"
        else:
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

        _canvas = self._canvas()
        for l in _iter_layers(layer):
            _canvas._plt_add_layer(l._backend)
            l._connect_canvas(self)

        if isinstance(layer, _l.LayerWrapper):
            # TODO: check if connecting LayerGroup is necessary
            layer._connect_canvas(self)
        # autoscale
        self._autoscale_for_layer(layer)
        self._cb_reordered()

    def _cb_reordered(self):
        layer_backends = []
        for layer in self.layers:
            if isinstance(layer, _l.PrimitiveLayer):
                layer_backends.append(layer._backend)
            elif isinstance(layer, _l.LayerGroup):
                for child in layer.iter_primitive():
                    layer_backends.append(child._backend)
            elif isinstance(layer, _l.LayerWrapper):
                for child in _iter_layers(layer):
                    layer_backends.append(child._backend)
            else:
                raise RuntimeError(f"type {type(layer)} not expected")
        self._canvas()._plt_reorder_layers(layer_backends)

    def _cb_removed(self, idx: int, layer: _l.Layer):
        if self._is_grouping:
            return
        _canvas = self._canvas()
        for l in _iter_layers(layer):
            _canvas._plt_remove_layer(l._backend)
            l._disconnect_canvas(self)

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

    @property
    def autoscale_enabled(self) -> bool:
        """Return whether autoscale is enabled."""
        return self._autoscale_enabled

    @autoscale_enabled.setter
    def autoscale_enabled(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise TypeError(f"Expected a bool, got {type(enabled)}.")
        self._autoscale_enabled = enabled

    @contextmanager
    def autoscale_context(self, enabled: bool):
        """Context manager to temporarily change the autoscale state."""
        _was_enabled = self.autoscale_enabled
        self.autoscale_enabled = enabled
        try:
            yield
        finally:
            self.autoscale_enabled = _was_enabled


class CanvasBase(CanvasNDBase):
    """Base class for any canvas object."""

    dims = Dims()
    overlays = _ll.LayerList()
    mouse = _ns.MouseNamespace()

    def __init__(self, palette: ColormapType | None = None):
        super().__init__(palette)
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
        self.layers.events.inserted.connect(
            self._cb_inserted, unique=True, max_args=None
        )
        self.layers.events.removed.connect(self._cb_removed, unique=True, max_args=None)
        self.layers.events.reordered.connect(
            self._cb_reordered, unique=True, max_args=None
        )
        self.layers.events.connect(self._draw_canvas, unique=True, max_args=None)

        self.overlays.events.inserted.connect(
            self._cb_overlay_inserted, unique=True, max_args=None
        )
        self.overlays.events.removed.connect(
            self._cb_removed, unique=True, max_args=None
        )
        self.overlays.events.connect(self._draw_canvas, unique=True, max_args=None)

        canvas = self._canvas()
        canvas._plt_connect_xlim_changed(self._emit_xlim_changed)
        canvas._plt_connect_ylim_changed(self._emit_ylim_changed)

        if hasattr(canvas, "_plt_canvas_hook"):
            canvas._plt_canvas_hook(self)

    def _install_mouse_events(self):
        canvas = self._canvas()
        canvas._plt_connect_mouse_click(self.mouse.clicked.emit)
        canvas._plt_connect_mouse_click(self.mouse.moved.emit)
        canvas._plt_connect_mouse_drag(self.mouse.moved.emit)
        canvas._plt_connect_mouse_release(self.mouse.moved.emit)
        canvas._plt_connect_mouse_double_click(self.mouse.double_clicked.emit)
        canvas._plt_connect_mouse_double_click(self.mouse.moved.emit)

    def _emit_xlim_changed(self, lim):
        self.x.events.lim.emit(lim)
        self.events.lims.emit(Rect(*lim, *self.y.lim))

    def _emit_ylim_changed(self, lim):
        self.y.events.lim.emit(lim)
        self.events.lims.emit(Rect(*self.x.lim, *lim))

    def _emit_mouse_moved(self, ev):
        """Emit mouse moved event with autoscaling blocked"""
        self.mouse.moved.emit(ev)

    @property
    def mouse_clicked(self):
        warnings.warn(
            "`canvas.events.mouse_clicked` is deprecated. Use `canvas.mouse.clicked` "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mouse.clicked

    @property
    def mouse_moved(self):
        warnings.warn(
            "`canvas.events.mouse_moved` is deprecated. Use `canvas.mouse.moved` "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mouse.clicked

    @property
    def mouse_double_clicked(self):
        warnings.warn(
            "`canvas.events.mouse_double_clicked` is deprecated. Use "
            "`canvas.mouse.double_clicked` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mouse.clicked

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

    @property
    def mouse_enabled(self) -> bool:
        warnings.warn(
            "`canvas.mouse_enabled` is deprecated. Use `canvas.mouse.enabled` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mouse.enabled

    @mouse_enabled.setter
    def mouse_enabled(self, enabled: bool):
        warnings.warn(
            "`canvas.mouse_enabled` is deprecated. Use `canvas.mouse.enabled` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.mouse.enabled = enabled

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
            if is_real_number(xpad):
                dx0 = dx1 = xpad * xrange
            else:
                dx0, dx1 = xpad[0] * xrange, xpad[1] * xrange
            xmin -= dx0
            xmax += dx1
        if ypad is not None:
            yrange = ymax - ymin
            if is_real_number(ypad):
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

    def install_second_x(self, *, palette: ColormapType | None = None) -> Canvas:
        """Create a twin canvas that has a secondary x-axis and shared y-axis."""
        try:
            new = self._canvas()._plt_twiny()
        except AttributeError:
            raise NotImplementedError(
                f"Backend {self._get_backend()} does not support `install_second_x`."
            ) from None
        canvas = Canvas.from_backend(new, palette=palette, backend=self._get_backend())
        canvas._init_canvas()
        return canvas

    def install_second_y(self, *, palette: ColormapType | None = None) -> Canvas:
        """Create a twin canvas that has a secondary y-axis and shared x-axis."""
        try:
            new = self._canvas()._plt_twinx()
        except AttributeError:
            raise NotImplementedError(
                f"Backend {self._get_backend()} does not support `install_second_y`."
            ) from None
        canvas = Canvas.from_backend(new, palette=palette, backend=self._get_backend())
        canvas._init_canvas()
        return canvas

    @overload
    def install_inset(
        self, left: float, right: float, bottom: float, top: float, *,
        palette: ColormapType | None = None
    ) -> Canvas:  # fmt: skip
        ...

    @overload
    def install_inset(
        self, rect: Rect | tuple[float, float, float, float], /, *,
        palette: ColormapType | None = None
    ) -> Canvas:  # fmt: skip
        ...

    def install_inset(self, *args, palette=None, **kwargs) -> Canvas:
        """
        Install a new canvas pointing to an inset of the current canvas.

        >>> canvas.install_inset(left=0.1, right=0.9, bottom=0.1, top=0.9)
        >>> canvas.install_inset([0.1, 0.9, 0.1, 0.9])  # or a sequence
        """
        # normalize input
        if len(args) == 1 and not kwargs:
            rect = args[0]
            if not isinstance(rect, Rect):
                rect = Rect.with_check(*rect)
        else:
            rect = Rect.with_check(*args, **kwargs)
        try:
            new = self._canvas()._plt_inset(rect)
        except AttributeError:
            raise NotImplementedError(
                f"Backend {self._get_backend()} does not support `install_inset`"
            ) from None
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
        *,
        visible: bool | None = None,
        color: ColorType | None = None,
    ):
        """
        Update axes appearance.

        Parameters
        ----------
        visible : bool, optional
            Whether to show the axes.
        color : color-like, optional
            Color of the axes.
        """
        if visible is not None:
            self.x.ticks.visible = self.y.ticks.visible = visible
        if color is not None:
            self.x.color = self.y.color = color
            self.x.ticks.color = self.y.ticks.color = color
            self.x.label.color = self.y.label.color = color
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
            self.title.visible = True
        if x is not None:
            self.x.label.text = x
            self.x.label.visible = True
        if y is not None:
            self.y.label.text = y
            self.y.label.visible = True
        return self

    def update_font(
        self,
        size: float | None = None,
        color: ColorType | None = None,
        family: str | None = None,
    ) -> Self:
        """
        Update all the fonts, including the title, x/y labels and x/y tick labels.

        Parameters
        ----------
        size : float, optional
            New font size.
        color : color-like, optional
            New font color.
        family : str, optional
            New font family.
        """
        if size is not None:
            self.title.size = self.x.label.size = self.y.label.size = size
            self.x.ticks.size = self.y.ticks.size = size
        if family is not None:
            self.title.family = self.x.label.family = self.y.label.family = family
            self.x.ticks.family = self.y.ticks.family = family
        if color is not None:
            self.title.color = self.x.label.color = self.y.label.color = color
            self.x.ticks.color = self.y.ticks.color = color
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
        x : str, optional
            Name of the column that will be used for the x-axis. Must be numerical.
        y : str, optional
            Name of the column that will be used for the y-axis. Must be numerical.
        update_labels : bool, default True
            If True, update the x/y labels to the corresponding names.

        Returns
        -------
        CatPlotter
            Plotter object.
        """
        plotter = _df.CatPlotter(self, data, x, y, update_labels=update_labels)
        return plotter

    def cat_x(
        self,
        data: _DF,
        x: str | Sequence[str] | None = None,
        y: str | None = None,
        *,
        update_labels: bool = True,
        numeric_axis: bool = False,
    ) -> _df.XCatPlotter[Self, _DF]:
        """
        Categorize input data for plotting with x-axis as a categorical axis.

        Parameters
        ----------
        data : tabular data
            Any categorizable data. Currently, dict, pandas.DataFrame, and
            polars.DataFrame are supported.
        x : str or sequence of str, optional
            Name of the column(s) that will be used for the x-axis. Must be categorical.
        y : str, optional
            Name of the column that will be used for the y-axis. Must be numerical.
        update_labels : bool, default True
            If True, update the x/y labels to the corresponding names.
        numeric_axis : bool, default False
            If True, the x-axis will be treated as a numerical axis. For example, if
            categories are [2, 4, 8], the x coordinates will be mapped to [0, 1, 2] by
            default, but if this option is True, the x coordinates will be [2, 4, 8].

        Returns
        -------
        XCatPlotter
            Plotter object.
        """
        return _df.XCatPlotter(self, data, x, y, update_labels, numeric=numeric_axis)

    def cat_y(
        self,
        data: _DF,
        x: str | None = None,
        y: str | Sequence[str] | None = None,
        *,
        update_labels: bool = True,
        numeric_axis: bool = False,
    ) -> _df.YCatPlotter[Self, _DF]:
        """
        Categorize input data for plotting with y-axis as a categorical axis.

        Parameters
        ----------
        data : tabular data
            Any categorizable data. Currently, dict, pandas.DataFrame, and
            polars.DataFrame are supported.
        x : str, optional
            Name of the column that will be used for the x-axis. Must be numerical.
        y : str or sequence of str, optional
            Name of the column(s) that will be used for the y-axis. Must be categorical.
        update_labels : bool, default True
            If True, update the x/y labels to the corresponding names.
        numeric_axis : bool, default False
            If True, the x-axis will be treated as a numerical axis. For example, if
            categories are [2, 4, 8], the y coordinates will be mapped to [0, 1, 2] by
            default, but if this option is True, the y coordinates will be [2, 4, 8].

        Returns
        -------
        YCatPlotter
            Plotter object
        """
        return _df.YCatPlotter(self, data, y, x, update_labels, numeric=numeric_axis)

    def cat_xy(
        self,
        data: _DF,
        x: str | Sequence[str],
        y: str | Sequence[str],
        *,
        update_labels: bool = True,
    ) -> _df.XYCatPlotter[Self, _DF]:
        """
        Categorize input data for plotting with both axes as categorical.

        Parameters
        ----------
        data : tabular data
            Any categorizable data. Currently, dict, pandas.DataFrame, and
            polars.DataFrame are supported.
        x : str or sequence of str, optional
            Name of the column(s) that will be used for the x-axis. Must be categorical.
        y : str or sequence of str, optional
            Name of the column(s) that will be used for the y-axis. Must be categorical.
        update_labels : bool, default True
            If True, update the x/y labels to the corresponding names.

        Returns
        -------
        XYCatPlotter
            Plotter object
        """
        return _df.XYCatPlotter(self, data, x, y, update_labels)

    def stack_over(self, layer: _L0) -> StackOverPlotter[Self, _L0]:
        """
        Stack new data over the existing layer.

        For example following code

        >>> bars_0 = canvas.add_bars(x, y0)
        >>> bars_1 = canvas.stack_over(bars_0).add(y1)
        >>> bars_2 = canvas.stack_over(bars_1).add(y2)

        will result in a bar plot like this

        ```
         ┌───┐
         ├───│┌───┐
         │   │├───│
         ├───│├───│
        ─┴───┴┴───┴─
        ```
        """
        if not isinstance(layer, (_l.Bars, _l.Band, _lg.StemPlot, _lg.LabeledBars)):
            raise TypeError(
                f"Only Bars, StemPlot and Band are supported as an input, "
                f"got {type(layer)!r}."
            )
        return StackOverPlotter(self, layer)

    # TODO
    # def annotate(self, layer, at: int):
    #     ...

    def between(self, l0, l1) -> BetweenPlotter[Self]:
        return BetweenPlotter(self, l0, l1)

    def fit(self, layer: _l.DataBoundLayer[_P]) -> FitPlotter[Self, _P]:
        """The fit plotter namespace."""
        return FitPlotter(self, layer)

    def add_legend(
        self,
        layers: Sequence[str | _l.Layer] | None = None,
        *,
        location: Location | LocationStr = "top_right",
        title: str | None = None,
        name_filter: Callable[[str], bool] = not_starts_with_underscore,
    ):
        """
        Add legend items to the canvas.

        Parameters
        ----------
        layers : sequence of layer or str, optional
            Which item to be added to the legend. If str is given, it will be converted
            into a legend title label.
        location : LegendLocation, default "top_right"
            Location of the legend. Can be combination of "top", "bottom", "left",
            "right" and "center" (e.g., "top_left", "center_right").

            ```
                   (2) left  center right
                         v     v     v
              (1)     ┌─────────────────┐
               top -> │                 │
            center -> │     canvas      │
            bottom -> │                 │
                      └─────────────────┘
            ```

            Some backends also support adding legend outside the canvas. Following
            strings suffixed with "_side" can be used in combination with those strings
            above (e.g., "bottom_side_rigth", "right_side_top").

            ```
               top_side -> ┌────────┐
                        ┌──┼────────┼──┐
            left_side ->│  │ canvas │  │<- right_side
                        └──┼────────┼──┘
            bottom_side -> └────────┘
            ```
        title : str, optional
            If given, title label will be added as the first legend item.
        name_filter : callable, default not_starts_with_underscore
            A callable that returns True if the name should be included in the legend.
        """
        if layers is None:
            layers = list(self.layers)
        if title is not None:
            layers = [title, *layers]
        location = Location.parse(location)

        items = list[tuple[str, _legend.LegendItem]]()
        for layer in layers:
            if isinstance(layer, str):
                items.append((layer, _legend.TitleItem()))
            elif isinstance(layer, _l.Layer):
                if not name_filter(layer.name):
                    continue
                items.append((layer.name, layer._as_legend_item()))
            else:
                raise TypeError(f"Expected a list of layer or str, got {type(layer)}.")
        self._canvas()._plt_make_legend(items, location)

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
        name = self._coerce_name(name)
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
        name = self._coerce_name(name)
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
    def add_step(
        self, ydata: ArrayLike1D, *, name: str | None = None,
        where: StepStyleStr | StepStyle = "pre", color: ColorType | None = None,
        width: float | None = None, style: LineStyle | str | None = None,
        alpha: float = 1.0, orient: OrientationLike = "horizontal",
        antialias: bool = True,
    ) -> _l.LineStep:  # fmt: skip
        ...

    @overload
    def add_step(
        self, xdata: ArrayLike1D, ydata: ArrayLike1D, *, name: str | None = None,
        where: StepStyleStr | StepStyle = "pre", color: ColorType | None = None,
        width: float | None = None, style: LineStyle | str | None = None,
        alpha: float = 1.0, orient: OrientationLike = "horizontal",
        antialias: bool = True,
    ) -> _l.LineStep:  # fmt: skip
        ...

    def add_step(
        self,
        *args,
        name=None,
        where="pre",
        color=None,
        width=None,
        style=None,
        alpha=1.0,
        antialias=True,
    ):
        """
        Add a step plot to the canvas.

        >>> canvas.add_step(y, ...)
        >>> canvas.add_step(x, y, ...)

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        where : str or StepStyle, default "pre"
            Where the step should be placed.
        color : color-like, optional
            Color of the steps.
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
        LineStep
            The line-step layer.
        """
        xdata, ydata = normalize_xy(*args)
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.LineStep(
            xdata, ydata, name=name, color=color, width=width, style=style, where=where,
            alpha=alpha, antialias=antialias, backend=self._get_backend()
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_bars(
        self, center: ArrayLike1D, height: ArrayLike1D, *,
        bottom: ArrayLike1D | None = None, name=None,
        orient: OrientationLike = "vertical", extent: float | None = None,
        color: ColorType | None = None, alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> _l.Bars[_mixin.ConstFace, _mixin.ConstEdge]:  # fmt: skip
        ...

    @overload
    def add_bars(
        self, height: ArrayLike1D, *, bottom: ArrayLike1D | None = None,
        name=None, orient: OrientationLike = "vertical",
        extent: float | None = None, color: ColorType | None = None,
        alpha: float = 1.0, hatch: str | Hatch | None = None,
    ) -> _l.Bars[_mixin.ConstFace, _mixin.ConstEdge]:  # fmt: skip
        ...

    def add_bars(
        self,
        *args,
        bottom=None,
        name=None,
        orient="vertical",
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
        orient : str or Orientation, default "vertical"
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
        name = self._coerce_name(name)
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
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        name: str | None = None,
        shape: Literal["step", "polygon", "bars"] = "bars",
        kind: Literal["count", "density", "frequency", "percent"] = "count",
        orient: OrientationLike = "vertical",
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
        bins : int or 1D array-like, default "auto"
            Bins of the histogram. This parameter will directly be passed
            to `np.histogram`.
        limits : (float, float), optional
            Limits in which histogram will be built. This parameter will equivalent to
            the `range` paraneter of `np.histogram`.
        name : str, optional
            Name of the layer.
        shape : {"step", "polygon", "bars"}, default "bars"
            Shape of the histogram. This parameter defines how to convert the data into
            the line nodes.
        kind : {"count", "density", "probability", "frequency", "percent"}, optional
            Kind of the histogram.
        orient : str or Orientation, default "vertical"
            Orientation of the bars.
        color : color-like, optional
            Color of the bars.

        Returns
        -------
        Bars
            The bars layer that represents the histogram.
        """
        name = self._coerce_name(name)
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
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
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

    def add_rects(
        self,
        coords: ArrayLike,
        *,
        name=None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> _l.Rects[_mixin.ConstFace, _mixin.ConstEdge]:
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        hatch = theme._default("bars.hatch", hatch)
        layer = _l.Rects(
            coords, name=name, color=color, alpha=alpha, hatch=hatch,
            backend=self._get_backend()
        )  # fmt: skip
        return self.add_layer(layer)

    def add_cdf(
        self,
        data: ArrayLike1D,
        *,
        name: str | None = None,
        orient: OrientationLike = "vertical",
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
        orient : str or Orientation, default "vertical"
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
        name = self._coerce_name(name)
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
        orient: OrientationLike = "vertical",
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
        orient : str or Orientation, default "vertical"
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
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        layer = _l.Spans(
            spans, name=name, orient=orient, color=color, alpha=alpha,
            hatch=hatch, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_vectors(
        self,
        x: ArrayLike1D,
        y: ArrayLike1D,
        vx: ArrayLike1D,
        vy: ArrayLike1D,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.Vectors:
        """
        Add a vector field to the canvas.

        >>> canvas.add_vectors(x, y, vx, vy)

        Parameters
        ----------
        x : array-like
            X coordinates of the vectors.
        y : array-like
            Y coordinates of the vectors.
        vx : array-like
            X components of the vectors.
        vy : array-like
            Y components of the vectors.
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
        Vectors
            The vectors layer.
        """
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.Vectors(
            as_array_1d(x, dtype=np.float32), as_array_1d(y, dtype=np.float32),
            as_array_1d(vx, dtype=np.float32), as_array_1d(vy, dtype=np.float32),
            name=name, color=color, width=width, style=style,
            alpha=alpha, antialias=antialias, backend=self._get_backend(),
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
        name = self._coerce_name(name)
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
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.InfCurve[_P]:
        """
        Add an infinite curve to the canvas.

        >>> canvas.add_infcurve(lambda x: x ** 2)  # parabola
        >>> canvas.add_infcurve(lambda x, a: np.sin(a*x)).update_params(2)  # parametric

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
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.InfCurve(
            model, bounds=bounds, name=name, color=color, width=width, alpha=alpha,
            style=style, antialias=antialias, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_hline(
        self,
        y: float,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.InfLine:
        """
        Add a infinite horizontal line to the canvas.

        Parameters
        ----------
        y : float
            Y coordinate of the line.
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
        width: float | None = None,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.InfLine:
        """
        Add a infinite vertical line to the canvas.

        Parameters
        ----------
        x : float
            X coordinate of the line.
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
        orient: OrientationLike = "vertical",
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
        orient : str, Orientation, default "vertical"
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
        name = self._coerce_name(name)
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
        orient: OrientationLike = "vertical",
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
        alpha: float = 1.0,
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
        orient : str or Orientation, default "vertical"
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
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = _l.Errorbars(
            xdata, ylow, yhigh, name=name, color=color, width=width,
            style=style, antialias=antialias, capsize=capsize, alpha=alpha,
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
        orient: OrientationLike = "vertical",
        color: ColorType = "black",
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        alpha: float = 1.0,
    ) -> _l.Rug:
        """
        Add input data as a rug plot.

        >>> canvas.add_rug([2, 4, 5, 8, 11])

        ```
          │ ││  │   │
        ──┴─┴┴──┴───┴──> x
          2 45  8   11
        ```

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
        orient : str or Orientation, default "vertical"
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
        name = self._coerce_name(name)
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
        orient: OrientationLike = "vertical",
        band_width: KdeBandWidthType = "scott",
        color: ColorType | None = None,
        width: float | None = None,
        style: LineStyle | str | None = None,
    ) -> _lg.Kde:
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
        orient : str, Orientation, default "vertical"
            Orientation of the KDE.
        band_width : float or str, default "scott"
            Method to calculate the estimator bandwidth.
        color : color-like, default None
            Color of the band face.
        width : float, optional
            Line width of the outline.
        style : str or LineStyle, optional
            Line style of the outline.

        Returns
        -------
        Kde
            The KDE layer.
        """
        name = self._coerce_name(name)
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
        self, x: ArrayLike1D, y: ArrayLike1D, string: list[str], *,
        color: ColorType = "black", size: float = 12, rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT, family: str | None = None,
    ) -> _l.Texts[_mixin.ConstFace, _mixin.ConstEdge, _mixin.ConstFont]:  # fmt: skip
        ...

    @overload
    def add_text(
        self, x: float, y: float, string: str, *, color: ColorType = "black",
        size: float = 12, rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT, family: str | None = None,
    ) -> _l.Texts[_mixin.ConstFace, _mixin.ConstEdge, _mixin.ConstFont]:  # fmt: skip
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
            X position(s) of the text.
        y : float or array-like
            Y position(s) of the text.
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
        Texts
            The text layer.
        """
        if is_real_number(x) and is_real_number(y) and isinstance(string, str):
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
        cmap: ColormapType | None = None,
        clim: tuple[float | None, float | None] | None = None,
        flip_canvas: bool = True,
        lock_aspect: bool = True,
    ) -> _l.Image:
        """
        Add an image layer to the canvas.

        This method automatically flips the image vertically by default. `add_heatmap`
        does the similar thing with slightly different default settings.

        Parameters
        ----------
        image : ArrayLike
            Image data. Must be 2D or 3D array. If 3D, the last dimension must be
            RGB(A). Note that the first dimension is the vertical axis.
        cmap : ColormapType, optional
            Colormap used for the image. If None, the theme default for image colormap
            will be used.
        clim : (float or None, float or None) or None
            Contrast limits. If None, the limits are automatically determined by min and
            max of the data. You can also pass None separately to either limit to use
            the default behavior.
        flip_canvas : bool, default True
            If True, flip the canvas vertically so that the image looks normal.
        lock_aspect : bool, default True
            If True, lock the aspect ratio of the canvas to 1:1.

        Returns
        -------
        Image
            The image layer.
        """
        cmap = theme._default("colormap_image", cmap)
        layer = _l.Image(
            image, name=name, cmap=cmap, clim=clim, backend=self._get_backend()
        )
        self.add_layer(layer)
        if flip_canvas and not self.y.flipped:
            self.y.flipped = True
        if lock_aspect:
            self.aspect_ratio = 1.0
        return layer

    def add_heatmap(
        self,
        image: ArrayLike,
        *,
        name: str | None = None,
        cmap: ColormapType = "inferno",
        clim: tuple[float | None, float | None] | None = None,
        flip_canvas: bool = False,
    ) -> _l.Image:
        """
        Add an image layer to the canvas as a heatmap.

        Use `add_image` to add the layer as an image.

        Parameters
        ----------
        image : ArrayLike
            Image data. Must be 2D or 3D array. If 3D, the last dimension must be
            RGB(A). Note that the first dimension is the vertical axis.
        cmap : ColormapType, default "gray"
            Colormap used for the image.
        clim : (float or None, float or None) or None
            Contrast limits. If None, the limits are automatically determined by min and
            max of the data. You can also pass None separately to either limit to use
            the default behavior.
        flip_canvas : bool, default False
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
    ) -> _l.LayerGroup: ...

    @overload
    def group_layers(
        self, *layers: _l.Layer, name: str | None = None
    ) -> _l.LayerGroup: ...

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

    def _autoscale_for_layer(
        self,
        layer: _l.Layer,
        pad_rel: float | None = None,
        maybe_empty: bool = True,
    ):
        """This function will be called when a layer is inserted to the canvas."""
        if not self.autoscale_enabled:
            return
        if pad_rel is None:
            pad_rel = 0 if layer._NO_PADDING_NEEDED else 0.025
        xmin, xmax, ymin, ymax = layer.bbox_hint()
        if len(self.layers) > 1 or not maybe_empty:
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
            if (
                xmin != 0
                or not layer._ATTACH_TO_AXIS
                or getattr(layer, "orient", None) is not Orientation.HORIZONTAL
            ):
                xmin -= dx
            xmax += dx
        if np.isnan(ymax) or np.isnan(ymin):
            ymin, ymax = self.y.lim
        elif ymax - ymin < small_diff:
            ymin -= 0.05
            ymax += 0.05
        else:
            dy = (ymax - ymin) * pad_rel
            if (
                ymin != 0
                or not layer._ATTACH_TO_AXIS
                or getattr(layer, "orient", None) is not Orientation.VERTICAL
            ):
                ymin -= dy
            ymax += dy
        self.lims = xmin, xmax, ymin, ymax

    def _cb_overlay_inserted(self, idx: int, layer: _l.Layer):
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
        yield from layer.iter_primitive()
    elif isinstance(layer, _l.LayerWrapper):
        yield from _iter_layers(layer._base_layer)
    else:
        raise TypeError(f"Unknown layer type: {type(layer).__name__}")
