from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Iterable, Sequence, TypeVar, overload

import numpy as np
from cmap import Color, Colormap
from numpy.typing import NDArray

from whitecanvas import layers as _l
from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _legend, _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.layers._deserialize import construct_layer
from whitecanvas.layers._legend import LegendItem
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._dataframe import DFRegPlot
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, from_dict
from whitecanvas.types import (
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    KdeBandWidthType,
    LineStyle,
    Orientation,
    Symbol,
    _Void,
)
from whitecanvas.utils.collections import OrderedSet
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas import CanvasBase

_DF = TypeVar("_DF")
_void = _Void()


class _MarkerLikeMixin:
    _source: DataFrameWrapper[_DF]

    @overload
    def update_color(self, value: ColorType) -> Self: ...

    @overload
    def update_color(
        self, by: str | Iterable[str], palette: ColormapType | None = None,
    ) -> Self:  # fmt: skip
        ...

    def update_color(self, by, /, palette=None) -> Self:
        """
        Update colors by a constant value or according to a column.

        >>> layer.update_color("red")  # set all components to red
        >>> layer.update_color("var")  # set color according to the column "var"
        >>> layer.update_color("var", palette=["pink", "#00FF3E"])  # use palette
        """
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            if palette is not None:
                raise TypeError("`palette` can only be used with a column name.")
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._apply_color(color_by.map(self._source))
        self._color_by = color_by
        return self

    @overload
    def update_alpha(self, value: float) -> Self: ...

    @overload
    def update_alpha(
        self,
        by: str | Iterable[str],
        map_from: tuple[float, float] | None = None,
        map_to: tuple[float, float] = (0.2, 1.0),
    ) -> Self: ...

    def update_alpha(self, by, /, map_from=None, map_to=(0.2, 1.0)) -> Self:
        if isinstance(by, str):
            alpha_by = _p.AlphaPlan.from_range(by, range=map_to, domain=map_from)
        else:
            if map_from is not None:
                raise TypeError("`map_from` can only be used with a column name.")
            alpha_by = _p.AlphaPlan.from_const(float(by))
        self._apply_alpha(alpha_by.map(self._source))
        self._alpha_by = alpha_by
        return self

    def update_colormap(
        self,
        by: str,
        cmap: ColormapType = "viridis",
        clim: tuple[float, float] | None = None,
    ) -> Self:
        """
        Update colors according to a column with numeric values using a colormap.

        >>> layer.update_colormap("var", cmap="viridis")  # map "var" column
        >>> layer.update_colormap("var", cmap="viridis", clim=(0, 10))  # set contrast
        """
        if not isinstance(by, str):
            raise ValueError("Can only colormap by a single column.")
        color_by = _p.ColormapPlan.from_colormap(by, cmap=Colormap(cmap), clim=clim)
        self._apply_color(color_by.map(self._source))
        self._color_by = color_by
        return self

    def _update_color_or_colormap(self, by: str | Iterable[str]) -> Self:
        if (
            isinstance(by, str)
            and by in self._source
            and self._source[by].dtype.kind in "fiu"
        ):
            return self.update_colormap(by, theme._default("colormap_categorical"))
        self.update_color(by)
        return self

    @overload
    def update_width(self, value: float) -> Self: ...

    @overload
    def update_width(
        self, by: str, map_from: tuple[float, float] | None = None,
        map_to: tuple[float, float] = (1.0, 4.0),
    ) -> Self:  # fmt: skip
        ...

    def update_width(self, by, /, map_from=None, map_to=(1.0, 4.0)) -> Self:
        """
        Update the line widths by a constant number or according to a column.

        >>> layer.update_width(2.0)  # set widths of all components to 2.0
        >>> layer.update_width("var")  # set width according to the column "var"

        Parameters
        ----------
        map_from : tuple of float, optional
            Limits of values that will be linearly mapped to the edge width. Data
            points outside this range will be clipped. If not specified, the min/max
            of the data will be used.
        map_to : tuple of float, optional
            Minimum and maximum size of the markers.
        """
        if isinstance(by, str):
            width_by = _p.WidthPlan.from_range(by, range=map_to, domain=map_from)
        else:
            if map_from is not None:
                raise TypeError("`map_from` can only be used with a column name.")
            width_by = _p.WidthPlan.from_const(float(by))
        self._apply_width(width_by.map(self._source))
        self._width_by = width_by
        return self

    @overload
    def update_style(self, value: ColorType) -> Self: ...

    @overload
    def update_style(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self: ...

    def update_style(self, by, /, palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            style_by = _p.StylePlan.new(cov.columns, palette)
        else:
            if palette is not None:
                raise TypeError("`palette` can only be used with a column name.")
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        self._apply_style(style_by.map(self._source))
        self._style_by = style_by
        return self

    def _init_color_for_canvas(self, color, alpha, canvas: CanvasBase):
        if (
            color is not None
            and not self._color_by.is_const()
            and isinstance(self._color_by, _p.ColorPlan)
        ):
            self.update_color(self._color_by.by, palette=canvas._color_palette)
        elif color is None:
            self.update_color(canvas._color_palette.next())
        if alpha is not None:
            self.update_alpha(alpha)
        return self

    def _apply_color(self, color):
        """Set color array to the layer."""
        raise NotImplementedError

    def _apply_width(self, width):
        """Set width array to the layer."""
        raise NotImplementedError

    def _apply_alpha(self, alpha):
        """Set alpha array to the layer."""
        raise NotImplementedError

    def _apply_style(self, style):
        """Set style array to the layer."""
        raise NotImplementedError


class DFMarkers(
    _shared.DataFrameLayerWrapper[_lg.MarkerCollection, _DF], _MarkerLikeMixin
):
    def __init__(
        self,
        base: _lg.MarkerCollection,
        source: DataFrameWrapper[_DF],
        color_by: _p.ColorPlan | _p.ColormapPlan,
        edge_color_by: _p.ColorPlan | _p.ColormapPlan,
        hatch_by: _p.HatchPlan,
        size_by: _p.SizePlan,
        symbol_by: _p.SymbolPlan,
        width_by: _p.WidthPlan,
    ):
        self._color_by = color_by
        self._edge_color_by = edge_color_by
        self._hatch_by = hatch_by
        self._size_by = size_by
        self._symbol_by = symbol_by
        self._width_by = width_by
        super().__init__(base, source)

    @classmethod
    def from_jitters(
        cls,
        source: DataFrameWrapper[_DF],
        x: _jitter.JitterBase,
        y: _jitter.JitterBase,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        symbol: str | tuple[str, ...] | None = None,
        size: str | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ) -> Self:
        base = _lg.MarkerCollection.from_arrays(
            x.map(source), y.map(source), name=name, backend=backend
        )
        self = cls(
            base,
            source,
            color_by=_p.ColorPlan.default(),
            edge_color_by=_p.ColorPlan.default(),
            hatch_by=_p.HatchPlan.default(),
            size_by=_p.SizePlan.default(),
            symbol_by=_p.SymbolPlan.default(),
            width_by=_p.WidthPlan.default(),
        )
        if color is not None:
            self._update_color_or_colormap(color)
        if hatch is not None:
            self.update_hatch(hatch)
        if symbol is not None:
            self.update_symbol(symbol)
        if size is not None:
            self.update_size(size)
        else:
            self.update_size(theme.get_theme().markers.size)

        # set default hover text
        self.with_hover_template(default_template(source.iter_items()))
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a DFViolinPlot from a dictionary."""
        base, source = d["base"], d["source"]
        if isinstance(base, dict):
            base = construct_layer(base, backend=backend)
        if isinstance(source, dict):
            source = from_dict(source)
        return cls(
            base,
            source,
            color_by=_p.ColorPlan.from_dict_or_plan(d["color_by"]),
            edge_color_by=_p.ColorPlan.from_dict_or_plan(d["edge_color_by"]),
            hatch_by=_p.HatchPlan.from_dict_or_plan(d["hatch_by"]),
            size_by=_p.SizePlan.from_dict_or_plan(d["size_by"]),
            symbol_by=_p.SymbolPlan.from_dict_or_plan(d["symbol_by"]),
            width_by=_p.WidthPlan.from_dict_or_plan(d["width_by"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "base": self._base_layer.to_dict(),
            "source": self._source,
            "color_by": self._color_by,
            "edge_color_by": self._edge_color_by,
            "hatch_by": self._hatch_by,
            "size_by": self._size_by,
            "symbol_by": self._symbol_by,
            "width_by": self._width_by,
        }

    def _apply_color(self, color):
        self.base.face.color = np.asarray(color, dtype=np.float32)

    def _apply_width(self, width):
        self.base.with_edge(color=_void, width=width, style=_void)

    def _apply_alpha(self, alpha):
        self.base.face.alpha = alpha
        self.base.edge.alpha = alpha

    def _apply_style(self, style):
        self.base.with_edge(color=_void, width=_void, style=style)

    def update_edge_color(self, by: str | Iterable[str], palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        colors = color_by.map(self._source)
        self.base.edge.color = colors
        self._edge_color_by = color_by
        return self

    def update_edge_colormap(
        self,
        by: str,
        cmap: ColormapType | None = None,
        clim: tuple[float, float] | None = None,
    ) -> Self:
        if not isinstance(by, str):
            raise ValueError("Can only colormap by a single column.")
        if cmap is None:
            cmap = Colormap("viridis")
        else:
            cmap = Colormap(cmap)
        color_by = _p.ColormapPlan.from_colormap(by, cmap=cmap, clim=clim)
        colors = color_by.map(self._source)
        self.base.edge.color = colors
        self._edge_color_by = color_by
        return self

    def update_hatch(self, by: str | Iterable[str], palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            hatch_by = _p.HatchPlan.new(cov.columns, values=palette)
        else:
            hatch_by = _p.HatchPlan.from_const(Hatch(cov.value))
        hatches = hatch_by.map(self._source)
        self.base.face.hatch = hatches
        self._hatch_by = hatch_by
        return self

    @overload
    def update_size(self, value: float) -> Self: ...

    @overload
    def update_size(
        self, by: str, /, map_from: tuple[float, float] | None = None,
        map_to: tuple[float, float] = (3, 15),
    ) -> Self:  # fmt: skip
        ...

    def update_size(self, by, /, map_from=None, map_to=(3, 15)):
        """
        Set the size of the markers.

        >>> layer.update_size(2.0)  # set size of all components to 2.0
        >>> layer.update_size("var")  # set sizes according to the column "var"

        Parameters
        ----------
        map_from : tuple of float, optional
            Limits of values that will be linearly mapped to the marker size. Data
            points outside this range will be clipped. If not specified, the min/max
            of the data will be used.
        map_to : tuple of float, optional
            Minimum and maximum size of the markers.
        """
        if isinstance(by, str):
            size_by = _p.SizePlan.from_range(by, range=map_to, domain=map_from)
        else:
            size_by = _p.SizePlan.from_const(float(by))
        self.base.size = size_by.map(self._source)
        self._size_by = size_by
        return self

    @overload
    def update_symbol(self, value: str | Symbol) -> Self: ...

    @overload
    def update_symbol(
        self,
        by: str | Iterable[str] | None = None,
        symbols: Sequence[Symbol] | None = None,
    ) -> Self: ...  # fmt: skip

    def update_symbol(self, by, /, symbols=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            symbol_by = _p.SymbolPlan.new(cov.columns, values=symbols)
        else:
            symbol_by = _p.SymbolPlan.from_const(Symbol(by))
        self.base.symbol = symbol_by.map(self._source)
        self._symbol_by = symbol_by
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
    ) -> Self:
        """Add edge to the markers."""
        if color is not None:
            self.update_edge_color(color)
        self.update_width(width)
        self.base.edge.style = LineStyle(style)
        return self

    def move(self, dx: float = 0.0, dy: float = 0.0, autoscale: bool = True) -> Self:
        """Add a constant shift to the layer."""
        _old_data = self.base.data
        self.base.set_data(xdata=_old_data.x + dx, ydata=_old_data.y + dy)
        if autoscale and (canvas := self._canvas_ref()):
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def as_edge_only(
        self,
        width: float = 3.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Self:
        """
        Convert the markers to edge-only mode.

        This method will set the face color to transparent and the edge color to the
        current face color.

        Parameters
        ----------
        width : float, default 3.0
            Width of the edge.
        style : str or LineStyle, default LineStyle.SOLID
            Line style of the edge.
        """
        for layer in self.base.iter_children():
            layer.as_edge_only(width=width, style=style)
        self._edge_color_by = self._color_by
        self._color_by = _p.ColorPlan.from_const("#00000000")
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = dict(self._source.iter_items())
        self.base.with_hover_template(template, extra=extra)
        return self

    def with_regression(
        self,
        *,
        split_by: str | list[str] | None = None,
        color=None,
        width: float | None = None,
        style: str | LineStyle | None = None,
    ) -> _lg.MainAndOtherLayers[Self, DFRegPlot[_DF]]:
        """
        Add a regression line to the markers.

        Parameters
        ----------
        split_by : str, list[str], optional
            Column names by which data will be split to draw regression lines.
        color : color-like, optional
            Constant color to draw regression.
        width : float, optional
            Width of the regression lines.
        style : str or LineStyle, optional
            Line style of the regression lines.
        """
        df = self._source
        splitby_default: list[str] = []
        colors_ref = self.base.face.color
        symbols_ref = self.base.symbol
        if self._edge_color_by.is_not_const() and isinstance(
            self._edge_color_by, _p.ColorPlan
        ):
            splitby_default.extend(self._edge_color_by.by)
            colors_ref = self.base.edge.color
        if self._color_by.is_not_const() and isinstance(self._color_by, _p.ColorPlan):
            splitby_default.extend(self._color_by.by)
            colors_ref = self.base.face.color
        if self._symbol_by.is_not_const():
            splitby_default.extend(self._symbol_by.by)

        # normalize split_by
        if split_by is None:
            split_by = splitby_default
        elif isinstance(split_by, str):
            if split_by not in splitby_default:
                raise ValueError(f"`split_by` must be one of {split_by!r}.")
            split_by = [split_by]
        else:
            split_by = list(split_by)
            for sb in split_by:
                if sb not in splitby_default:
                    raise ValueError(f"`split_by` must be one of {split_by!r}.")

        xs = []
        ys = []
        colors = []
        symbols = []
        data = self.base.data
        for sl, _ in df.group_by(split_by):
            arr_indices = np.all(
                np.stack([df[c] == v for c, v in zip(split_by, sl)], axis=0),
                axis=0,
            )
            xs.append(data.x[arr_indices])
            ys.append(data.y[arr_indices])
            colors.append(colors_ref[arr_indices][0][:3])
            symbols.append(symbols_ref[arr_indices][0])

        if color is not None:
            colors = [Color(color)] * len(xs)
        if style is not None:
            styles = [LineStyle(style)] * len(xs)
        else:
            # make symbol to linestyle mapping
            linestyles = _p.StylePlan._default_values()
            _mapping = {
                sym: linestyles[i % len(linestyles)]
                for i, sym in enumerate(OrderedSet(symbols))
            }
            styles = [LineStyle(_mapping[sym]) for sym in symbols]
        regplot = DFRegPlot.from_arrays(
            self._source,
            xs,
            ys,
            colors=colors,
            width=width,
            styles=styles,
            backend=self._base_layer._backend_name,
            name=f"{self.name}:regression",
        )
        return _lg.MainAndOtherLayers(self, regplot)

    def _as_legend_item(self) -> LegendItem:
        items = []
        color_default = Color("transparent")
        symbol_default = Symbol.CIRCLE
        size_default = 8
        edge_info = self._base_layer.edge._as_legend_info()
        if self._symbol_by.is_const():
            symbol_default = self._symbol_by.get_const_value()
        if self._size_by.is_const():
            size_default = self._size_by.get_const_value()
        if self._color_by.is_const():
            color_default = self._color_by.map(self._source)[0]
        elif isinstance(self._color_by, _p.ColorPlan):
            color_entries = self._color_by.to_entries(self._source)
            items.append((", ".join(self._color_by.by), _legend.TitleItem()))
            for label, color in color_entries:
                item = (
                    label,
                    _legend.MarkersLegendItem(
                        symbol_default, size_default, _legend.FaceInfo(color), edge_info
                    ),
                )
                items.append(item)
        elif isinstance(self._color_by, _p.MapPlan):
            if _map := self._color_by.get_colormap_map():
                items.append((", ".join(self._color_by._on), _legend.TitleItem()))
                for color, value in _map.create_samples(self._source):
                    item = (
                        _safe_str(value),
                        _legend.MarkersLegendItem(
                            symbol_default,
                            size_default,
                            _legend.FaceInfo(color),
                            edge_info,
                        ),
                    )
                    items.append(item)

        if self._hatch_by.is_not_const():
            hatch_entries = self._hatch_by.to_entries(self._source)
            items.append((", ".join(self._hatch_by.by), _legend.TitleItem()))
            for label, hatch in hatch_entries:
                item = (
                    label,
                    _legend.MarkersLegendItem(
                        symbol_default,
                        size_default,
                        _legend.FaceInfo(color_default, hatch),
                        edge_info,
                    ),
                )
                items.append(item)
        if self._symbol_by.is_not_const():
            symbol_entries = self._symbol_by.to_entries(self._source)
            items.append((", ".join(self._symbol_by.by), _legend.TitleItem()))
            for label, symbol in symbol_entries:
                item = (
                    label,
                    _legend.MarkersLegendItem(
                        symbol, size_default, _legend.FaceInfo(color_default), edge_info
                    ),
                )
                items.append(item)
        if self._size_by.is_not_const():
            if _map := self._size_by.get_ranged_map():
                items.append((", ".join(self._size_by._on), _legend.TitleItem()))
                for size, value in _map.create_samples(self._source):
                    item = (
                        _safe_str(value),
                        _legend.MarkersLegendItem(
                            symbol_default,
                            size,
                            _legend.FaceInfo(color_default),
                            edge_info,
                        ),
                    )
                    items.append(item)
        return _legend.LegendItemCollection(items)


class DFMarkerGroups(DFMarkers):
    def __init__(self, *args, orient: Orientation = Orientation.VERTICAL, **kwargs):
        super().__init__(*args, **kwargs)
        self._orient = Orientation.parse(orient)

    @classmethod
    def from_jitters(
        cls,
        *args,
        orient: str | Orientation = Orientation.VERTICAL,
        **kwargs,
    ) -> Self:
        self = super().from_jitters(*args, **kwargs)
        self._orient = Orientation.parse(orient)
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        self = super().from_dict(d, backend)
        self._orient = Orientation.parse(d["orient"])
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "orient": self._orient.value,
        }

    @property
    def orient(self) -> Orientation:
        """Orientation of the plot."""
        return self._orient

    def move(self, shift: float = 0.0, autoscale: bool = True) -> Self:
        """Add a constant shift to the layer."""
        _old_data = self._base_layer.data
        if self.orient.is_vertical:
            self._base_layer.set_data(xdata=_old_data.x + shift)
        else:
            self._base_layer.set_data(ydata=_old_data.y + shift)
        if autoscale and (canvas := self._canvas_ref()):
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self


class DFBars(
    _shared.DataFrameLayerWrapper[_l.Bars[_mixin.MultiFace, _mixin.MultiEdge], _DF],
    _MarkerLikeMixin,
    Generic[_DF],
):
    def __init__(
        self,
        base: _l.Bars[_mixin.MultiFace, _mixin.MultiEdge],
        source: DataFrameWrapper[_DF],
        stackby: tuple[str, ...],
        splitby: tuple[str, ...],
        color_by: _p.ColorPlan | _p.ColormapPlan,
        edge_color_by: _p.ColorPlan | _p.ColormapPlan,
        hatch_by: _p.HatchPlan,
        style_by: _p.StylePlan,
        width_by: _p.WidthPlan,
    ):
        self._color_by = color_by
        self._edge_color_by = edge_color_by
        self._hatch_by = hatch_by
        self._style_by = style_by
        self._splitby = splitby
        self._stackby = stackby
        self._width_by = width_by
        super().__init__(base, source)

    @classmethod
    def from_arrays(
        cls,
        source: DataFrameWrapper[_DF],
        x: ArrayLike1D,
        y: ArrayLike1D,
        bottom: ArrayLike1D | None = None,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        stackby: tuple[str, ...] = (),
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ) -> Self:
        splitby = _shared.join_columns(color, hatch, stackby, source=source)
        base = _l.Bars(
            x, y, bottom=bottom, name=name, orient=orient, extent=extent,
            backend=backend
        ).with_face_multi().with_edge_multi()  # fmt: skip
        self = cls(
            base, source, stackby=stackby, splitby=splitby,
            color_by=_p.ColorPlan.default(), edge_color_by=_p.ColorPlan.default(),
            hatch_by=_p.HatchPlan.default(), style_by=_p.StylePlan.default(),
            width_by=_p.WidthPlan.default(),
        )  # fmt: skip
        if color is not None:
            self.update_color(color)
        if hatch is not None:
            self.update_hatch(hatch)
        self.with_hover_template(default_template(source.iter_items()))
        return self

    @property
    def orient(self) -> Orientation:
        """Orientation of the plot."""
        return self.base.orient

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        x: str | _jitter.JitterBase,
        y: str | _jitter.JitterBase,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        extent: float = 0.8,
        orient: Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFBars[_DF]:
        if isinstance(x, _jitter.JitterBase):
            xj = x
        else:
            xj = _jitter.IdentityJitter(x)
        if isinstance(y, _jitter.JitterBase):
            yj = y
        else:
            yj = _jitter.IdentityJitter(y)
        x0 = xj.map(df)
        y0 = yj.map(df)
        return DFBars.from_arrays(
            df, x0, y0, name=name, color=color, hatch=hatch, extent=extent,
            orient=orient, backend=backend,
        )  # fmt: skip

    @classmethod
    def from_table_stacked(
        cls,
        df: DataFrameWrapper[_DF],
        x: str | _jitter.JitterBase,
        y: str | _jitter.JitterBase,
        stackby: str | tuple[str, ...] | None,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        extent: float = 0.8,
        orient: Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFBars[_DF]:
        if stackby is None:
            stackby = _shared.join_columns(color, hatch, source=df)
        x0, y0, b0 = _shared.resolve_stacking(df, x, y, stackby)
        return DFBars.from_arrays(
            df, x0, y0, b0, color=color, hatch=hatch, stackby=stackby, name=name,
            extent=extent, orient=orient, backend=backend,
        )  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a DFViolinPlot from a dictionary."""
        base, source = d["base"], d["source"]
        if isinstance(base, dict):
            base = construct_layer(base, backend=backend)
        if isinstance(source, dict):
            source = from_dict(source)
        return cls(
            base,
            source,
            stackby=tuple(d["stack_by"]),
            splitby=tuple(d["split_by"]),
            color_by=_p.ColorPlan.from_dict_or_plan(d["color_by"]),
            edge_color_by=_p.ColorPlan.from_dict_or_plan(d["edge_color_by"]),
            hatch_by=_p.HatchPlan.from_dict_or_plan(d["hatch_by"]),
            style_by=_p.StylePlan.from_dict_or_plan(d["style_by"]),
            width_by=_p.WidthPlan.from_dict_or_plan(d["width_by"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "base": self._base_layer.to_dict(),
            "source": self._source,
            "stack_by": self._stackby,
            "split_by": self._splitby,
            "color_by": self._color_by,
            "edge_color_by": self._edge_color_by,
            "hatch_by": self._hatch_by,
            "style_by": self._style_by,
            "width_by": self._width_by,
        }

    def _apply_color(self, color):
        self.base.face.color = np.asarray(color, dtype=np.float32)

    def _apply_width(self, width):
        self._base_layer.with_edge_multi(color=_void, width=width, style=_void)

    def _apply_alpha(self, alpha):
        self.base.face.alpha = alpha
        self.base.edge.alpha = alpha

    def _apply_style(self, style):
        self._base_layer.with_edge_multi(color=_void, width=_void, style=style)

    def update_hatch(self, by: str | Iterable[str], choices=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot hatch by a column other than {self._splitby}")
            hatch_by = _p.HatchPlan.new(cov.columns, values=choices)
        else:
            hatch_by = _p.HatchPlan.from_const(Hatch(cov.value))
        self._base_layer.face.hatch = hatch_by.map(self._source)
        self._hatch_by = hatch_by
        return self

    def as_edge_only(
        self,
        width: float = 3.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Self:
        """
        Convert the bars to edge-only mode.

        This method will set the face color to transparent and the edge color to the
        current face color.

        Parameters
        ----------
        width : float, default 3.0
            Width of the edge.
        style : str or LineStyle, default LineStyle.SOLID
            Line style of the edge.
        """
        self.base.as_edge_only(width=width, style=style)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = dict(self._source.iter_items())
        self.base.with_hover_template(template, extra=extra)
        return self

    def _as_legend_item(self) -> LegendItem:
        items = []
        color_default = theme.get_theme().background_color
        edge_info = self._base_layer.edge._as_legend_info()
        if self._color_by.is_const():
            color_default = self._color_by.map(self._source)[0]
        else:
            color_entries = self._color_by.to_entries(self._source)
            items.append((", ".join(self._color_by.by), _legend.TitleItem()))
            for label, color in color_entries:
                items.append(
                    (label, _legend.BarLegendItem(_legend.FaceInfo(color), edge_info))
                )

        if self._hatch_by.is_not_const():
            hatch_entries = self._hatch_by.to_entries(self._source)
            items.append((", ".join(self._hatch_by.by), _legend.TitleItem()))
            for label, hatch in hatch_entries:
                item = (
                    label,
                    _legend.BarLegendItem(
                        _legend.FaceInfo(color_default, hatch), edge_info
                    ),
                )
                items.append(item)
        return _legend.LegendItemCollection(items)


class DFRug(_shared.DataFrameLayerWrapper[_l.Rug, _DF], _MarkerLikeMixin, Generic[_DF]):
    def __init__(
        self,
        base: _l.Rug,
        source: DataFrameWrapper[_DF],
        color_by: _p.ColorPlan | _p.ColormapPlan,
        width_by: _p.WidthPlan,
        style_by: _p.StylePlan,
        scale_by: _p.WidthPlan,
    ):
        self._color_by = color_by
        self._width_by = width_by
        self._style_by = style_by
        self._scale_by = scale_by
        super().__init__(base, source)

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFRug[_DF]:
        ori = Orientation.parse(orient)
        base = _l.Rug(
            df[value], name=name, orient=ori, low=low, high=high, backend=backend,
        )  # fmt: skip
        self = cls(
            base, df,
            color_by=_p.ColorPlan.default(), width_by=_p.WidthPlan.default(),
            style_by=_p.StylePlan.default(), scale_by=_p.WidthPlan.default(),
        )  # fmt: skip
        if color is not None:
            self._update_color_or_colormap(color)
        if isinstance(width, str):
            self.update_width(width)
        elif is_real_number(width):
            self.base.width = width
        if style is not None:
            self.update_style(style)
        self.with_hover_template(default_template(df.iter_items()))
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a DFViolinPlot from a dictionary."""
        base, source = d["base"], d["source"]
        if isinstance(base, dict):
            base = construct_layer(base, backend=backend)
        if isinstance(source, dict):
            source = from_dict(source)
        return cls(
            base,
            source,
            color_by=_p.ColorPlan.from_dict_or_plan(d["color_by"]),
            width_by=_p.WidthPlan.from_dict_or_plan(d["width_by"]),
            style_by=_p.StylePlan.from_dict_or_plan(d["style_by"]),
            scale_by=_p.WidthPlan.from_dict_or_plan(d["scale_by"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "base": self._base_layer.to_dict(),
            "source": self._source,
            "color_by": self._color_by,
            "width_by": self._width_by,
            "style_by": self._style_by,
            "scale_by": self._scale_by,
        }

    @property
    def orient(self) -> Orientation:
        """Orientation of the rugs"""
        return self.base.orient

    def _apply_color(self, color):
        self.base.color = np.asarray(color, dtype=np.float32)

    def _apply_alpha(self, alpha):
        self.base.alpha = alpha

    def _apply_width(self, width):
        self.base.width = width

    def _apply_style(self, style):
        self.base.style = style

    def update_length(
        self,
        lengths: float | NDArray[np.number],
        *,
        offset: float | None = None,
        align: str = "low",
    ) -> Self:
        """
        Update the length of the rug lines.

        Parameters
        ----------
        lengths : float or array-like
            Length of the rug lines. If a scalar, all the lines have the same length.
            If an array, each line has a different length.
        offset : float, optional
            Offset of the lines. If not given, the mean of the lower and upper bounds is
            used.
        align : {'low', 'high', 'center'}, optional
            How to align the rug lines around the offset. This parameter is defined as
            follows.

            ```
               "low"     "high"    "center"
              ──┴─┴──   ──┬─┬──    ──┼─┼──
            ```
        """
        self.base.update_length(lengths=lengths, offset=offset, align=align)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = dict(self._source.iter_items())
        self.base.with_hover_template(template, extra=extra)
        return self


class DFRugGroups(DFRug[_DF]):
    def __init__(
        self,
        base: _l.Rug,
        source: DataFrameWrapper[_DF],
        value: str,
        splitby: tuple[str, ...],
        color_by: _p.ColorPlan | _p.ColormapPlan,
        width_by: _p.WidthPlan,
        style_by: _p.StylePlan,
        scale_by: _p.WidthPlan,
    ):
        super().__init__(base, source, color_by, width_by, style_by, scale_by)
        self._splitby = splitby
        self._value = value

    @property
    def orient(self) -> Orientation:
        """Orientation of the plot."""
        return self.base.orient.transpose()

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a DFRugGroups from a dictionary."""
        base, source = d["base"], d["source"]
        if isinstance(base, dict):
            base = construct_layer(base, backend=backend)
        if isinstance(source, dict):
            source = from_dict(source)
        return cls(
            base,
            source,
            value=d["value"],
            splitby=tuple(d["split_by"]),
            color_by=_p.ColorPlan.from_dict_or_plan(d["color_by"]),
            width_by=_p.WidthPlan.from_dict_or_plan(d["width_by"]),
            style_by=_p.StylePlan.from_dict_or_plan(d["style_by"]),
            scale_by=_p.WidthPlan.from_dict_or_plan(d["scale_by"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "value": self._value,
            "split_by": self._splitby,
        }

    def move(self, shift: float = 0.0, autoscale: bool = True) -> Self:
        """Add a constant shift to the layer."""
        _old = self._base_layer.data_full
        self._base_layer.data_full = _old.x, _old.y0 + shift, _old.y1 + shift
        if autoscale and (canvas := self._canvas_ref()):
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        jitter: _jitter.CategoricalJitter,
        value: str,
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        extent: float = 0.8,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFRugGroups[_DF]:
        # orientation of the rugs are always opposite to the orientation of the plot
        ori = Orientation.parse(orient).transpose()
        x = jitter.map(df)
        dx = extent / 2
        base = _l.Rug(
            df[value], orient=ori, low=x - dx, high=x + dx, name=name, backend=backend,
        )  # fmt: skip
        self = cls(
            base, df, value, jitter.by,
            color_by=_p.ColorPlan.default(), width_by=_p.WidthPlan.default(),
            style_by=_p.StylePlan.default(), scale_by=_p.WidthPlan.default(),
        )  # fmt: skip
        if color is not None:
            self._update_color_or_colormap(color)
        if isinstance(width, str):
            self.update_width(width)
        elif is_real_number(width):
            self.base.width = width
        if style is not None:
            self.update_style(style)
        self.with_hover_template(default_template(df.iter_items()))
        return self

    def scale_by_density(
        self,
        *,
        align: str = "center",
        band_width: KdeBandWidthType = "scott",
    ) -> Self:
        """
        Set the height of the lines by density.

        Parameters
        ----------
        align : {'low', 'high', 'center'}, optional
            How to align the rug lines around the offset. This parameter is defined as
            follows.

            ```
               "low"     "high"    "center"
              ──┴─┴──   ──┬─┬──    ──┼─┼──
            ```
        band_width : float, "scott" or "silverman", optional
            Method to calculate the estimator bandwidth.
        """
        from whitecanvas.utils.kde import gaussian_kde

        data_full = self.base.data_full
        densities: list[np.ndarray] = []
        slices: list[np.ndarray] = []
        offsets: list[float] = []
        for _sl, sub in self._source.group_by(self._splitby):
            arr = sub[self._value]
            density = gaussian_kde(arr, band_width)(arr)
            densities.append(density)
            _ar_bool = np.column_stack(
                [self._source[col] == s for col, s in zip(self._splitby, _sl)]
            ).all(axis=1)
            slices.append(_ar_bool)
            offsets.append(data_full.ycenter[_ar_bool].mean())
        density_max = max(d.max() for d in densities)
        diff = np.unique(self.base.high - self.base.low)
        if diff.size == 0:
            extent = 0.8
        else:
            extent = diff.max()
        normed = [d / density_max * extent / 2 for d in densities]

        # sort densities
        normed_sorted = np.empty(self._source.shape[0], dtype=np.float32)
        off_sorted = np.empty_like(normed_sorted)
        for _sl, _norm, _off in zip(slices, normed, offsets):
            normed_sorted[_sl] = _norm
            off_sorted[_sl] = _off

        if align == "low":
            y0 = off_sorted
            y1 = off_sorted + normed_sorted
        elif align == "high":
            y0 = off_sorted - normed_sorted
            y1 = off_sorted
        elif align == "center":
            y0 = off_sorted - normed_sorted
            y1 = off_sorted + normed_sorted
        else:
            raise ValueError(
                f"`align` must be 'low', 'high', or 'center', got {align!r}."
            )
        self.base.data_full = data_full.x, y0, y1
        return self


def default_template(it: Iterable[tuple[str, np.ndarray]], max_rows: int = 10) -> str:
    """
    Default template string for markers

    This template can only be used for those plot that has one tooltip for each data
    point, which includes markers, bars and rugs.
    """
    fmt_list = list[str]()
    for ikey, (key, value) in enumerate(it):
        if not key:
            continue
        if ikey >= max_rows:
            break
        if value.dtype.kind == "f":
            fmt_list.append(f"{key}: {{{key}:.4g}}")
        else:
            fmt_list.append(f"{key}: {{{key}!r}}")
    return "\n".join(fmt_list)


def _safe_str(value) -> str:
    if is_real_number(value):
        return f"{value:.3g}"
    else:
        return str(value)
