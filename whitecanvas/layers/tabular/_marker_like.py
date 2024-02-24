from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    Iterable,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from cmap import Color, Colormap
from numpy.typing import NDArray

from whitecanvas import layers as _l
from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _legend, _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.layers._legend import LegendItem
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper
from whitecanvas.types import (
    ColormapType,
    ColorType,
    Hatch,
    KdeBandWidthType,
    LineStyle,
    Orientation,
    Symbol,
    _Void,
)
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

_DF = TypeVar("_DF")
_Cols = Union[str, "tuple[str, ...]"]
_void = _Void()


class _MarkerLikeMixin:
    _source: DataFrameWrapper[_DF]

    @overload
    def update_color(self, value: ColorType) -> Self:
        ...

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

    @overload
    def update_width(self, value: float) -> Self:
        ...

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
    def update_style(self, value: ColorType) -> Self:
        ...

    @overload
    def update_style(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

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

    def _apply_color(self, color):
        """Set color array to the layer."""
        raise NotImplementedError

    def _apply_width(self, width):
        """Set width array to the layer."""
        raise NotImplementedError

    def _apply_style(self, style):
        """Set style array to the layer."""
        raise NotImplementedError


class DFMarkers(
    _shared.DataFrameLayerWrapper[_lg.MarkerCollection, _DF], _MarkerLikeMixin
):
    def __init__(
        self,
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
    ):
        self._x = x
        self._y = y
        self._color_by: _p.ColorPlan | _p.ColormapPlan = _p.ColorPlan.default()
        self._edge_color_by: _p.ColorPlan | _p.ColormapPlan = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._size_by = _p.SizePlan.default()
        self._symbol_by = _p.SymbolPlan.default()
        self._width_by = _p.WidthPlan.default()

        base = _lg.MarkerCollection.from_arrays(
            x.map(source), y.map(source), name=name, backend=backend
        )

        super().__init__(base, source)
        if color is not None:
            self.update_color(color)
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

    def _apply_color(self, color):
        self.base.face.color = color

    def _apply_width(self, width):
        self._base_layer.with_edge(color=_void, width=width, style=_void)

    def _apply_style(self, style):
        self._base_layer.with_edge(color=_void, width=_void, style=style)

    def update_edge_color(self, by: str | Iterable[str], palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        colors = color_by.map(self._source)
        self._base_layer.edge.color = colors
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
        self._base_layer.edge.color = colors
        self._edge_color_by = color_by
        return self

    def update_hatch(self, by: str | Iterable[str], palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            hatch_by = _p.HatchPlan.new(cov.columns, values=palette)
        else:
            hatch_by = _p.HatchPlan.from_const(Hatch(cov.value))
        hatches = hatch_by.map(self._source)
        self._base_layer.face.hatch = hatches
        self._hatch_by = hatch_by
        return self

    @overload
    def update_size(self, value: float) -> Self:
        ...

    @overload
    def update_size(
        self,
        by: str,
        map_from: tuple[float, float] | None = None,
        map_to: tuple[float, float] = (3, 15),
    ) -> Self:
        ...

    def update_size(self, by, /, map_from=None, map_to=(3, 15)):
        """Set the size of the markers."""
        if isinstance(by, str):
            size_by = _p.SizePlan.from_range(by, range=map_to, domain=map_from)
        else:
            size_by = _p.SizePlan.from_const(float(by))
        self._base_layer.size = size_by.map(self._source)
        self._size_by = size_by
        return self

    @overload
    def update_symbol(self, value: str | Symbol) -> Self:
        ...

    @overload
    def update_symbol(
        self, by: str | Iterable[str] | None = None, symbols=None
    ) -> Self:
        ...

    def update_symbol(self, by, /, symbols=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            symbol_by = _p.SymbolPlan.new(cov.columns, values=symbols)
        else:
            symbol_by = _p.SymbolPlan.from_const(Symbol(by))
        self._base_layer.symbol = symbol_by.map(self._source)
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
        self._base_layer.edge.style = LineStyle(style)
        return self

    def move(self, dx: float = 0.0, dy: float = 0.0, autoscale: bool = True) -> Self:
        """Add a constant shift to the layer."""
        _old_data = self._base_layer.data
        self._base_layer.set_data(xdata=_old_data.x + dx, ydata=_old_data.y + dy)
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
        source: DataFrameWrapper[_DF],
        x,
        y,
        bottom=None,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        stackby: tuple[str, ...] = (),
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        splitby = _shared.join_columns(color, hatch, stackby, source=source)
        self._color_by = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._style_by = _p.StylePlan.default()
        self._splitby = splitby
        self._stackby = stackby

        base = _l.Bars(
            x, y, bottom=bottom, name=name, orient=orient, extent=extent,
            backend=backend
        ).with_face_multi()  # fmt: skip
        super().__init__(base, source)
        if color is not None:
            self.update_color(color)
        if hatch is not None:
            self.update_hatch(hatch)
        self.with_hover_template(default_template(source.iter_items()))

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
        return DFBars(
            df, x0, y0, name=name, color=color, hatch=hatch, extent=extent,
            orient=orient, backend=backend,
        )  # fmt: skip

    @classmethod
    def from_table_stacked(
        cls,
        df: DataFrameWrapper[_DF],
        x: str | _jitter.JitterBase,
        y: str | _jitter.JitterBase,
        stackby: str | tuple[str, ...],
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
        # pre-calculate all the possible xs
        all_x = xj.map(df)

        def _hash_rule(x: float) -> int:
            return int(round(x * 1000))

        ycumsum = {_hash_rule(_x): 0.0 for _x in all_x}
        x0 = list[NDArray[np.number]]()
        y0 = list[NDArray[np.number]]()
        b0 = list[NDArray[np.number]]()
        for _, sub in df.group_by(stackby):
            this_x = xj.map(sub)
            this_h = yj.map(sub)
            bottom = []
            for _x, _h in zip(this_x, this_h):
                _x_hash = _hash_rule(_x)
                dy = ycumsum[_x_hash]
                bottom.append(dy)
                ycumsum[_x_hash] += _h
            x0.append(this_x)
            y0.append(this_h)
            b0.append(bottom)
        x0 = np.concatenate(x0)
        y0 = np.concatenate(y0)
        b0 = np.concatenate(b0)
        return DFBars(
            df, x0, y0, b0, color=color, hatch=hatch, stackby=stackby, name=name,
            extent=extent, orient=orient, backend=backend,
        )  # fmt: skip

    def _apply_color(self, color):
        self.base.face.color = color

    def _apply_width(self, width):
        self._base_layer.with_edge(color=_void, width=width, style=_void)

    def _apply_style(self, style):
        self._base_layer.with_edge(color=_void, width=_void, style=style)

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
        source: DataFrameWrapper[_DF],
        base: _l.Rug,
        color: _Cols | None = None,
        width: str | float | None = None,
        style: str | Iterable[str] | None = None,
    ):
        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._scale_by = _p.WidthPlan.default()
        super().__init__(base, source)
        if color is not None:
            self.update_color(color)
        if isinstance(width, str):
            self.update_width(width)
        elif is_real_number(width):
            self.base.width = width
        if style is not None:
            self.update_style(style)
        self.with_hover_template(default_template(source.iter_items()))

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
        return cls(df, base, color=color, width=width, style=style)

    @property
    def orient(self) -> Orientation:
        """Orientation of the rugs"""
        return self.base.orient

    def _apply_color(self, color):
        self.base.color = color

    def _apply_width(self, width):
        self.base.width = width

    def _apply_style(self, style):
        self.base.style = style

    # def update_scale(self, by: str | float, align: str = "low") -> Self:
    #     ...

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
        source: DataFrameWrapper[_DF],
        base: _l.Rug,
        value: str,
        splitby: tuple[str, ...],
        color: str | tuple[str, ...] | None = None,
        width: str | None = None,
        style: str | tuple[str, ...] | None = None,
        extent: float = 0.8,
    ):
        super().__init__(source, base, color=color, width=width, style=style)
        self._splitby = splitby
        self._value = value
        self._extent = extent

    @property
    def orient(self) -> Orientation:
        """Orientation of the plot."""
        return self.base.orient.transpose()

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
        return cls(
            df, base, value, jitter.by, color=color, width=width, style=style,
            extent=extent,
        )  # fmt: skip

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
        normed = [d / density_max * self._extent / 2 for d in densities]

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
