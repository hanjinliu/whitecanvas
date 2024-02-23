"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, Literal, Sequence, TypeVar

import numpy as np
from cmap import Color

from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import Layer, _legend, _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, parse
from whitecanvas.types import (
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    Symbol,
    _Void,
)
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas.dataframe._base import CatIterator
    from whitecanvas.layers.tabular import DFMarkerGroups, DFRugGroups

    _FE = _mixin.AbstractFaceEdgeMixin[_mixin.FaceNamespace, _mixin.EdgeNamespace]

_DF = TypeVar("_DF")
_L = TypeVar("_L", bound=Layer)
_void = _Void()


def _norm_color_hatch(
    color,
    hatch,
    df: DataFrameWrapper[_DF],
) -> tuple[_p.ColorPlan, _p.HatchPlan]:
    color_cov = _shared.ColumnOrValue(color, df)
    if color_cov.is_column:
        color_by = _p.ColorPlan.from_palette(color_cov.columns)
    elif color_cov.value is not None:
        color_by = _p.ColorPlan.from_const(Color(color_cov.value))
    else:
        color_by = _p.ColorPlan.default()
    hatch_cov = _shared.ColumnOrValue(hatch, df)
    if hatch_cov.is_column:
        hatch_by = _p.HatchPlan.new(hatch_cov.columns)
    elif hatch_cov.value is not None:
        hatch_by = _p.HatchPlan.from_const(Hatch(hatch_cov.value))
    else:
        hatch_by = _p.HatchPlan.default()
    return color_by, hatch_by


class _BoxLikeMixin:
    _source: DataFrameWrapper[_DF]

    def __init__(
        self,
        categories: list[tuple],
        splitby: tuple[str, ...],
        color_by: _p.ColorPlan,
        hatch_by: _p.HatchPlan,
    ):
        self._splitby = splitby
        self._categories = categories
        self._color_by = color_by
        self._hatch_by = hatch_by
        self._get_base().face.color = color_by.generate(self._categories, self._splitby)
        self._get_base().face.hatch = hatch_by.generate(self._categories, self._splitby)
        if not hasattr(self, "with_hover_template"):
            return
        self.with_hover_template("\n".join(f"{k}: {{{k}!r}}" for k in self._splitby))

    def _get_base(self) -> _FE:
        """Just for typing."""
        return self._base_layer

    def _normalize_by_arg(self, by, default: tuple[str, ...]) -> tuple[str, ...]:
        if by is None:
            by = default
        elif isinstance(by, str):
            if by not in self._splitby:
                raise ValueError(
                    f"Cannot color by {by!r} as the plot is not split by this column. "
                    f"Valid columns are: {self._splitby!r}."
                )
            by = (by,)
        else:
            for b in by:
                if not isinstance(b, str):
                    raise TypeError("`by` must be a str or sequence of str.")
                if b not in self._splitby:
                    raise ValueError(
                        f"Cannot color by {by!r} as the plot is not split by this "
                        f"column. Valid columns are: {self._splitby!r}."
                    )
        return by

    def update_color_palette(
        self,
        palette: ColormapType | None = None,
        *,
        alpha: float | None = None,
        cycle_by: str | Sequence[str] | None = None,
    ) -> Self:
        """
        Update the colors by a color palette.

        Parameters
        ----------
        palette : colormap type
            Color palette used to generate colors for each category. A color palette
            can be a list of colors or any types that can be converted into a
            `cmap.Colormap` object.
        alpha : float, optional
            Additional alpha value that will be applied to the palette colors.
        cycle_by : str or sequence of str, optional
            If given, colors will be cycled on this column name(s).
        """
        by = self._normalize_by_arg(cycle_by, self._color_by.by)
        color_by = _p.ColorPlan.from_palette(by, palette=palette)
        colors = color_by.generate(self._categories, self._splitby)
        color_arr = np.stack([c.rgba for c in colors], dtype=np.float32)
        if alpha is not None:
            if is_real_number(alpha) and 0 <= alpha <= 1:
                color_arr[:, 3] = alpha
            else:
                raise TypeError(
                    f"`alpha` must be a scalar value between 0 and 1, got {alpha!r}."
                )
        self._get_base().face.color = color_arr
        self._color_by = color_by
        return self

    def update_hatch_palette(
        self,
        palette: Sequence[str | Hatch],
        *,
        cycle_by: str | Sequence[str] | None = None,
    ) -> Self:
        """
        Update the hatch patterns by a list of hatch values.

        Parameters
        ----------
        palette : sequence of str or Hatch
            Hatch palette used to generate colors for each category.
        """
        by = self._normalize_by_arg(cycle_by, self._hatch_by.by)
        hatch_by = _p.HatchPlan.new(by, values=palette)
        self._get_base().face.hatch = hatch_by.generate(self._categories, self._splitby)
        self._hatch_by = hatch_by
        return self

    def update_const(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: str | Hatch | _Void = _void,
    ) -> Self:
        """
        Update the plot features to the constant values.

        Parameters
        ----------
        color : color-type, optional
            Constant colors used for the plot.
        hatch : str or Hatch, optional
            Constant hatch used for the plot.
        """
        cat = self._categories
        if color is not _void:
            color_by = _p.ColorPlan.from_const(Color(color))
            self._get_base().face.color = color_by.generate(cat, self._splitby)
            self._color_by = color_by
        if hatch is not _void:
            hatch_by = _p.HatchPlan.from_const(Hatch(hatch))
            self._get_base().face.hatch = hatch_by.generate(cat, self._splitby)
            self._hatch_by = hatch_by
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Self:
        """Add edge to the plot with given settings."""
        self._get_base().with_edge(color=color, width=width, style=style, alpha=alpha)
        return self

    def _prep_legend_info(
        self,
    ) -> tuple[list[tuple[str, ColorType]], list[tuple[str, Hatch]]]:
        df = self._simple_dataframe()
        color_entries = self._color_by.to_entries(df)
        hatch_entries = self._hatch_by.to_entries(df)
        return color_entries, hatch_entries

    def _as_legend_item(self) -> _legend.LegendItemCollection:
        colors, hatches = self._prep_legend_info()
        items = []
        color_default = theme.get_theme().background_color
        edge_info = self._get_base().edge._as_legend_info()
        if self._color_by.is_const():
            color_default = Color(self._color_by.values[0])
        else:
            items.append((", ".join(self._color_by.by), _legend.TitleItem()))
            for label, color in colors:
                item = _legend.BarLegendItem(_legend.FaceInfo(color), edge_info)
                items.append((label, item))
        if self._hatch_by.is_not_const():
            items.append((", ".join(self._hatch_by.by), _legend.TitleItem()))
            for label, hatch in hatches:
                item = _legend.BarLegendItem(
                    _legend.FaceInfo(color_default, hatch), edge_info
                )
                items.append((label, item))
        return _legend.LegendItemCollection(items)

    def _simple_dataframe(self) -> DataFrameWrapper[dict]:
        return _shared.list_to_df(self._categories, self._splitby)


class DFViolinPlot(
    _shared.DataFrameLayerWrapper[_lg.ViolinPlot, _DF],
    _BoxLikeMixin,
    Generic[_DF],
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        shape: Literal["both", "left", "right"] = "both",
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _shared.norm_dodge(
            cat.df, cat.offsets, color, hatch, dodge=dodge
        )  # fmt: skip
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _extent = cat.zoom_factor(dodge=dodge) * extent
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat.df)
        base = _lg.ViolinPlot.from_arrays(
            x, arr, name=name, orient=orient, shape=shape, extent=_extent,
            backend=backend,
        )  # fmt: skip
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)
        self._offsets = cat.offsets
        self._value = value
        self._dodge = dodge
        self.with_hover_template("\n".join(f"{k}: {{{k}!r}}" for k in self._splitby))

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._base_layer.orient

    def move(self, shift: float = 0.0) -> Self:
        """Move the layer by the given shift."""
        for layer in self._base_layer:
            _old = layer.data
            layer.set_data(edge_low=_old.y0 + shift, edge_high=_old.y1 + shift)
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def with_hover_text(self, text: str | list[str]) -> Self:
        """Set the hover tooltip text for the layer."""
        self.base.with_hover_text(text)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._categories]
        self.base.with_hover_template(template, extra=extra)
        return self

    def with_rug(
        self,
        *,
        width: float = 1.0,
        color: ColorType | None = None,
    ) -> _lg.MainAndOtherLayers[Self, DFRugGroups[_DF]]:
        """Overlay rug plot on the violins and return the violin layer."""
        from whitecanvas.layers.tabular import DFRugGroups

        canvas = self._canvas_ref()
        if canvas is None:
            raise ValueError("No canvas to add the rug plot.")
        _extent = self.base.extent
        if color is not None:
            colors = Color(color)
        elif self._is_edge_only():
            colors = self._color_by.by
        else:
            colors = Color("#1F1F1F")
        jitter = _jitter.CategoricalJitter(
            self._splitby,
            self._make_cat_iterator().prep_position_map(self._splitby, self._dodge),
        )
        if self.base._shape == "both":
            align = "center"
        elif self.base._shape == "left":
            align = "high"
        else:
            align = "low"
        rug = DFRugGroups.from_table(
            self._source, jitter, self._value, color=colors, width=width,
            extent=_extent, backend=self.base._backend_name,
        ).scale_by_density(align=align)  # fmt: skip
        return _combine_main_and_others(self, rug)

    def with_box(
        self,
        *,
        color: ColorType | None = None,
        median_color: ColorType = "white",
        width: float | None = None,
        extent: float = 0.1,
        capsize: float = 0.0,
    ) -> _lg.MainAndOtherLayers[Self, DFBoxPlot[_DF]]:
        """
        Overlay box plot on the violins and return the violin layer.

        Following the convension of many statistical software, the box plot is colored
        by black if the violin faces are colored, and colored by the edge color
        otherwise. The median line is colored by the given median color.

        Parameters
        ----------
        color : color-type, optional
            Color of the box plot. If not given, it will be colored by "#1F1F1F" if
            the violin faces are colored, and by the edge color of the violin plot
            otherwise.
        median_color : color-type, optional
            Color of the median line of the box plot.
        width : float, optional
            Width of the whiskers of the boxplot. Use violin edge width if not given.
        extent : float, optional
            Relative width of the boxes.
        capsize : float, optional
            Relative size of the caps of the whiskers.
        """

        canvas = self._canvas_ref()
        if canvas is None:
            raise ValueError("No canvas to add the box plot.")
        if color is not None:
            colors = Color(color)
        else:
            if np.all(self.base.edge.width > 0) and np.all(self.base.edge.alpha > 0):
                colors = self.base.edge.color
            else:
                colors = Color("#1F1F1F")
        if width is None:
            width = self.base.edge.width.mean()
        box = DFBoxPlot(
            self._make_cat_iterator(), self._value, name=f"boxplot-of-{self.name}",
            color=None, hatch=Hatch.SOLID, dodge=self._dodge, width=width,
            orient=self.orient, capsize=capsize, extent=extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        box.base.boxes.face.color = colors
        box.base.edge.color = colors
        box.base.medians.color = Color(median_color)
        return _combine_main_and_others(self, box)

    def with_outliers(
        self,
        *,
        color: ColorType | None = None,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float | None = None,
        ratio: float = 1.5,
        extent: float = 0.1,
        seed: int | None = 0,
    ) -> _lg.MainAndOtherLayers[Self, DFMarkerGroups[_DF]]:
        """
        Overlay outliers on the box plot and return the box plot layer.

        Parameters
        ----------
        color : color-type, optional
            Color of the outliers. To make sure the outliers are easily visible, face
            color will always be transparent. If a constant color is given, all the
            edges will be colored by the same color. By default, the edge colors are
            the same as the edge colors of the box plot.
        symbol : str or Symbol, optional
            Symbol of the outlier markers.
        size : float, optional
            Size of the outlier markers. If not given, it will be set to the theme
            default.
        ratio : float, optional
            Ratio of the interquartile range (IQR) to determine the outliers. Data
            points outside of the range [Q1 - ratio * IQR, Q3 + ratio * IQR] will be
            considered as outliers.
        extent : float, optional
            Relative width of the jitter range (same effect as the `extent` argument of
            the `add_stripplot` method).
        seed : int, optional
            Random seed for the jitter (same effect as the `seed` argument of the
            `add_stripplot` method).
        """
        from whitecanvas.canvas.dataframe._base import CatIterator
        from whitecanvas.layers.tabular import DFMarkerGroups

        canvas = self._canvas_ref()
        size = theme._default("markers.size", size)
        if canvas is None:
            raise ValueError("No canvas to add the outliers.")

        is_edge_only = self._is_edge_only()

        # category iterator is used to calculate positions and indices
        _cat_self = CatIterator(self._source, offsets=self._offsets)
        _pos_map = _cat_self.prep_position_map(self._splitby, self._dodge)
        _extent = _cat_self.zoom_factor(self._dodge) * extent

        # calculate outliers and update the separators
        df_outliers = {c: [] for c in (*self._splitby, self._value)}
        colors = []
        for idx_cat, (sl, sub) in enumerate(self._source.group_by(self._splitby)):
            arr = sub[self._value]
            q1, q3 = np.quantile(arr, [0.25, 0.75])
            iqr = q3 - q1  # interquartile range
            low = q1 - ratio * iqr  # lower bound of inliers
            high = q3 + ratio * iqr  # upper bound of inliers
            outliers = arr[(arr < low) | (arr > high)]
            for _cat, _s in zip(sl, self._splitby):
                df_outliers[_s].extend([_cat] * outliers.size)
            df_outliers[self._value].extend(outliers)
            if is_edge_only:
                _this_color = self.base.edge.color[idx_cat]
            else:
                _this_color = self.base.face.color[idx_cat]
            colors.extend([_this_color] * outliers.size)

        df_outliers = parse(df_outliers)
        xj = _jitter.UniformJitter(self._splitby, _pos_map, extent=_extent, seed=seed)
        yj = _jitter.IdentityJitter(self._value).check(df_outliers)
        new = DFMarkerGroups(
            df_outliers, xj, yj, name=f"outliers-of-{self.name}", color=Color("black"),
            orient=self.orient, symbol=symbol, size=size, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is None:
            if is_edge_only:  # edge only
                new._apply_color(np.stack(colors, axis=0, dtype=np.float32))
            new.as_edge_only(width=self.base.edge.width.mean())
        return _combine_main_and_others(self, new)

    def with_strip(
        self,
        *,
        color: ColorType | None = None,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: str | None = None,
        extent: float = 0.2,
        seed: int | None = 0,
    ) -> _lg.MainAndOtherLayers[Self, DFMarkerGroups[_DF]]:
        """
        Overlay strip plot on the violins.

        Parameters
        ----------
        color : color-type, optional
            Color of the strip plot. If not given, it will be colored by the violin
            face color.
        symbol : str or Symbol, optional
            Symbol of the strip plot markers.
        size : float, optional
            Size of the strip plot markers. If not given, it will be set to the theme
            default.
        extent : float, optional
            Relative width of the jitter range.
        seed : int, optional
            Random seed for the jitter.
        """
        from whitecanvas.canvas.dataframe._base import CatIterator
        from whitecanvas.layers.tabular import DFMarkerGroups

        canvas = self._canvas_ref()
        size = theme._default("markers.size", size)
        if canvas is None:
            raise ValueError("No canvas to add the outliers.")

        if color is None:
            color = self._color_by.by
        else:
            color = Color(color)

        # category iterator is used to calculate positions and indices
        _cat_self = CatIterator(self._source, offsets=self._offsets)
        _pos_map = _cat_self.prep_position_map(self._splitby, self._dodge)
        _extent = _cat_self.zoom_factor(self._dodge) * extent
        df = self._source
        xj = _jitter.UniformJitter(self._splitby, _pos_map, extent=_extent, seed=seed)
        yj = _jitter.IdentityJitter(self._value).check(df)
        new = DFMarkerGroups(
            df, xj, yj, name=f"outliers-of-{self.name}", color=color,
            orient=self.orient, symbol=symbol, size=size, backend=canvas._get_backend(),
        )  # fmt: skip
        if self._is_edge_only():
            new.as_edge_only(width=self.base.edge.width.mean())
        return _combine_main_and_others(self, new)

    def with_swarm(
        self,
        *,
        color: ColorType | None = None,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: str | None = None,
        extent: float = 0.8,
        sort: bool = False,
    ) -> _lg.MainAndOtherLayers[Self, DFMarkerGroups[_DF]]:
        """
        Overlay swarm plot on the violins.

        Parameters
        ----------
        color : color-type, optional
            Color of the strip plot. If not given, it will be colored by the violin
            face color.
        symbol : str or Symbol, optional
            Symbol of the strip plot markers.
        size : float, optional
            Size of the strip plot markers. If not given, it will be set to the theme
            default.
        extent : float, optional
            Relative width of the jitter range.
        sort : bool, default False
            If True, the markers will be sorted by the value.
        """
        from whitecanvas.canvas.dataframe._base import CatIterator
        from whitecanvas.layers.tabular import DFMarkerGroups

        canvas = self._canvas_ref()
        size = theme._default("markers.size", size)
        if canvas is None:
            raise ValueError("No canvas to add the outliers.")

        if color is None:
            color = self._color_by.by
        else:
            color = Color(color)

        # category iterator is used to calculate positions and indices
        _cat_self = CatIterator(self._source, offsets=self._offsets)
        _pos_map = _cat_self.prep_position_map(self._splitby, self._dodge)
        _extent = _cat_self.zoom_factor(self._dodge) * extent
        df = self._source

        if sort:
            df = df.sort(self._value)
        lims = df[self._value].min(), df[self._value].max()
        xj = _jitter.SwarmJitter(
            self._splitby, _pos_map, self._value, lims, extent=_extent
        )
        yj = _jitter.IdentityJitter(self._value).check(df)
        new = DFMarkerGroups(
            df, xj, yj, name=f"outliers-of-{self.name}", color=color,
            orient=self.orient, symbol=symbol, size=size, backend=canvas._get_backend(),
        )  # fmt: skip
        if self._is_edge_only():
            new.as_edge_only(width=self.base.edge.width.mean())
        return _combine_main_and_others(self, new)

    def as_edge_only(
        self,
        width: float = 3.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Self:
        """
        Replace the violin edge color with the face color and delete the face color.

        Parameters
        ----------
        width : float, optional
            Width of the edge.
        style : str or LineStyle, optional
            Style of the edge.
        """
        self.base.with_edge(color=self.base.face.color, width=width, style=style)
        self.base.face.update(alpha=0.0)
        return self

    def _as_legend_item(self) -> _legend.LegendItemCollection:
        return _BoxLikeMixin._as_legend_item(self)

    def _make_cat_iterator(self) -> CatIterator[_DF]:
        from whitecanvas.canvas.dataframe._base import CatIterator

        return CatIterator(self._source, offsets=self._offsets)

    def _is_edge_only(self) -> bool:
        return np.all(self.base.face.alpha < 1e-6)


class DFBoxPlot(
    _shared.DataFrameLayerWrapper[_lg.BoxPlot, _DF], _BoxLikeMixin, Generic[_DF]
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        width: float = 1.0,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        capsize: float = 0.1,
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _shared.norm_dodge(
            cat.df, cat.offsets, color, hatch, dodge=dodge,
        )  # fmt: skip
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _extent = cat.zoom_factor(dodge=dodge) * extent
        _capsize = cat.zoom_factor(dodge=dodge) * capsize
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat.df)
        base = _lg.BoxPlot.from_arrays(
            x, arr, name=name, orient=orient, capsize=_capsize, extent=_extent,
            backend=backend,
        )  # fmt: skip
        base.edge.width = width
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)
        self._offsets = cat.offsets
        self._value = value
        self._dodge = dodge

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._base_layer.orient

    def move(self, shift: float = 0.0) -> Self:
        """Move the layer by the given shift."""
        self._base_layer.move(shift)
        return self

    def with_hover_text(self, text: str | list[str]) -> Self:
        """Set the hover tooltip text for the layer."""
        self.base.boxes.with_hover_text(text)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._categories]
        self.base.boxes.with_hover_template(template, extra=extra)
        return self

    def with_outliers(
        self,
        *,
        color: ColorType | None = None,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float | None = None,
        ratio: float = 1.5,
        extent: float = 0.1,
        seed: int | None = 0,
        update_whiskers: bool = True,
    ) -> _lg.MainAndOtherLayers[Self, DFMarkerGroups[_DF]]:
        """
        Overlay outliers on the box plot.

        Parameters
        ----------
        color : color-type, optional
            Color of the outliers. To make sure the outliers are easily visible, face
            color will always be transparent. If a constant color is given, all the
            edges will be colored by the same color. By default, the edge colors are
            the same as the edge colors of the box plot.
        symbol : str or Symbol, optional
            Symbol of the outlier markers.
        size : float, optional
            Size of the outlier markers. If not given, it will be set to the theme
            default.
        ratio : float, optional
            Ratio of the interquartile range (IQR) to determine the outliers. Data
            points outside of the range [Q1 - ratio * IQR, Q3 + ratio * IQR] will be
            considered as outliers.
        extent : float, optional
            Relative width of the jitter range (same effect as the `extent` argument of
            the `add_stripplot` method).
        seed : int, optional
            Random seed for the jitter (same effect as the `seed` argument of the
            `add_stripplot` method).
        update_whiskers : bool, default True
            If True, the whiskers of the box plot will be updated to exclude the
            outliers.
        """
        from whitecanvas.canvas.dataframe._base import CatIterator
        from whitecanvas.layers.tabular import DFMarkerGroups

        canvas = self._canvas_ref()
        size = theme._default("markers.size", size)
        if canvas is None:
            raise ValueError("No canvas to add the outliers.")

        is_edge_only = np.all(self.base.boxes.face.alpha < 1e-6)

        # category iterator is used to calculate positions and indices
        _cat_self = CatIterator(self._source, offsets=self._offsets)
        _pos_map = _cat_self.prep_position_map(self._splitby, self._dodge)
        _extent = _cat_self.zoom_factor(self._dodge) * extent

        # calculate outliers and update the separators
        df_outliers = {c: [] for c in (*self._splitby, self._value)}
        agg_values = self.base._get_sep_values()  # for updating whiskers
        colors = []
        for idx_cat, (sl, sub) in enumerate(self._source.group_by(self._splitby)):
            arr = sub[self._value]
            q1, q3 = np.quantile(arr, [0.25, 0.75])
            iqr = q3 - q1  # interquartile range
            low = q1 - ratio * iqr  # lower bound of inliers
            high = q3 + ratio * iqr  # upper bound of inliers
            is_inlier = (low <= arr) & (arr <= high)
            inliers = arr[is_inlier]
            agg_values[0, idx_cat] = inliers.min()
            agg_values[4, idx_cat] = inliers.max()
            outliers = arr[~is_inlier]
            for _cat, _s in zip(sl, self._splitby):
                df_outliers[_s].extend([_cat] * outliers.size)
            df_outliers[self._value].extend(outliers)
            if is_edge_only:
                _this_color = self.base.edge.color[idx_cat]
            else:
                _this_color = self.base.face.color[idx_cat]
            colors.extend([_this_color] * outliers.size)

        df_outliers = parse(df_outliers)
        xj = _jitter.UniformJitter(self._splitby, _pos_map, extent=_extent, seed=seed)
        yj = _jitter.IdentityJitter(self._value).check(df_outliers)
        new = DFMarkerGroups(
            df_outliers, xj, yj, name=f"outliers-of-{self.name}", color=Color("black"),
            orient=self.orient, symbol=symbol, size=size, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is None:
            if is_edge_only:  # edge only
                new._apply_color(np.stack(colors, axis=0, dtype=np.float32))
            new.as_edge_only(width=self.base.edge.width.mean())
        if update_whiskers:
            self.base._update_data(agg_values)
        return _combine_main_and_others(self, new)

    def as_edge_only(
        self,
        width: float = 3.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Self:
        """
        Replace the violin edge color with the face color and delete the face color.

        Parameters
        ----------
        width : float, optional
            Width of the edge.
        style : str or LineStyle, optional
            Style of the edge.
        """
        self.base.with_edge(color=self.base.face.color, width=width, style=style)
        self.base.face.update(alpha=0.0)
        return self

    def _as_legend_item(self) -> _legend.LegendItemCollection:
        return _BoxLikeMixin._as_legend_item(self)


class _EstimatorMixin(_BoxLikeMixin):
    orient: Orientation

    def est_by_mean(self) -> Self:
        """Set estimator to mean."""

        def est_func(x):
            return np.mean(x)

        return self._update_estimate(est_func)

    def est_by_median(self) -> Self:
        """Set estimator to median."""

        def est_func(x):
            return np.median(x)

        return self._update_estimate(est_func)

    def err_by_sd(self, scale: float = 1.0, *, ddof: int = 1) -> Self:
        """Set error to standard deviation."""

        def err_func(x):
            _mean = np.mean(x)
            _sd = np.std(x, ddof=ddof) * scale
            return _mean - _sd, _mean + _sd

        return self._update_error(err_func)

    def err_by_se(self, scale: float = 1.0, *, ddof: int = 1) -> Self:
        """Set error to standard error."""

        def err_func(x):
            _mean = np.mean(x)
            _er = np.std(x, ddof=ddof) / np.sqrt(len(x)) * scale
            return _mean - _er, _mean + _er

        return self._update_error(err_func)

    def err_by_quantile(self, low: float = 0.25, high: float | None = None) -> Self:
        """Set error to quantile."""
        if low < 0 or low > 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {low}")
        if high is None:
            high = 1 - low
        elif high < 0 or high > 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {high}")

        def err_func(x):
            _qnt = np.quantile(x, [low, high])
            return _qnt[0], _qnt[1]

        return self._update_error(err_func)

    def _update_estimate(self, est_func: Callable[[np.ndarray], float]) -> Self:
        arrays = self._get_arrays()
        est = [est_func(arr) for arr in arrays]
        self._set_estimation_values(est)
        return self

    def _update_error(
        self,
        err_func: Callable[[np.ndarray], tuple[float, float]],
    ) -> Self:
        arrays = self._get_arrays()
        err_low = []
        err_high = []
        for arr in arrays:
            low, high = err_func(arr)
            err_low.append(low)
            err_high.append(high)
        self._set_error_values(err_low, err_high)
        return self


class DFPointPlot(
    _shared.DataFrameLayerWrapper[_lg.LabeledPlot, _DF], _EstimatorMixin, Generic[_DF]
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _shared.norm_dodge(
            cat.df, cat.offsets, color, hatch, dodge=dodge,
        )  # fmt: skip
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _capsize = cat.zoom_factor(dodge=dodge) * capsize
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat.df)
        base = _lg.LabeledPlot.from_arrays(
            x, arr, name=name, orient=orient, capsize=_capsize, backend=backend,
        )  # fmt: skip
        self._arrays = arr
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)
        base.with_edge(color=theme.get_theme().foreground_color)
        self._orient = orient

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._orient

    def move(self, shift: float = 0.0) -> Self:
        """Move the layer by the given shift."""
        base = self._base_layer
        data = base.data
        if self._orient.is_vertical:
            base.set_data(data.x + shift, data.y)
        else:
            base.set_data(data.x, data.y + shift)
        return self

    def _get_arrays(self) -> list[np.ndarray]:
        return self._arrays

    def _set_estimation_values(self, est):
        if self.orient.is_vertical:
            self._base_layer.set_data(ydata=est)
        else:
            self._base_layer.set_data(xdata=est)

    def _set_error_values(self, err_low, err_high):
        mdata = self._base_layer.data
        if self.orient.is_vertical:
            self._base_layer.yerr.set_data(mdata.x, err_low, err_high)
        else:
            self._base_layer.xerr.set_data(err_low, err_high, mdata.y)

    def with_hover_text(self, text: str | list[str]) -> Self:
        """Set the hover tooltip text for the layer."""
        self.base.markers.with_hover_text(text)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._categories]
        self.base.markers.with_hover_template(template, extra=extra)
        return self

    def _as_legend_item(self) -> _legend.LegendItemCollection:
        return _BoxLikeMixin._as_legend_item(self)


class DFBarPlot(
    _shared.DataFrameLayerWrapper[_lg.LabeledBars, _DF], _EstimatorMixin, Generic[_DF]
):
    def __init__(
        self,
        cat: CatIterator[_DF],
        value: str,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        dodge: str | tuple[str, ...] | bool | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        _splitby, dodge = _shared.norm_dodge(
            cat.df, cat.offsets, color, hatch, dodge=dodge,
        )  # fmt: skip
        x, arr, categories = cat.prep_arrays(_splitby, value, dodge=dodge)
        _extent = cat.zoom_factor(dodge=dodge) * extent
        _capsize = cat.zoom_factor(dodge=dodge) * capsize
        color_by, hatch_by = _norm_color_hatch(color, hatch, cat.df)
        base = _lg.LabeledBars.from_arrays(
            x, arr, name=name, orient=orient, capsize=_capsize, extent=_extent,
            backend=backend,
        )  # fmt: skip
        self._arrays = arr
        super().__init__(base, cat.df)
        _BoxLikeMixin.__init__(self, categories, _splitby, color_by, hatch_by)
        base.with_edge(color=theme.get_theme().foreground_color)
        self._orient = orient

    @property
    def orient(self) -> Orientation:
        return self._base_layer.bars.orient

    def _get_arrays(self) -> list[np.ndarray]:
        return self._arrays

    def _set_estimation_values(self, est):
        if self.orient.is_vertical:
            self._base_layer.set_data(ydata=est)
        else:
            self._base_layer.set_data(xdata=est)

    def _set_error_values(self, err_low, err_high):
        mdata = self._base_layer.data
        if self.orient.is_vertical:
            self._base_layer.yerr.set_data(mdata.x, err_low, err_high)
        else:
            self._base_layer.xerr.set_data(err_low, err_high, mdata.y)

    def with_hover_text(self, text: str | list[str]) -> Self:
        """Set the hover tooltip text for the layer."""
        self.base.bars.with_hover_text(text)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._categories]
        self.base.bars.with_hover_template(template, extra=extra)
        return self

    def _as_legend_item(self) -> _legend.LegendItemCollection:
        return _BoxLikeMixin._as_legend_item(self)


_L0 = TypeVar("_L0", bound=Layer)
_L1 = TypeVar("_L1", bound=Layer)


def _combine_main_and_others(
    layer: _L0,
    incoming: _L1,
) -> _lg.MainAndOtherLayers[_L0, _L1]:
    if layer._group_layer_ref is None:
        return _lg.MainAndOtherLayers([layer, incoming], name=layer.name)
    group_layer = layer._group_layer_ref()
    if group_layer is None:
        raise ValueError("Parent layer group is deleted.")
    elif not isinstance(group_layer, _lg.MainAndOtherLayers):
        raise ValueError(f"Parent layer group is incorrect type {type(group_layer)}.")
    group_layer._insert(incoming)
    return group_layer
