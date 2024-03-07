"""Layer with a dataframe bound to it."""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, TypeVar, overload

import numpy as np
from cmap import Color, Colormap

from whitecanvas import layers as _l
from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers import group as _lg
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, parse
from whitecanvas.types import (
    ColormapType,
    ColorType,
    Hatch,
    HistBinType,
    KdeBandWidthType,
    LineStyle,
    Orientation,
    OrientationLike,
    Rect,
    Symbol,
)
from whitecanvas.utils.hist import histograms

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas.dataframe._base import CatIterator


_DF = TypeVar("_DF")


class DFLines(_shared.DataFrameLayerWrapper[_lg.LineCollection, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        segs: list[np.ndarray],
        categories: list[tuple[Any, ...]],
        color: str | tuple[str, ...] | None = None,
        width: float = 1.0,
        style: str | tuple[str, ...] | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._style_by = _p.StylePlan.default()
        self._categories = categories
        self._splitby = splitby
        base = _lg.LineCollection(segs, name=name, backend=backend)
        super().__init__(base, source)
        if color is not None:
            self.update_color(color)
        self.update_width(width)
        if style is not None:
            self.update_style(style)
        self.with_hover_template("\n".join(f"{k}: {{{k}!r}}" for k in self._splitby))

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        x: str | _jitter.JitterBase,
        y: str | _jitter.JitterBase,
        color: str | None = None,
        width: float | None = None,
        style: str | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ) -> DFLines[_DF]:
        splitby = _shared.join_columns(color, style, source=df)
        segs = []
        labels: list[tuple[Any, ...]] = []
        if isinstance(x, _jitter.JitterBase):
            xj = x
        else:
            xj = _jitter.IdentityJitter(x)
        if isinstance(y, _jitter.JitterBase):
            yj = y
        else:
            yj = _jitter.IdentityJitter(y)
        for sl, sub in df.group_by(splitby):
            labels.append(sl)
            segs.append(np.column_stack([xj.map(sub), yj.map(sub)]))
        return DFLines(
            df, segs, labels, name=name, color=color, width=width, style=style,
            backend=backend,
        )  # fmt: skip

    @overload
    def update_color(self, value: ColorType) -> Self:
        ...

    @overload
    def update_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def update_color(self, by, /, palette=None):
        """Update the color rule of the layer."""
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.color = color_by.generate(self._categories, self._splitby)
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        """Update width of the lines."""
        self._base_layer.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
        """Update the styling rule of the layer."""
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        self._base_layer.style = style_by.generate(self._categories, self._splitby)
        self._style_by = style_by
        return self

    def move(self, dx: float = 0.0, dy: float = 0.0, autoscale: bool = True) -> Self:
        """Add a constant shift to the layer."""
        for layer in self._base_layer:
            old_data = layer.data
            new_data = old_data[0] + dx, old_data[1] + dy
            layer.data = new_data
        if autoscale and (canvas := self._canvas_ref()):
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def with_hover_template(self, template: str) -> Self:
        """Define hover template to the layer."""
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._categories]
        self.base.with_hover_template(template, extra=extra)
        return self

    def with_markers(
        self,
        *,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float | None = None,
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _lg.MainAndOtherLayers[DFLines[_DF], _lg.MarkerCollection]:
        markers = self._base_layer._prep_markers(
            symbol=symbol, size=size, alpha=alpha, hatch=hatch,
        )  # fmt: skip
        return _lg.MainAndOtherLayers([self, markers], name=self.name)

    def _simple_dataframe(self) -> DataFrameWrapper[dict]:
        return _shared.list_to_df(self._categories, self._splitby)

    def _prep_legend_info(
        self,
    ) -> tuple[list[tuple[str, ColorType]], list[tuple[str, LineStyle]]]:
        df = self._simple_dataframe()
        color_entries = self._color_by.to_entries(df)
        style_entries = self._style_by.to_entries(df)
        return color_entries, style_entries

    def _as_legend_item(self) -> _legend.LegendItemCollection | _legend.LineLegendItem:
        colors, styles = self._prep_legend_info()
        ncolor = len(colors)
        nstyle = len(styles)
        widths = self._base_layer.width

        color_default = theme.get_theme().foreground_color
        style_default = LineStyle.SOLID
        if ncolor == 1:
            _, color_default = colors[0]
        if nstyle == 1:
            _, style_default = styles[0]

        if ncolor == 1 and nstyle == 1:
            return _legend.LineLegendItem(color_default, widths[0], style_default)
        items = []
        if ncolor > 1:
            items.append((", ".join(self._color_by.by), _legend.TitleItem()))
            for (label, color), w in zip(colors, widths):
                item = _legend.LineLegendItem(color, w, style_default)
                items.append((label, item))
        if nstyle > 1:
            items.append((", ".join(self._style_by.by), _legend.TitleItem()))
            for (label, style), w in zip(styles, widths):
                item = _legend.LineLegendItem(color_default, w, style)
                items.append((label, item))
        return _legend.LegendItemCollection(items)


class DFHeatmap(_shared.DataFrameLayerWrapper[_lg.LabeledImage, _DF], Generic[_DF]):
    @property
    def cmap(self) -> Colormap:
        """Colormap of the heatmap."""
        return self._base_layer.cmap

    @cmap.setter
    def cmap(self, cmap: ColormapType):
        self._base_layer.cmap = Colormap(cmap)

    @property
    def clim(self) -> tuple[float, float]:
        """Color limits of the heatmap."""
        return self._base_layer.clim

    @clim.setter
    def clim(self, clim: tuple[float, float]):
        self._base_layer.clim = clim

    @classmethod
    def from_array(
        cls,
        src: DataFrameWrapper[_DF],
        arr: np.ndarray,
        name: str | None = None,
        cmap: ColormapType = "gray",
        clim: tuple[float | None, float | None] | None = None,
        backend: Backend | str | None = None,
    ) -> DFHeatmap[_DF]:
        base = _lg.LabeledImage(
            _l.Image(
                arr, name=f"image-of-{name}", cmap=cmap, clim=clim, backend=backend
            ),
            name=name,
        )
        return cls(base, src)

    def with_text(
        self,
        *,
        size: int = 8,
        color_rule: ColorType | Callable[[np.ndarray], ColorType] | None = None,
        fmt: str = "",
        text_invalid: str | None = None,
    ) -> Self:
        """Add text layer to the heatmap."""
        self._base_layer.with_text(
            size=size, color_rule=color_rule, fmt=fmt, text_invalid=text_invalid
        )
        return self

    def with_colorbar(
        self,
        bbox: Rect | None = None,
        orient: OrientationLike = "vertical",
    ) -> Self:
        """Add colorbar to the heatmap."""
        self._base_layer.with_colorbar(bbox=bbox, orient=orient)
        return self


class DFMultiHeatmap(
    _shared.DataFrameLayerWrapper[_lg.LayerCollectionBase[_l.Image], _DF],
    Generic[_DF],
):
    _NO_PADDING_NEEDED = True

    def __init__(
        self,
        base: _lg.LayerCollectionBase[_l.Image],
        source: DataFrameWrapper[_DF],
        color_by: _p.ColorPlan,
        categories: list[tuple],
    ):
        self._color_by = color_by
        self._categories = categories
        super().__init__(base, source)

    @classmethod
    def build_hist(
        cls,
        df: _DF,
        x: str,
        y: str,
        name: str | None = None,
        color: str | list[str] | None = None,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        range=None,
        palette: ColormapType = "tab10",
        backend: Backend | str | None = None,
    ) -> Self:
        src, color = cls._norm_df_xy_color(df, [x, y], color)
        # normalize bins
        if isinstance(bins, tuple):
            xbins, ybins = bins
        else:
            xbins = ybins = bins
        if range is None:
            xrange = yrange = None
        else:
            xrange, yrange = range
        _bins = (
            np.histogram_bin_edges(src[x], xbins, xrange),
            np.histogram_bin_edges(src[y], ybins, yrange),
        )

        color_by = _p.ColorPlan.from_palette(color, palette)
        image_layers: list[_l.Image] = []
        categories = []
        color_iter = cycle(color_by.values)
        for sl, sub in src.group_by(color):
            categories.append(sl)
            xdata, ydata = sub[x], sub[y]
            cmap = _gen_cmap_from_color(next(color_iter))
            img = _l.Image.build_hist(
                xdata, ydata, name=f"heatmap-{sl}", cmap=cmap, bins=_bins, density=True,
                backend=backend,
            )  # fmt: skip
            image_layers.append(img)
        base = _lg.LayerCollectionBase(image_layers, name=name)
        return cls(base, src, color_by, categories)

    @classmethod
    def build_kde(
        cls,
        df: _DF,
        x: str,
        y: str,
        name: str | None = None,
        color: str | list[str] | None = None,
        band_width: KdeBandWidthType = "scott",
        palette: ColormapType = "tab10",
        backend: Backend | str | None = None,
    ) -> Self:
        src, color = cls._norm_df_xy_color(df, [x, y], color)
        color_by = _p.ColorPlan.from_palette(color, palette)
        image_layers: list[_l.Image] = []
        categories = []
        color_iter = cycle(color_by.values)
        xrange = src[x].min(), src[x].max()
        yrange = src[y].min(), src[y].max()
        for sl, sub in src.group_by(color):
            categories.append(sl)
            xdata, ydata = sub[x], sub[y]
            cmap = _gen_cmap_from_color(next(color_iter))
            img = _l.Image.build_kde(
                xdata, ydata, name=f"heatmap-{sl}", cmap=cmap, band_width=band_width,
                range=(xrange, yrange), backend=backend,
            )  # fmt: skip
            image_layers.append(img)
        base = _lg.LayerCollectionBase(image_layers, name=name)
        return cls(base, src, color_by, categories)

    @classmethod
    def build_hist_1d(
        cls,
        cat: CatIterator[_DF],
        offsets: str | list[str],
        value: str,
        name: str | None = None,
        color: str | list[str] | None = None,
        dodge: str | list[str] | bool = False,
        bins: HistBinType = "auto",
        range=None,
        palette: ColormapType = "tab10",
        orient: OrientationLike = "vertical",
        backend: Backend | str | None = None,
    ) -> DFMultiHeatmap[_DF]:
        if not isinstance(bins, (int, np.integer, str)):
            raise NotImplementedError("Only equal-width bins are supported.")
        src, color = cls._norm_df_xy_color(cat.df, [value], color)
        ori = Orientation.parse(orient)
        splitby, dodge = _shared.norm_dodge(cat.df, offsets, color, dodge=dodge)
        xs, arrays, categories = cat.prep_arrays(splitby, value, dodge=dodge, width=1.0)
        extent = cat.zoom_factor(dodge, width=1.0)
        hist = histograms(arrays, bins, range)
        ymin, ymax = hist.edges[0], hist.edges[-1]
        image_layers = list[_l.Image]()
        densities = hist.density()
        color_by = _p.ColorPlan.from_palette(color, palette)
        clim = (0, max(dens.max() for dens in densities))
        colors = color_by.generate(categories, splitby)
        for i, dens in enumerate(densities):
            cmap = _gen_cmap_from_color(colors[i])
            x = xs[i]
            if ori.is_vertical:
                img = _l.Image(
                    dens.reshape(-1, 1), cmap=cmap, clim=clim, backend=backend,
                ).fit_to(x - extent / 2, x + extent / 2, ymin, ymax)  # fmt: skip
            else:
                img = _l.Image(
                    dens.reshape(1, -1), cmap=cmap, clim=clim, backend=backend,
                ).fit_to(ymin, ymax, x - extent / 2, x + extent / 2)  # fmt: skip
            image_layers.append(img)
        base = _lg.LayerCollectionBase(image_layers, name=name)
        return cls(base, src, color_by, categories)

    @staticmethod
    def _norm_df_xy_color(df, cols: list[str], color):
        src = parse(df)
        # dtype check
        for x in cols:
            if src[x].dtype.kind not in "fiub":
                raise ValueError(f"Column {x!r} is not numeric.")

        if isinstance(color, str):
            color = (color,)
        elif color is None:
            color = ()
        else:
            color = tuple(color)
        return src, color

    def _as_legend_item(self) -> _legend.LegendItem:
        if len(self._categories) == 1:
            face = _legend.FaceInfo(self._color_by.values[0])
            edge = _legend.EdgeInfo(self._color_by.values[0], width=0)
            return _legend.BarLegendItem(face, edge)
        df = _shared.list_to_df(self._categories, self._color_by.by)
        colors = self._color_by.to_entries(df)
        items = [(", ".join(self._color_by.by), _legend.TitleItem())]
        for label, color in colors:
            face = _legend.FaceInfo(color)
            edge = _legend.EdgeInfo(color, width=0)
            items.append((label, _legend.BarLegendItem(face, edge)))
        return _legend.LegendItemCollection(items)


def _gen_cmap_from_color(next_color: Color):
    next_background = Color([*next_color.rgba[:3], 0.0])
    return [next_background, next_color]


class DFPointPlot2D(_shared.DataFrameLayerWrapper[_lg.LabeledPlot, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        x: str,
        y: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        size: float | None = None,
        capsize: float = 0.15,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        cols = _shared.join_columns(color, hatch, source=source)
        xdata = []
        ydata = []
        for _, sub in source.group_by(cols):
            xdata.append(sub[x])
            ydata.append(sub[y])
        base = _lg.LabeledPlot.from_arrays_2d(
            xdata, ydata, name=name, capsize=capsize, backend=backend
        )
        if size is not None:
            base.markers.size = size
        super().__init__(base, source)


_L = TypeVar("_L", bound=_lg.LineFillBase)


class DFLineFillBase(
    _shared.DataFrameLayerWrapper[_lg.LayerCollectionBase[_L], _DF],
    Generic[_L, _DF],
):
    _ATTACH_TO_AXIS = True

    def __init__(
        self,
        base: _lg.LayerCollectionBase[_L],
        source: DataFrameWrapper[_DF],
        categories: list[tuple[Any, ...]],
        splitby: tuple[str, ...],
    ):
        self._categories = categories
        self._splitby = splitby
        super().__init__(base, source)
        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._hatch_by = _p.HatchPlan.default()

    @overload
    def update_color(self, value: ColorType) -> Self:
        ...

    @overload
    def update_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def update_color(self, by, /, palette=None):
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        for i, col in enumerate(color_by.generate(self._categories, self._splitby)):
            self._base_layer[i].color = col
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        for hist in self._base_layer:
            hist.line.width = value
        return self

    def update_style(self, by: str | Iterable[str], palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=palette)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        for i, st in enumerate(style_by.generate(self._categories, self._splitby)):
            self._base_layer[i].line.style = st
        self._style_by = style_by
        return self

    def update_hatch(self, by: str | Iterable[str], styles=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot hatch by a column other than {self._splitby}")
            hatch_by = _p.HatchPlan.new(cov.columns, values=styles)
        else:
            hatch_by = _p.HatchPlan.from_const(cov.value)
        for i, st in enumerate(hatch_by.generate(self._categories, self._splitby)):
            self._base_layer[i].fill.face.hatch = st
        self._hatch_by = hatch_by
        return self

    def _as_legend_item(self) -> _legend.LegendItemCollection:
        if len(self.base) == 0:
            return _legend.LegendItemCollection([])
        df = _shared.list_to_df(self._categories, self._splitby)
        colors = self._color_by.to_entries(df)
        hatches = self._hatch_by.to_entries(df)
        items = []
        color_default = theme.get_theme().background_color
        fill_alpha = self.base[0].fill_alpha
        width = self.base[0].line.width
        style = self.base[0].line.style
        if self._color_by.is_const():
            color_default = Color(self._color_by.values[0])
        else:
            items.append((", ".join(self._color_by.by), _legend.TitleItem()))
            for label, color in colors:
                fc = Color([*color.rgba[:3], fill_alpha])
                _face = _legend.FaceInfo(fc)
                _edge = _legend.EdgeInfo(color, width, style)
                item = _legend.BarLegendItem(_face, _edge)
                items.append((label, item))
        if self._hatch_by.is_not_const():
            items.append((", ".join(self._hatch_by.by), _legend.TitleItem()))
            for label, hatch in hatches:
                fc = Color([*color_default.rgba[:3], fill_alpha])
                _face = _legend.FaceInfo(fc, hatch)
                _edge = _legend.EdgeInfo(color_default, width, style)
                item = _legend.BarLegendItem(_face, _edge)
                items.append((label, item))
        return _legend.LegendItemCollection(items)


class DFHistograms(DFLineFillBase[_lg.Histogram, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        base: _lg.LayerCollectionBase[_lg.Histogram],
        labels: list[tuple[Any, ...]],
        color: str | tuple[str, ...] | None = None,
        width: str | None = None,
        style: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        super().__init__(base, source, labels, splitby)
        if color is not None:
            self.update_color(color)
        if isinstance(width, str):
            self.update_width(width)
        if style is not None:
            self.update_style(style)
        if hatch is not None:
            self.update_hatch(hatch)

    @property
    def orient(self) -> Orientation:
        if len(self.base) > 0:
            return self.base[0].orient
        return Orientation.VERTICAL

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        kind="count",
        shape="bars",
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        backend: str | Backend | None = None,
    ) -> DFHistograms[_DF]:
        splitby = _shared.join_columns(color, style, source=df)
        ori = Orientation.parse(orient)
        arrays: list[np.ndarray] = []
        labels: list[tuple] = []
        for sl, sub in df.group_by(splitby):
            labels.append(sl)
            arrays.append(sub[value])
        hist = histograms(arrays, bins, limits)

        layers = []
        for arr in arrays:
            each_layer = _lg.Histogram.from_array(
                arr, kind=kind, bins=hist.edges, limits=limits, width=width,
                orient=ori, shape=shape, backend=backend,
            )  # fmt: skip
            layers.append(each_layer)
        base = _lg.LayerCollectionBase(layers, name=name)
        return cls(df, base, labels, color=color, width=width, style=style, hatch=hatch)


class DFKde(DFLineFillBase[_lg.Kde, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        base: _lg.LayerCollectionBase[_lg.Kde],
        labels: list[tuple[Any, ...]],
        color: str | tuple[str, ...] | None = None,
        width: str | None = None,
        style: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        super().__init__(base, source, labels, splitby)
        if color is not None:
            self.update_color(color)
        if isinstance(width, str):
            self.update_width(width)
        if style is not None:
            self.update_style(style)
        if hatch is not None:
            self.update_hatch(hatch)

    @property
    def orient(self) -> Orientation:
        if len(self.base) > 0:
            return self.base[0].orient
        return Orientation.VERTICAL

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        band_width: KdeBandWidthType = "scott",
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        backend: str | Backend | None = None,
    ) -> DFKde[_DF]:
        splitby = _shared.join_columns(color, style, source=df)
        ori = Orientation.parse(orient)
        arrays: list[np.ndarray] = []
        labels: list[tuple] = []
        for sl, sub in df.group_by(splitby):
            labels.append(sl)
            arrays.append(sub[value])
        layers = []
        for arr in arrays:
            each_layer = _lg.Kde.from_array(
                arr, width=width, band_width=band_width, orient=ori, backend=backend,
            )  # fmt: skip
            layers.append(each_layer)
        base = _lg.LayerCollectionBase(layers, name=name)
        return cls(df, base, labels, color=color, width=width, style=style, hatch=hatch)
