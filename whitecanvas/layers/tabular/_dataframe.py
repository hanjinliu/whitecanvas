"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar, Union, overload

import numpy as np
from cmap import Color, Colormap

from whitecanvas import layers as _l
from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, parse
from whitecanvas.types import (
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    Symbol,
    _Void,
)
from whitecanvas.utils.hist import histograms

if TYPE_CHECKING:
    from typing_extensions import Self

_DF = TypeVar("_DF")
_Cols = Union[str, "tuple[str, ...]"]
_void = _Void()


class DFLines(_shared.DataFrameLayerWrapper[_lg.LineCollection, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        segs: list[np.ndarray],
        labels: list[tuple[Any, ...]],
        color: _Cols | None = None,
        width: float = 1.0,
        style: _Cols | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = labels
        self._splitby = splitby
        base = _lg.LineCollection(segs, name=name, backend=backend)
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        self.with_width(width)
        if style is not None:
            self.with_style(style)

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

    @classmethod
    def build_kde(
        cls,
        df: _DF,
        value: str,
        band_width: float | None = None,
        color: str | None = None,
        width: str | None = None,
        style: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFLines[_DF]:
        from whitecanvas.utils.kde import gaussian_kde

        src = parse(df)
        splitby = _shared.join_columns(color, style, source=src)
        ori = Orientation.parse(orient)
        segs = []
        labels: list[tuple[Any, ...]] = []
        for sl, df in src.group_by(splitby):
            labels.append(sl)
            each = df[value]
            kde = gaussian_kde(each, bw_method=band_width)
            sigma = np.sqrt(kde.covariance[0, 0])
            pad = sigma * 2.5
            x = np.linspace(each.min() - pad, each.max() + pad, 100)
            y = kde(x)
            if ori.is_vertical:
                segs.append(np.column_stack([x, y]))
            else:
                segs.append(np.column_stack([y, x]))
        return DFLines(
            src, segs, labels, name=name, color=color, width=width, style=style,
            backend=backend,
        )  # fmt: skip

    @overload
    def with_color(self, value: ColorType) -> Self:
        ...

    @overload
    def with_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def with_color(self, by, /, palette=None):
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.color = color_by.generate(self._labels, self._splitby)
        self._color_by = color_by
        return self

    def with_width(self, value: float) -> Self:
        self._base_layer.width = value
        return self

    def with_style(self, by: str | Iterable[str], styles=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        self._base_layer.style = style_by.generate(self._labels, self._splitby)
        self._style_by = style_by
        return self

    def with_shift(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> Self:
        """Add a constant shift to the layer."""
        for layer in self._base_layer:
            old_data = layer.data
            new_data = (old_data[0] + dx, old_data[1] + dy)
            layer.data = new_data
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self


class DFMarkers(_shared.DataFrameLayerWrapper[_lg.MarkerCollection, _DF]):
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
        self._color_by = _p.ColorPlan.default()
        self._edge_color_by = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._size_by = _p.SizePlan.default()
        self._symbol_by = _p.SymbolPlan.default()
        self._width_by = _p.WidthPlan.default()

        base = _lg.MarkerCollection(
            x.map(source), y.map(source), name=name, backend=backend
        )

        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)
        if symbol is not None:
            self.with_symbol(symbol)
        if size is not None:
            self.with_size(size)
        else:
            self.with_size(theme.get_theme().markers.size)

    @overload
    def with_color(self, value: ColorType) -> Self:
        ...

    @overload
    def with_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def with_color(self, by, /, palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        colors = color_by.map(self._source)
        self._base_layer.face.color = colors
        self._color_by = color_by
        return self

    def with_colormap(
        self,
        by: str,
        cmap: ColormapType | None = None,
        clim: tuple[float, float] | None = None,
    ) -> Self:
        """Update the face colormap."""
        if not isinstance(by, str):
            raise ValueError("Can only colormap by a single column.")
        if cmap is None:
            cmap = Colormap("viridis")
        else:
            cmap = Colormap(cmap)
        color_by = _p.ColormapPlan.from_colormap(by, cmap=cmap, clim=clim)
        colors = color_by.map(self._source)
        self._base_layer.face.color = colors
        self._color_by = color_by
        return self

    def with_edge_color(self, by: str | Iterable[str], palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        colors = color_by.map(self._source)
        self._base_layer.edge.color = colors
        self._edge_color_by = color_by
        return self

    def with_edge_colormap(
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

    def with_hatch(self, by: str | Iterable[str], palette=None) -> Self:
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
    def with_size(self, value: float) -> Self:
        ...

    @overload
    def with_size(self, by: str, limits=None) -> Self:
        ...

    def with_size(self, by, /, limits=None):
        """Set the size of the markers."""
        if isinstance(by, str):
            size_by = _p.SizePlan.from_range(by, limits=limits)
        else:
            size_by = _p.SizePlan.from_const(float(by))
        self._base_layer.size = size_by.map(self._source)
        self._size_by = size_by
        return self

    @overload
    def with_symbol(self, value: str | Symbol) -> Self:
        ...

    @overload
    def with_symbol(self, by: str | Iterable[str] | None = None, symbols=None) -> Self:
        ...

    def with_symbol(self, by, /, symbols=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            symbol_by = _p.SymbolPlan.new(cov.columns, values=symbols)
        else:
            symbol_by = _p.SymbolPlan.from_const(Symbol(by))
        self._base_layer.symbol = symbol_by.map(self._source)
        self._symbol_by = symbol_by
        return self

    @overload
    def with_width(self, value: float) -> Self:
        ...

    @overload
    def with_width(self, by: str, limits=None) -> Self:
        ...

    def with_width(self, by, /, limits=None) -> Self:
        if isinstance(by, str):
            width_by = _p.WidthPlan.from_range(by, limits=limits)
        else:
            width_by = _p.WidthPlan.from_const(float(by))
        self._base_layer.with_edge(color=_void, width=width_by.map(self._source))
        self._width_by = width_by
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
    ) -> Self:
        if color is not None:
            self.with_edge_color(color)
        self.with_width(width)
        self._base_layer.edge.style = LineStyle(style)
        return self

    def with_shift(self, dx: float = 0.0, dy: float = 0.0) -> Self:
        """Add a constant shift to the layer."""
        _old_data = self._base_layer.data
        self._base_layer.set_data(xdata=_old_data.x + dx, ydata=_old_data.y + dy)
        if canvas := self._canvas_ref():
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
        return self


class DFMarkerGroups(DFMarkers):
    def __init__(self, *args, orient: Orientation = Orientation.VERTICAL, **kwargs):
        super().__init__(*args, **kwargs)
        self._orient = Orientation.parse(orient)

    @property
    def orient(self) -> Orientation:
        """Orientation of the plot."""
        return self._orient

    def with_shift(self, shift: float = 0.0) -> Self:
        """Add a constant shift to the layer."""
        _old_data = self._base_layer.data
        if self.orient.is_vertical:
            self._base_layer.set_data(xdata=_old_data.x + shift)
        else:
            self._base_layer.set_data(ydata=_old_data.y + shift)
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self


class DFBars(
    _shared.DataFrameLayerWrapper[_l.Bars[_mixin.MultiFace, _mixin.MultiEdge], _DF],
    Generic[_DF],
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        x,
        y,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        splitby = _shared.join_columns(color, hatch, source=source)
        self._color_by = _p.ColorPlan.default()
        self._style_by = _p.StylePlan.default()
        self._splitby = splitby

        base = _l.Bars(
            x, y, name=name, orient=orient, extent=extent, backend=backend
        ).with_face_multi()
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

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
        splitby = _shared.join_columns(color, hatch, source=df)
        if isinstance(x, _jitter.JitterBase):
            xj = x
        else:
            xj = _jitter.IdentityJitter(x)
        if isinstance(y, _jitter.JitterBase):
            yj = y
        else:
            yj = _jitter.IdentityJitter(y)
        xs = []
        ys = []
        for _, sub in df.group_by(splitby):
            xs.append(xj.map(sub))
            ys.append(yj.map(sub))
        x0 = np.concatenate(xs)
        y0 = np.concatenate(ys)
        return DFBars(
            df, x0, y0, name=name, color=color, hatch=hatch, extent=extent,
            orient=orient, backend=backend,
        )  # fmt: skip

    def with_color(self, by: str | Iterable[str] | ColorType, palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.face.color = color_by.map(self._source)
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str], choices=None) -> Self:
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


class DFHeatmap(_shared.DataFrameLayerWrapper[_l.Image, _DF], Generic[_DF]):
    @property
    def cmap(self) -> Colormap:
        return self._base_layer.cmap

    @cmap.setter
    def cmap(self, cmap: ColormapType):
        self._base_layer.cmap = Colormap(cmap)

    @property
    def clim(self) -> tuple[float, float]:
        return self._base_layer.clim

    @clim.setter
    def clim(self, clim: tuple[float, float]):
        self._base_layer.clim = clim

    @classmethod
    def build_hist(
        cls,
        df: _DF,
        x: str,
        y: str,
        name: str | None = None,
        cmap: ColormapType = "gray",
        bins: int | tuple[int, int] = 10,
        range=None,
        density: bool = False,
        backend: Backend | str | None = None,
    ) -> Self:
        src = parse(df)
        xdata = src[x]
        ydata = src[y]
        if xdata.dtype.kind not in "fiub":
            raise ValueError(f"Column {x!r} is not numeric.")
        if ydata.dtype.kind not in "fiub":
            raise ValueError(f"Column {y!r} is not numeric.")
        base = _l.Image.build_hist(
            xdata, ydata, name=name, cmap=cmap, bins=bins, range=range,
            density=density, backend=backend,
        )  # fmt: skip
        return cls(base, src)

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
        return cls(_l.Image(arr, name=name, cmap=cmap, clim=clim, backend=backend), src)


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


class DFHistograms(
    _shared.DataFrameLayerWrapper[_lg.LayerCollectionBase[_lg.Histogram], _DF],
    Generic[_DF],
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        base: _lg.LayerCollectionBase[_lg.Histogram],
        labels: list[tuple[Any, ...]],
        color: _Cols | None = None,
        width: str | None = None,
        style: _Cols | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = labels
        self._splitby = splitby
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if isinstance(width, str):
            self.with_width(width)
        if style is not None:
            self.with_style(style)

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        bins: int | ArrayLike1D,
        limits: tuple[float, float] | None = None,
        kind="count",
        shape="bars",
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
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
                arr,
                kind=kind,
                bins=hist.edges,
                limits=limits,
                width=width,
                orient=ori,
                shape=shape,
                backend=backend,
            )
            layers.append(each_layer)
        base = _lg.LayerCollectionBase(layers, name=name)
        return cls(df, base, labels, color=color, width=width, style=style)

    @overload
    def with_color(self, value: ColorType) -> Self:
        ...

    @overload
    def with_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def with_color(self, by, /, palette=None):
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        for i, col in enumerate(color_by.generate(self._labels, self._splitby)):
            self._base_layer[i].color = col
        self._color_by = color_by
        return self

    def with_width(self, value: float) -> Self:
        for hist in self._base_layer:
            hist.line.width = value
        return self

    def with_style(self, by: str | Iterable[str], styles=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        for i, st in enumerate(style_by.generate(self._labels, self._splitby)):
            self._base_layer[i].line.style = st
        self._style_by = style_by
        return self


class DFKde(
    _shared.DataFrameLayerWrapper[_lg.LayerCollectionBase[_lg.Kde], _DF],
    Generic[_DF],
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        base: _lg.LayerCollectionBase[_lg.Kde],
        labels: list[tuple[Any, ...]],
        color: _Cols | None = None,
        width: str | None = None,
        style: _Cols | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = labels
        self._splitby = splitby
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if isinstance(width, str):
            self.with_width(width)
        if style is not None:
            self.with_style(style)

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        band_width: float | None = None,
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFHistograms[_DF]:
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
        return cls(df, base, labels, color=color, width=width, style=style)

    @overload
    def with_color(self, value: ColorType) -> Self:
        ...

    @overload
    def with_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def with_color(self, by, /, palette=None):
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        for i, col in enumerate(color_by.generate(self._labels, self._splitby)):
            self._base_layer[i].color = col
        self._color_by = color_by
        return self

    def with_width(self, value: float) -> Self:
        for hist in self._base_layer:
            hist.line.width = value
        return self

    def with_style(self, by: str | Iterable[str], styles=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        for i, st in enumerate(style_by.generate(self._labels, self._splitby)):
            self._base_layer[i].line.style = st
        self._style_by = style_by
        return self
