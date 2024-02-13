"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    TypeVar,
    Union,
    overload,
)

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
    ColormapType,
    ColorType,
    Hatch,
    HistBinType,
    KdeBandWidthType,
    LineStyle,
    Orientation,
    Symbol,
    _Void,
)
from whitecanvas.utils.hist import histograms
from whitecanvas.utils.type_check import is_real_number

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

    def update_width(self, value: float) -> Self:
        self._base_layer.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
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
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._labels]
        self.base.with_hover_template(template, extra=extra)
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

    def update_color(self, by, /, palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.face.color = color_by.map(self._source)
        self._color_by = color_by
        return self

    def update_colormap(
        self,
        by: str,
        cmap: ColormapType = "viridis",
        clim: tuple[float, float] | None = None,
    ) -> Self:
        """Update the face colormap."""
        if not isinstance(by, str):
            raise ValueError("Can only colormap by a single column.")
        color_by = _p.ColormapPlan.from_colormap(by, cmap=Colormap(cmap), clim=clim)
        self._base_layer.face.color = color_by.map(self._source)
        self._color_by = color_by
        return self

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
    def update_size(self, by: str, limits=None) -> Self:
        ...

    def update_size(self, by, /, limits=None):
        """Set the size of the markers."""
        if isinstance(by, str):
            size_by = _p.SizePlan.from_range(by, limits=limits)
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

    @overload
    def update_width(self, value: float) -> Self:
        ...

    @overload
    def update_width(self, by: str, limits=None) -> Self:
        ...

    def update_width(self, by, /, limits=None) -> Self:
        """Update the width of the markers."""
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
        return self

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = dict(self._source.iter_items())
        self.base.with_hover_template(template, extra=extra)
        return self


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
        splitby = _shared.join_columns(color, hatch, source=df)
        if isinstance(x, _jitter.JitterBase):
            xj = x
        else:
            xj = _jitter.IdentityJitter(x)
        if isinstance(y, _jitter.JitterBase):
            yj = y
        else:
            yj = _jitter.IdentityJitter(y)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for _, sub in df.group_by(splitby):
            xcur = xj.map(sub)
            ycur = yj.map(sub)
            order = np.argsort(xcur)
            xs.append(xcur[order])
            ys.append(ycur[order])
        # BUG: order of coloring and x/y do not match
        x0 = np.concatenate(xs)
        y0 = np.concatenate(ys)
        return DFBars(
            df, x0, y0, name=name, color=color, hatch=hatch, extent=extent,
            orient=orient, backend=backend,
        )  # fmt: skip

    def update_color(self, by: str | Iterable[str] | ColorType, palette=None) -> Self:
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

    def with_hover_template(self, template: str) -> Self:
        """Set the hover tooltip template for the layer."""
        extra = dict(self._source.iter_items())
        self.base.with_hover_template(template, extra=extra)
        return self


class DFHeatmap(_shared.DataFrameLayerWrapper[_l.Image, _DF], Generic[_DF]):
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
    def build_hist(
        cls,
        df: _DF,
        x: str,
        y: str,
        name: str | None = None,
        cmap: ColormapType = "gray",
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
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
            self.update_color(color)
        if isinstance(width, str):
            self.update_width(width)
        if style is not None:
            self.update_style(style)

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
                arr, kind=kind, bins=hist.edges, limits=limits, width=width,
                orient=ori, shape=shape, backend=backend,
            )  # fmt: skip
            layers.append(each_layer)
        base = _lg.LayerCollectionBase(layers, name=name)
        return cls(df, base, labels, color=color, width=width, style=style)

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
        for i, col in enumerate(color_by.generate(self._labels, self._splitby)):
            self._base_layer[i].color = col
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        for hist in self._base_layer:
            hist.line.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
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
            self.update_color(color)
        if isinstance(width, str):
            self.update_width(width)
        if style is not None:
            self.update_style(style)

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        band_width: KdeBandWidthType = "scott",
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
        for i, col in enumerate(color_by.generate(self._labels, self._splitby)):
            self._base_layer[i].color = col
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        for hist in self._base_layer:
            hist.line.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
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


class DFRug(_shared.DataFrameLayerWrapper[_l.Rug, _DF], Generic[_DF]):
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

    def update_color(self, by, /, palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.color = color_by.map(self._source)
        self._color_by = color_by
        return self

    def update_colormap(
        self,
        by: str,
        cmap: ColormapType = "viridis",
        clim: tuple[float, float] | None = None,
    ) -> Self:
        """Update the face colormap."""
        if not isinstance(by, str):
            raise ValueError("Can only colormap by a single column.")
        color_by = _p.ColormapPlan.from_colormap(by, cmap=Colormap(cmap), clim=clim)
        self._base_layer.color = color_by.map(self._source)
        self._color_by = color_by
        return self

    @overload
    def update_width(self, value: float) -> Self:
        ...

    @overload
    def update_width(self, by: str, limits=None) -> Self:
        ...

    def update_width(self, by, /, limits=None) -> Self:
        """Update the width of the markers."""
        if isinstance(by, str):
            width_by = _p.WidthPlan.from_range(by, limits=limits)
        else:
            if limits is not None:
                raise TypeError("Cannot set limits for a constant width.")
            width_by = _p.WidthPlan.from_const(float(by))
        self._base_layer.width = width_by.map(self._source)
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
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        self._base_layer.style = style_by.map(self._source)
        self._style_by = style_by
        return self

    def update_scale(self, by: str | float, align: str = "low") -> Self:
        ...

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
            df[value],
            orient=ori,
            low=x - dx,
            high=x + dx,
            name=name,
            backend=backend,
        )
        return cls(
            df,
            base,
            value,
            jitter.by,
            color=color,
            width=width,
            style=style,
            extent=extent,
        )

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

                │ │                  │ │
              ──┴─┴──   ──┬─┬──    ──┼─┼──
                          │ │        │ │
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
