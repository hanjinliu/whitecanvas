"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar, Union, overload

import numpy as np
from cmap import Color, Colormap

from whitecanvas import layers as _l
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

if TYPE_CHECKING:
    from typing_extensions import Self

_DF = TypeVar("_DF")
_Cols = Union[str, "tuple[str, ...]"]
_void = _Void()


class WrappedLines(
    _shared.DataFrameLayerWrapper[_lg.LineCollection, _DF], Generic[_DF]
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        segs: list[np.ndarray],
        labels: list[tuple[Any, ...]],
        color: _Cols | None = None,
        width: str | None = None,
        style: _Cols | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = labels
        self._splitby = splitby
        base = _lg.LineCollection(segs, name=name, backend=backend)
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
        df: _DF,
        x: str,
        y: str,
        color: str | None = None,
        width: str | None = None,
        style: str | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ) -> WrappedLines[_DF]:
        src = parse(df)
        splitby = _shared.join_columns(color, style, source=src)
        segs = []
        labels: list[tuple[Any, ...]] = []
        for sl, df in src.group_by(splitby):
            labels.append(sl)
            segs.append(np.column_stack([df[x], df[y]]))
        return WrappedLines(
            src, segs, labels, name=name, color=color, width=width, style=style,
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
    ) -> WrappedLines[_DF]:
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
        return WrappedLines(
            src, segs, labels, name=name, color=color, width=width, style=style,
            backend=backend,
        )  # fmt: skip

    @classmethod
    def build_hist(
        cls,
        df: _DF,
        value: str,
        bins: int | ArrayLike1D = 10,
        density: bool = False,
        range: tuple[float, float] | None = None,
        color: str | None = None,
        width: str | None = None,
        style: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> WrappedLines[_DF]:
        src = parse(df)
        splitby = _shared.join_columns(color, style, source=src)
        ori = Orientation.parse(orient)
        segs = []
        labels: list[tuple[Any, ...]] = []
        for sl, df in src.group_by(splitby):
            labels.append(sl)
            each = df[value]
            counts, edges = np.histogram(each, bins=bins, density=density, range=range)
            x = np.empty(2 * counts.size + 2, dtype=np.float32)
            y = np.empty(2 * counts.size + 2, dtype=np.float32)
            x[0] = edges[0]
            x[-1] = edges[-1]
            y[0] = y[-1] = 0
            x[1:-1:2] = edges[:-1]
            x[2:-1:2] = edges[1:]
            y[1:-1:2] = counts
            y[2:-1:2] = counts
            if ori.is_vertical:
                segs.append(np.column_stack([x, y]))
            else:
                segs.append(np.column_stack([y, x]))
        return WrappedLines(
            src, segs, labels, name=name, color=color, width=width, style=style,
            backend=backend,
        )  # fmt: skip

    @property
    def color(self) -> _p.ColorPlan:
        return self._color_by

    @property
    def width(self) -> _p.WidthPlan:
        return self._width_by

    @property
    def style(self) -> _p.StylePlan:
        return self._style_by

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
        self._base_layer.width = width_by.map(self._source)
        self._width_by = width_by
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


class WrappedMarkers(_shared.DataFrameLayerWrapper[_lg.MarkerCollection, _DF]):
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

    def _generate_labels(self):
        pos, labels = self._x.generate_labels(self._source)
        return pos, ["\n".join(lbl) for lbl in labels]

    @property
    def symbol(self) -> _p.SymbolPlan:
        return self._symbol_by

    @property
    def size(self) -> _p.SizePlan:
        return self._size_by

    @property
    def color(self) -> _p.ColorPlan:
        return self._color_by

    @property
    def hatch(self) -> _p.HatchPlan:
        return self._hatch_by

    @property
    def width(self) -> _p.WidthPlan:
        return self._width_by

    @classmethod
    def from_table(
        cls,
        df: _DF,
        x: str,
        y: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        symbol: str | tuple[str, ...] | None = None,
        size: str | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ) -> WrappedMarkers[_DF]:
        src = parse(df)
        xj = _jitter.identity_or_categorical(src, x)
        yj = _jitter.identity_or_categorical(src, y)
        return WrappedMarkers(
            src, xj, yj, name=name, color=color, hatch=hatch, symbol=symbol,
            size=size, backend=backend,
        )  # fmt: skip

    @classmethod
    def build_stripplot(
        cls,
        df: _DF,
        label: str,
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        symbol: str | tuple[str, ...] | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        seed: int | None = 0,
        backend: str | Backend | None = None,
    ) -> WrappedMarkerGroups[_DF]:
        src = parse(df)
        xj = _jitter.UniformJitter(label, extent=extent, seed=seed)
        yj = _jitter.identity_or_categorical(src, value)
        if not Orientation.parse(orient).is_vertical:
            xj, yj = yj, xj
        return WrappedMarkerGroups(
            src, xj, yj, name=name, color=color, hatch=hatch, orient=orient,
            symbol=symbol, size=size, backend=backend,
        )  # fmt: skip

    @classmethod
    def build_swarmplot(
        cls,
        df: _DF,
        label: str,
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        symbol: str | tuple[str, ...] | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        sort: bool = False,
        backend: str | Backend | None = None,
    ) -> WrappedMarkerGroups[_DF]:
        src = parse(df)
        if sort:
            src = src.sort(value)
        lims = src[value].min(), src[value].max()
        xj = _jitter.SwarmJitter(label, value, limits=lims, extent=extent)
        yj = _jitter.identity_or_categorical(src, value)
        if not Orientation.parse(orient).is_vertical:
            xj, yj = yj, xj
        return WrappedMarkerGroups(
            src, xj, yj, name=name, color=color, hatch=hatch, orient=orient,
            symbol=symbol, size=size, backend=backend,
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

    def with_hatch(self, by: str | Iterable[str], choices=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            hatch_by = _p.HatchPlan.new(cov.columns, values=choices)
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


class WrappedMarkerGroups(WrappedMarkers):
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


class WrappedBars(
    _shared.DataFrameLayerWrapper[_l.Bars[_mixin.MultiFace, _mixin.MultiEdge], _DF],
    Generic[_DF],
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        offset: str,
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        if isinstance(offset, str):
            offset = (offset,)
        splitby = _shared.join_columns(offset, color, hatch, source=source)
        unique_sl: list[tuple[Any, ...]] = []
        values = []
        for sl, df in source.group_by(splitby):
            unique_sl.append(sl)
            series = df[value]
            if len(series) != 1:
                raise ValueError(f"More than one value found for category {sl!r}.")
            values.append(series[0])

        self._color_by = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._offset_by = _p.OffsetPlan.default().more_by(*offset)
        self._labels = unique_sl
        self._splitby = splitby

        x = self._offset_by.generate(self._labels, splitby)
        base = _l.Bars(
            x, values, name=name, orient=orient, extent=extent, backend=backend
        ).with_face_multi()
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

    @classmethod
    def from_table(
        cls,
        df: _DF,
        x: str,
        y: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ) -> WrappedBars[_DF]:
        src = parse(df)
        return WrappedBars(
            src, x, y, name=name, color=color, hatch=hatch, extent=extent,
            backend=backend
        )  # fmt: skip

    @classmethod
    def build_count(
        cls,
        df: _DF,
        offset: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ) -> WrappedBars[_DF]:
        src = parse(df)
        splitby = _shared.join_columns(offset, color, hatch, source=src)
        new_src = src.value_count(splitby)
        return WrappedBars(
            new_src, offset, "size", name=name, color=color, hatch=hatch,
            orient=orient, extent=extent, backend=backend
        )  # fmt: skip

    @property
    def color(self) -> _p.ColorPlan:
        """Return the color plan object."""
        return self._color_by

    @property
    def hatch(self) -> _p.HatchPlan:
        """Return the hatch plan object."""
        return self._hatch_by

    def with_color(self, by: str | Iterable[str] | ColorType, palette=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.face.color = color_by.generate(self._labels, self._splitby)
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
        self._base_layer.face.hatch = hatch_by.generate(self._labels, self._splitby)
        self._hatch_by = hatch_by
        return self
