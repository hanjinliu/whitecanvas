"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar, Union, overload

import numpy as np
from cmap import Color

from whitecanvas import layers as _l
from whitecanvas.backend import Backend
from whitecanvas.layers import _mixin
from whitecanvas.layers import group as _lg
from whitecanvas.layers._base import Layer, LayerWrapper
from whitecanvas.layers.tabular import _jitter
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, parse
from whitecanvas.types import (
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    Symbol,
)

if TYPE_CHECKING:
    from typing_extensions import Self

_L = TypeVar("_L", bound="Layer")
_DF = TypeVar("_DF")
_Cols = Union[str, "tuple[str, ...]"]


class DataFrameLayerWrapper(LayerWrapper[_L], Generic[_L, _DF]):
    def __init__(self, base: _L, source: DataFrameWrapper[_DF]):
        super().__init__(base)
        self._source = source

    @property
    def data(self) -> _DF:
        """The internal dataframe."""
        return self._source.get_native()


class WrappedLines(DataFrameLayerWrapper[_lg.LineCollection, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        x: str,
        y: str,
        color: _Cols | None = None,
        width: str | None = None,
        style: _Cols | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        splitby = _concat_by(color, style)
        segs = []
        unique_sl: list[tuple[Any, ...]] = []
        for sl, df in source.group_by(splitby):
            unique_sl.append(sl)
            segs.append(np.column_stack([df[x], df[y]]))

        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = unique_sl
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
        return WrappedLines(
            src,
            x,
            y,
            name=name,
            color=color,
            width=width,
            style=style,
            backend=backend,
        )

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
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = self._color_by.update(*cov.columns, values=palette)
        else:
            color_by = self._color_by.with_const(Color(cov.value))
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
            width_by = self._width_by.with_range(by, limits=limits)
        else:
            width_by = self._width_by.with_const(float(by))
        self._base_layer.width = width_by.map(self._source)
        self._width_by = width_by
        return self

    def with_style(self, by: str | Iterable[str], styles=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = self._style_by.update(*cov.columns, values=styles)
        else:
            style_by = self._style_by.with_const(LineStyle(cov.value))
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


def _unique_tuple(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    b_filt = tuple(x for x in b if x not in a)
    return a + b_filt


class WrappedViolinPlot(DataFrameLayerWrapper[_lg.ViolinPlot, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        offset: str | tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        shape: str = "both",
        backend: str | Backend | None = None,
    ):
        if isinstance(offset, str):
            offset = (offset,)
        splitby = _concat_by(offset, color, hatch)
        self._y = value
        self._splitby = splitby
        self._color_by = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._offset_by = _p.OffsetPlan.default().more_by(*offset)
        self._source = source
        arrays, self._labels = self._generate_datasets(splitby)
        x = self._offset_by.generate(self._labels, self._splitby)
        base = _lg.ViolinPlot.from_arrays(
            x, arrays, name=name, orient=orient, shape=shape, extent=extent,
            backend=backend,
        )  # fmt: skip
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

    def _generate_labels(self):
        pos: list[float] = []
        labels: list[str] = []
        for p, lbl in self._offset_by.iter_ticks(self._labels, self._splitby):
            pos.append(p)
            labels.append("\n".join(lbl))
        return pos, labels

    @classmethod
    def from_table(
        cls,
        df: _DF,
        offset: tuple[str, ...],
        value: str,
        color: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        shape: str = "both",
        backend: str | Backend | None = None,
    ) -> WrappedViolinPlot[_DF]:
        src = parse(df)
        self = WrappedViolinPlot(
            src, offset, value, orient=orient, name=name, extent=extent,
            color=color, hatch=hatch, shape=shape, backend=backend
        )  # fmt: skip
        return self

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._base_layer.orient

    @property
    def color(self) -> _p.ColorPlan:
        return self._color_by

    @property
    def hatch(self) -> _p.HatchPlan:
        return self._hatch_by

    def _generate_datasets(
        self,
        by_all: tuple[str, ...],
    ) -> tuple[list[np.ndarray], list[tuple[Any, ...]]]:
        datasets = []
        unique_sl: list[tuple[Any, ...]] = []
        for sl, df in self._source.group_by(by_all):
            unique_sl.append(sl)
            datasets.append(df[self._y])
        return datasets, unique_sl

    def with_color(self, by: str | Iterable[str], palette=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            by_all = _unique_tuple(cov.columns, self._hatch_by.by)
            color_by = self._color_by.update(*cov.columns, values=palette)
            _, self._labels = self._generate_datasets(by_all)
            self._splitby = by_all
        else:
            color_by = self._color_by.with_const(Color(cov.value))
        self._base_layer.face.color = color_by.generate(self._labels, self._splitby)
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str], choices=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            by_all = _unique_tuple(self._color_by.by, cov.columns)
            hatch_by = self._hatch_by.update(*cov.columns, values=choices)
            _, self._labels = self._generate_datasets(by_all)
            self._splitby = by_all
        else:
            hatch_by = self._hatch_by.with_const(Hatch(cov.value))
        self._base_layer.face.hatch = hatch_by.generate(self._labels, self._splitby)
        self._hatch_by = hatch_by
        return self

    def with_shift(
        self,
        shift: float = 0.0,
    ) -> Self:
        for layer in self._base_layer:
            _old = layer.data
            layer.set_data(edge_low=_old.y0 + shift, edge_high=_old.y1 + shift)
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Self:
        self._base_layer.with_edge(color=color, width=width, style=style, alpha=alpha)
        return self


class WrappedBoxPlot(DataFrameLayerWrapper[_lg.BoxPlot, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        offset: str | tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ):
        if isinstance(offset, str):
            offset = (offset,)
        splitby = _concat_by(offset, color, hatch)
        self._y = value
        self._splitby = splitby
        self._color_by = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._offset_by = _p.OffsetPlan.default().more_by(*offset)
        self._source = source
        arrays, self._labels = self._generate_datasets(splitby)
        x = self._offset_by.generate(self._labels, self._splitby)
        base = _lg.BoxPlot.from_arrays(
            x,
            arrays,
            name=name,
            orient=orient,
            capsize=capsize,
            extent=extent,
            backend=backend,
        )
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

    def _generate_labels(self):
        pos: list[float] = []
        labels: list[str] = []
        for p, lbl in self._offset_by.iter_ticks(self._labels, self._splitby):
            pos.append(p)
            labels.append("\n".join(lbl))
        return pos, labels

    @classmethod
    def from_table(
        cls,
        df: _DF,
        offset: tuple[str, ...],
        value: str,
        color: str | None = None,
        hatch: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ) -> WrappedBoxPlot[_DF]:
        src = parse(df)
        self = WrappedBoxPlot(
            src, offset, value, orient=orient, name=name, color=color, hatch=hatch,
            capsize=capsize, extent=extent, backend=backend
        )  # fmt: skip
        return self

    @property
    def orient(self) -> Orientation:
        """Orientation of the violins."""
        return self._base_layer.orient

    @property
    def color(self) -> _p.ColorPlan:
        return self._color_by

    @property
    def hatch(self) -> _p.HatchPlan:
        return self._hatch_by

    def _generate_datasets(
        self,
        by_all: tuple[str, ...],
    ) -> tuple[list[np.ndarray], list[tuple[Any, ...]]]:
        datasets = []
        unique_sl: list[tuple[Any, ...]] = []
        for sl, df in self._source.group_by(by_all):
            unique_sl.append(sl)
            datasets.append(df[self._y])
        return datasets, unique_sl

    def with_color(self, by: str | Iterable[str], palette=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            by_all = _unique_tuple(cov.columns, self._hatch_by.by)
            color_by = self._color_by.update(*cov.columns, values=palette)
            _, self._labels = self._generate_datasets(by_all)
            self._splitby = by_all
        else:
            color_by = self._color_by.with_const(Color(cov.value))
        colors = color_by.generate(self._labels, self._splitby)
        self._base_layer.boxes.with_face_multi(color=colors)
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str], choices=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            by_all = _unique_tuple(self._color_by.by, cov.columns)
            hatch_by = self._hatch_by.update(*cov.columns, values=choices)
            _, self._labels = self._generate_datasets(by_all)
            self._splitby = by_all
        else:
            hatch_by = self._hatch_by.with_const(Hatch(cov.value))
            hatches = hatch_by.generate(self._labels, self._splitby)
        self._base_layer.boxes.with_face_multi(hatch=hatches)
        self._hatch_by = hatch_by
        return self

    def with_shift(
        self,
        shift: float = 0.0,
    ) -> Self:
        self._base_layer.with_shift(shift)
        return self


class WrappedMarkers(
    DataFrameLayerWrapper[
        _lg.MarkerCollection,
        _DF,
    ],
    Generic[_DF],
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
        self._color_by = _p.ColorPlan.default()
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

    def with_color(self, by: str | Iterable[str], palette=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            color_by = self._color_by.update(*cov.columns, values=palette)
        else:
            color_by = self._color_by.with_const(Color(cov.value))
        colors = color_by.map(self._source)
        self._base_layer.face.color = colors
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str], choices=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            hatch_by = self._hatch_by.update(*cov.columns, values=choices)
        else:
            hatch_by = self._hatch_by.with_const(Hatch(cov.value))
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
            size_by = self._size_by.with_range(by, limits=limits)
        else:
            size_by = self._size_by.with_const(float(by))
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
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            symbol_by = self._symbol_by.update(*cov.columns, values=symbols)
        else:
            symbol_by = self._symbol_by.with_const(Symbol(by))
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
            width_by = self._width_by.with_range(by, limits=limits)
        else:
            const_size = float(by)
            width_by = self._width_by.with_const(const_size)
        self._base_layer.with_edge(width=width_by.map(self._source))
        self._width_by = width_by
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> Self:
        self._base_layer.with_edge(color=color, style=style, width=width, alpha=alpha)
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
    DataFrameLayerWrapper[_l.Bars[_mixin.MultiFace, _mixin.MultiEdge], _DF],
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
        splitby = _concat_by(offset, color, hatch)
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
            x, values, name=name, orient=orient, bar_width=extent, backend=backend
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
        splitby = _concat_by(offset, color, hatch)
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
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = self._color_by.update(*cov.columns, values=palette)
        else:
            color_by = self._color_by.with_const(Color(cov.value))
        self._base_layer.face.color = color_by.generate(self._labels, self._splitby)
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str], choices=None) -> Self:
        cov = ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot hatch by a column other than {self._splitby}")
            hatch_by = self._hatch_by.update(*cov.columns, values=choices)
        else:
            hatch_by = self._hatch_by.with_const(Color(cov.value))
        self._base_layer.face.hatch = hatch_by.generate(self._labels, self._splitby)
        self._hatch_by = hatch_by
        return self


# class WrappedPointPlot


def _concat_by(*args: str | tuple[str] | None) -> tuple[str, ...]:
    """
    Concatenate the given arguments into a tuple of strings.

    >>> _concat_by("a", "b", "c")  # ("a", "b", "c")
    >>> _concat_by("a", ("b", "c"))  # ("a", "b", "c")
    >>> _concat_by("a", None, "c")  # ("a", "c")
    """
    by_all: list[str] = []
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, str):
            if arg not in by_all:
                by_all.append(arg)
        else:
            for a in arg:
                if a not in by_all:
                    by_all.append(a)
    return tuple(by_all)


class ColumnOrValue:
    def __init__(self, by, df: DataFrameWrapper[_DF]):
        if isinstance(by, str):
            if by in df.iter_keys():
                self._is_columns = True
                self._value = (by,)
            else:
                self._is_columns = False
                self._value = by
        elif hasattr(by, "__iter__"):
            self._is_columns = all(isinstance(each, str) for each in by)
            if self._is_columns:
                columns = set(df.iter_keys())
                for each in by:
                    if each not in columns:
                        raise ValueError(f"{each!r} is not a column.")
                self._value = tuple(by)
            else:
                self._value = by

    @property
    def is_column(self) -> bool:
        """True if the value is column name(s)."""
        return self._is_columns

    @property
    def value(self) -> Any:
        """Return the value."""
        return self._value

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the column name(s)."""
        if self._is_columns:
            return self._value
        else:
            raise ValueError("The value is not a column name(s).")
