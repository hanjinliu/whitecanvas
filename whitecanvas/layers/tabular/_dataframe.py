"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, TypeVar, Generic, Union, overload

import numpy as np
from numpy.typing import NDArray
from whitecanvas.layers._base import LayerWrapper, Layer
from whitecanvas import layers as _l
from whitecanvas.layers import group as _lg, _mixin
from whitecanvas.types import Orientation, Symbol
from whitecanvas.backend import Backend

from . import _plans as _p, _jitter
from ._df_compat import DataFrameWrapper, parse

if TYPE_CHECKING:
    from whitecanvas.canvas import Canvas
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

    def with_color(self, by: str | Iterable[str] | None = None, palette=None) -> Self:
        if by is None:
            by = self._color_by.by
        elif isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        if set(by) > set(self._splitby):
            raise ValueError(f"Cannot color by a column other than {self._splitby}")
        color_by = self._color_by.update(*by, values=palette)
        self._base_layer.color = color_by.generate(self._labels, self._splitby)
        self._color_by = color_by
        return self

    def with_width(self, by: str, limits=None):
        width_by = self._width_by.with_range(by, limits=limits)
        self._base_layer.width = width_by.map(self._source)
        self._width_by = width_by
        return self

    def with_style(self, by: str | Iterable[str] | None = None, styles=None) -> Self:
        if by is None:
            if styles is None:
                raise ValueError("Either `by` or `styles` should be given.")
            self._style_by = self._style_by.with_choices(styles)
            self._base_layer.color = self._style_by.generate(
                self._labels, self._splitby
            )
            return self
        if isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        if set(by) > set(self._splitby):
            raise ValueError(f"Cannot color by a column other than {self._splitby}")
        style_by = self._style_by.update(*by, values=styles)
        self._base_layer.style = style_by.generate(self._labels, self._splitby)
        self._style_by = style_by
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
        backend: str | Backend | None = None,
    ):
        if isinstance(offset, str):
            offset = (offset,)
        splitby = _concat_by(offset, color, hatch)
        unique_sl: list[tuple[Any, ...]] = []
        arrays = []
        for sl, df in source.group_by(splitby):
            unique_sl.append(sl)
            arrays.append(df[value])
        self._y = value
        self._color_by = _p.ColorPlan.default()
        self._hatch_by = _p.HatchPlan.default()
        self._offset_by = _p.OffsetPlan.default().more_by(*offset)
        self._labels = unique_sl
        x = self._offset_by.generate(self._labels, splitby)
        base = _lg.ViolinPlot.from_arrays(
            x,
            arrays,
            name=name,
            orient=orient,
            backend=backend,
        )
        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)

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
        backend: str | Backend | None = None,
    ) -> WrappedViolinPlot[_DF]:
        src = parse(df)
        self = WrappedViolinPlot(
            src, offset, value, orient=orient, name=name,
            color=color, hatch=hatch, backend=backend
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

    def with_color(self, by: str | Iterable[str] | None = None, palette=None) -> Self:
        if by is None:
            by = self._color_by.by
        elif isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        by_all = _unique_tuple(by, self._hatch_by.by)
        color_by = self._color_by.update(*by, values=palette)
        segs, unique_sl = self._generate_datasets(by_all)
        # self._base_layer.data = segs
        self._base_layer.face.color = color_by.generate(unique_sl, by_all)
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str], choices=None) -> Self:
        if isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        by_all = _unique_tuple(self._color_by.by, by)
        hatch_by = self._hatch_by.update(*by, values=choices)
        segs, unique_sl = self._generate_datasets(by_all)
        self._base_layer.face.hatch = hatch_by.generate(unique_sl, by_all)
        self._hatch_by = hatch_by
        return self


class WrappedMarkers(
    DataFrameLayerWrapper[
        _l.Markers[_mixin.MultiFace, _mixin.MultiEdge, NDArray[np.floating]], _DF
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

        base = _l.Markers(x.map(source), y.map(source), name=name, backend=backend)

        super().__init__(base, source)
        if color is not None:
            self.with_color(color)
        if hatch is not None:
            self.with_hatch(hatch)
        if symbol is not None:
            self.with_symbol(symbol)
        if size is not None:
            self.with_size(size)

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
    ) -> WrappedMarkers[_DF]:
        src = parse(df)
        xj = _jitter.UniformJitter(label, extent=extent, seed=seed)
        yj = _jitter.identity_or_categorical(src, value)
        if not Orientation.parse(orient).is_vertical:
            xj, yj = yj, xj
        return WrappedMarkers(
            src, xj, yj, name=name, color=color, hatch=hatch,
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
    ) -> WrappedMarkers[_DF]:
        src = parse(df)
        if sort:
            src = src.sort(value)
        lims = src[value].min(), src[value].max()
        xj = _jitter.SwarmJitter(label, value, limits=lims, extent=extent)
        yj = _jitter.identity_or_categorical(src, value)
        if not Orientation.parse(orient).is_vertical:
            xj, yj = yj, xj
        return WrappedMarkers(
            src, xj, yj, name=name, color=color, hatch=hatch,
            symbol=symbol, size=size, backend=backend,
        )  # fmt: skip

    def with_color(self, by: str | Iterable[str] | None = None, palette=None) -> Self:
        if by is None:
            by = self._color_by.by
        elif isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        color_by = self._color_by.update(*by, values=palette)
        self._base_layer.with_face_multi(color=color_by.map(self._source))
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str] | None = None, choices=None) -> Self:
        if by is None:
            by = self._hatch_by.by
        elif isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        hatch_by = self._hatch_by.update(*by, values=choices)
        self._base_layer.with_face_multi(hatch=hatch_by.map(self._source))
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
            const_size = float(by)
            size_by = self._size_by.with_const(const_size)
        self._base_layer.with_size_multi(size_by.map(self._source))
        self._size_by = size_by
        return self

    @overload
    def with_symbol(self, value: str | Symbol) -> Self:
        ...

    @overload
    def with_symbol(self, by: str | Iterable[str] | None = None, symbols=None) -> Self:
        ...

    def with_symbol(self, by, /, symbols=None) -> Self:
        if isinstance(by, str):
            if by in self._source.iter_keys():
                symbol_by = self._symbol_by.update(by, values=symbols)
            else:
                try:
                    const_symbol = Symbol(by)
                except ValueError:
                    raise ValueError(
                        f"{by!r} does not exist either as a column or a symbol name."
                    )
                symbol_by = self._symbol_by.with_const(const_symbol)
        elif isinstance(by, Symbol):
            symbol_by = self._symbol_by.with_const(by)
        else:
            symbol_by = self._symbol_by.update(*by, values=symbols)
        self._base_layer.symbol = symbol_by.map(self._source)
        self._symbol_by = symbol_by
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
        )
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
        new_src = src.agg_by(splitby, on=splitby[0], method="size")
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

    def with_color(self, by: str | Iterable[str] | None = None, palette=None) -> Self:
        if by is None:
            by = self._color_by.by
        elif isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        if set(by) > set(self._splitby):
            raise ValueError(f"Cannot color by a column other than {self._splitby}")
        color_by = self._color_by.update(*by, values=palette)
        self._base_layer.face.color = color_by.generate(self._labels, self._splitby)
        self._color_by = color_by
        return self

    def with_hatch(self, by: str | Iterable[str] | None = None, choices=None) -> Self:
        if by is None:
            by = self._hatch_by.by
        elif isinstance(by, str):
            by = (by,)
        else:
            by = tuple(by)
        if set(by) > set(self._splitby):
            raise ValueError(f"Cannot color by a column other than {self._splitby}")
        hatch_by = self._hatch_by.update(*by, values=choices)
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
