from __future__ import annotations

import weakref
from typing import (
    TYPE_CHECKING,
    Generic,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np

from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.canvas.dataframe._utils import PlotArg
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular._dataframe import parse
from whitecanvas.types import (
    ColorType,
    Hatch,
    Orientation,
    Symbol,
)

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.layers.tabular._dataframe import DataFrameWrapper

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")
NStr = Union[str, Sequence[str]]
_Orientation = Union[str, Orientation]


class _Plotter(Generic[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
        update_label: bool = False,
    ):
        self._canvas_ref = weakref.ref(canvas)
        self._df = df
        self._update_label = update_label

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    def _update_xy_label(
        self,
        x: str | tuple[str, ...],
        y: str | tuple[str, ...],
        orient: Orientation = Orientation.VERTICAL,
    ) -> None:
        canvas = self._canvas()
        if not isinstance(x, str):
            x = "/".join(x)
        if not isinstance(y, str):
            y = "/".join(y)
        if orient.is_vertical:
            canvas.x.label.text = x
            canvas.y.label.text = y
        else:
            canvas.x.label.text = y
            canvas.y.label.text = x

    def _update_xy_ticks(self, pos, label, orient: Orientation = Orientation.VERTICAL):
        canvas = self._canvas()
        if orient.is_vertical:
            canvas.x.ticks.set_labels(pos, label)
        else:
            canvas.y.ticks.set_labels(pos, label)


class DataFramePlotter(_Plotter[_C, _DF]):
    def add_line(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ) -> _lt.WrappedLines[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedLines.from_table(
            self._df, x, y, name=name, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            self._update_xy_label(x, y)
        return canvas.add_layer(layer)

    def add_markers(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: NStr | None = None,
        hatch: NStr | None = None,
        size: str | None = None,
        symbol: NStr | None = None,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.from_table(
            self._df, x, y, name=name, color=color, hatch=hatch, size=size,
            symbol=symbol, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            self._update_xy_label(x, y)
        return canvas.add_layer(layer)

    def add_bar(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: NStr | None = None,
        hatch: NStr | None = None,
        extent: float = 0.8,
    ) -> _lt.WrappedBars[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedBars.from_table(
            self._df, x, y, name=name, color=color, hatch=hatch, extent=extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            self._update_xy_label(x, y)
        return canvas.add_layer(layer)

    def add_violinplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        extent: float = 0.8,
        shape: str = "both",
        name: str | None = None,
        orient: _Orientation = Orientation.VERTICAL,
    ) -> _lt.WrappedViolinPlot[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedViolinPlot.from_table(
            self._df, offset, value, name=name, color=color, hatch=hatch, extent=extent,
            shape=shape, orient=orient, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            pos, labels = layer._generate_labels()
            self._update_xy_ticks(pos, labels, orient=orient)
            self._update_xy_label(offset, value, orient=orient)

        return canvas.add_layer(layer)

    def add_boxplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        name: str | None = None,
        orient: _Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
        extent: float = 0.8,
    ) -> _lt.WrappedBoxPlot[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedBoxPlot.from_table(
            self._df, offset, value, name=name, color=color, hatch=hatch, orient=orient,
            capsize=capsize, extent=extent, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            pos, labels = layer._generate_labels()
            self._update_xy_ticks(pos, labels, orient=orient)
            self._update_xy_label(offset, value, orient=orient)

        return canvas.add_layer(layer)

    def add_stripplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        symbol: NStr | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: _Orientation = Orientation.VERTICAL,
        extent: float = 0.5,
        seed: int | None = 0,
    ) -> _lt.WrappedMarkerGroups[_DF]:
        canvas = self._canvas()
        orient = Orientation.parse(orient)
        layer = _lt.WrappedMarkers.build_stripplot(
            self._df, offset, value, name=name, color=color, hatch=hatch, symbol=symbol,
            size=size, orient=orient, extent=extent, seed=seed,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            pos, labels = layer._generate_labels()
            self._update_xy_ticks(pos, labels, orient=orient)
            self._update_xy_label(offset, value, orient=orient)
        return canvas.add_layer(layer)

    def add_swarmplot(
        self,
        offset: NStr,
        value: str,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        symbol: NStr | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: _Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        sort: bool = False,
    ) -> _lt.WrappedMarkerGroups[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.build_swarmplot(
            self._df, offset, value, name=name, color=color, hatch=hatch, symbol=symbol,
            size=size, orient=orient, extent=extent, sort=sort,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            pos, labels = layer._generate_labels()
            self._update_xy_ticks(pos, labels, orient=orient)
            self._update_xy_label(offset, value, orient=orient)
        return canvas.add_layer(layer)

    def add_countplot(
        self,
        offset: NStr,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        name: str | None = None,
        orient: _Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
    ) -> _lt.WrappedBars[_DF]:
        canvas = self._canvas()
        orient = Orientation.parse(orient)
        layer = _lt.WrappedBars.build_count(
            self._df, offset, color=color, hatch=hatch, orient=orient, extent=extent,
            name=name, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        if self._update_label:
            self._update_xy_label(offset, "count", orient=orient)
        return canvas.add_layer(layer)

    def mean(self, orient: _Orientation = "vertical") -> DataFrameAggPlotter[_C, _DF]:
        """Return a mean-plotter."""
        return self._agg_plotter("mean", orient)

    def std(self, orient: _Orientation = "vertical") -> DataFrameAggPlotter[_C, _DF]:
        """Return a std-plotter."""
        return self._agg_plotter("std", orient)

    def median(self, orient: _Orientation = "vertical") -> DataFrameAggPlotter[_C, _DF]:
        """Return a median-plotter."""
        return self._agg_plotter("median", orient)

    def min(self, orient: _Orientation = "vertical") -> DataFrameAggPlotter[_C, _DF]:
        """Return a min-plotter."""
        return self._agg_plotter("min", orient)

    def max(self, orient: _Orientation = "vertical") -> DataFrameAggPlotter[_C, _DF]:
        """Return a max-plotter."""
        return self._agg_plotter("max", orient)

    def sum(self, orient: _Orientation = "vertical") -> DataFrameAggPlotter[_C, _DF]:
        """Return a sum-plotter."""
        return self._agg_plotter("sum", orient)

    def _agg_plotter(
        self,
        method: str,
        orient: str | Orientation,
    ) -> DataFrameAggPlotter[_C, _DF]:
        return DataFrameAggPlotter(
            self._canvas(),
            self._df,
            self._update_label,
            method=method,
            orient=Orientation.parse(orient),
        )


class DataFrameAggPlotter(_Plotter[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
        update_label: bool,
        method: str,
        orient: Orientation,
    ):
        super().__init__(canvas, df, update_label)
        self._agg_method = method
        self._orient = orient

    def add_line(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ) -> _lt.WrappedLines[_DF]:
        canvas = self._canvas()
        df = parse(self._df)
        keys = list(df.iter_keys())
        _color = PlotArg.from_color(keys, color)
        _style = PlotArg.from_style(keys, style)
        df_agg = self._aggregate(df, self._concat_tuple(x, y, _color, _style), y)
        layer = _lt.WrappedLines.from_table(
            df_agg, x, y, name=name, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(_color.value, palette=canvas._color_palette)
        if self._update_label:
            self._update_xy_label(x, y)
        return canvas.add_layer(layer)

    def add_markers(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: NStr | ColorType | None = None,
        hatch: NStr | Hatch | None = None,
        size: np.str_ | float | None = None,
        symbol: NStr | Symbol | None = None,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        df = parse(self._df)
        keys = list(df.iter_keys())
        _color = PlotArg.from_color(keys, color)
        _hatch = PlotArg.from_hatch(keys, hatch)
        _symbol = PlotArg.from_symbol(keys, symbol)
        df_agg = self._aggregate(
            df, self._concat_tuple(x, y, _color, _hatch, _symbol), y
        )
        layer = _lt.WrappedMarkers.from_table(
            df_agg, x, y, name=name, color=color, hatch=hatch, size=size,
            symbol=symbol, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(_color.value, palette=canvas._color_palette)
        if self._update_label:
            self._update_xy_label(x, y)
        return canvas.add_layer(layer)

    def _aggregate(
        self,
        df: DataFrameWrapper,
        by: tuple[str, ...],
        on: str,
    ) -> DataFrameWrapper[_DF]:
        return df.agg_by(by, on, self._agg_method)

    def _concat_tuple(self, x, y, *args: PlotArg) -> tuple:
        """
        Concatenate the arguments into a tuple.

        This method may return a tuple of str or other types such as Symbol, Color, etc.
        """
        out = []
        if self._orient.is_vertical:
            out.append(x)
        else:
            out.append(y)
        for a in args:
            if not a.is_column:
                continue
            elif isinstance(val := a.value, str):
                out.append(val)
            else:
                out.extend(val)
        return out
