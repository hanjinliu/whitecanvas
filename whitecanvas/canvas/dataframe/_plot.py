from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    Sequence,
    TypeVar,
    Union,
)
import weakref
from cmap import Color
import numpy as np
from numpy.typing import NDArray

from whitecanvas import layers as _l
from whitecanvas.layers import group as _lg
from whitecanvas.types import (
    LineStyle,
    Symbol,
    ColorType,
    ColormapType,
    Hatch,
    Orientation,
)
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.layers import tabular as _lt
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.layers.tabular._dataframe import parse
from ._utils import PlotArg

if TYPE_CHECKING:
    from typing_extensions import Self
    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.layers.tabular._dataframe import DataFrameWrapper

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")
nStr = Union[str, Sequence[str]]


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


class DataFramePlotter(_Plotter[_C, _DF]):
    def add_line(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: nStr | None = None,
        width: str | None = None,
        style: nStr | None = None,
    ) -> _lt.WrappedLines[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedLines.from_table(
            self._df, x, y, name=name, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            canvas.x.label.text = x
            canvas.y.label.text = y
        return canvas.add_layer(layer)

    def add_markers(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: nStr | None = None,
        hatch: nStr | None = None,
        size: str | None = None,
        symbol: nStr | None = None,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.from_table(
            self._df, x, y, name=name, color=color, hatch=hatch, size=size,
            symbol=symbol, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            canvas.x.label.text = x
            canvas.y.label.text = y
        return canvas.add_layer(layer)

    def add_bar(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: nStr | None = None,
        hatch: nStr | None = None,
        extent: float = 0.8,
    ) -> _lt.WrappedBars[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedBars.from_table(
            self._df, x, y, name=name, color=color, hatch=hatch, extent=extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            canvas.x.label.text = x
            canvas.y.label.text = y
        return canvas.add_layer(layer)

    def add_violinplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: nStr | None = None,
        hatch: nStr | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
    ) -> _lt.WrappedViolinPlot[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedViolinPlot(
            self._df, offset, value, name=name, color=color, hatch=hatch,
            orient=orient, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            ...
        return canvas.add_layer(layer)

    def add_stripplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: nStr | None = None,
        hatch: nStr | None = None,
        symbol: nStr | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.5,
        seed: int | None = 0,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.build_stripplot(
            self._df, offset, value, name=name, color=color, hatch=hatch, symbol=symbol,
            size=size, orient=orient, extent=extent, seed=seed,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            ...
        return canvas.add_layer(layer)

    def add_swarmplot(
        self,
        offset: nStr,
        value: str,
        *,
        color: nStr | None = None,
        hatch: nStr | None = None,
        symbol: nStr | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        sort: bool = False,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.build_swarmplot(
            self._df, offset, value, name=name, color=color, hatch=hatch, symbol=symbol,
            size=size, orient=orient, extent=extent, sort=sort,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            ...
        return canvas.add_layer(layer)

    def add_countplot(
        self,
        offset: nStr,
        *,
        color: nStr | None = None,
        hatch: nStr | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
    ) -> _lt.WrappedBars[_DF]:
        canvas = self._canvas()

        layer = _lt.WrappedBars.build_count(
            self._df,
            offset,
            color=color,
            hatch=hatch,
            orient=orient,
            extent=extent,
            name=name,
            backend=canvas._get_backend(),
        )
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        return canvas.add_layer(layer)

    def mean(self, orient: str | Orientation) -> DataFrameAggPlotter[_C, _DF]:
        """Return a mean-plotter."""
        return self._agg_plotter("mean", orient)

    def std(self, orient: str | Orientation) -> DataFrameAggPlotter[_C, _DF]:
        """Return a std-plotter."""
        return self._agg_plotter("std", orient)

    def median(self, orient: str | Orientation) -> DataFrameAggPlotter[_C, _DF]:
        """Return a median-plotter."""
        return self._agg_plotter("median", orient)

    def min(self, orient: str | Orientation) -> DataFrameAggPlotter[_C, _DF]:
        """Return a min-plotter."""
        return self._agg_plotter("min", orient)

    def max(self, orient: str | Orientation) -> DataFrameAggPlotter[_C, _DF]:
        """Return a max-plotter."""
        return self._agg_plotter("max", orient)

    def sum(self, orient: str | Orientation) -> DataFrameAggPlotter[_C, _DF]:
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
        color: nStr | None = None,
        width: str | None = None,
        style: nStr | None = None,
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
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            canvas.x.label.text = x
            canvas.y.label.text = y
        return canvas.add_layer(layer)

    def add_markers(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: nStr | ColorType | None = None,
        hatch: nStr | Hatch | None = None,
        size: np.str_ | float | None = None,
        symbol: nStr | Symbol | None = None,
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
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            canvas.x.label.text = x
            canvas.y.label.text = y
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
