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

from whitecanvas import theme
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.canvas.dataframe._utils import PlotArg
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular._dataframe import parse
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    Hatch,
    Orientation,
    Symbol,
)

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.layers.tabular._box_like import _BoxLikeMixin
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
        """Update the x and y labels using the column names"""
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
        """Update the x or y ticks to categorical ticks"""
        canvas = self._canvas()
        if orient.is_vertical:
            canvas.x.ticks.set_labels(pos, label)
        else:
            canvas.y.ticks.set_labels(pos, label)


class DataFramePlotter(_Plotter[_C, _DF]):
    ### 0-D categorical ###
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
        """
        Add a categorical line plot.

        >>> ### Use "time" column as x-axis and "value" column as y-axis
        >>> canvas.cat(df).add_line("time", "value")

        >>> ### Multiple lines colored by column "group"
        >>> canvas.cat(df).add_line("time", "value", color="group")

        >>> ### Multiple lines styled by column "group"
        >>> canvas.cat(df).add_line("time", "value", style="group")

        Parameters
        ----------
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        width : str, optional
            Column name for line width. Must be numerical.
        style : str or sequence of str, optional
            Column name(s) for styling the lines. Must be categorical.

        Returns
        -------
        WrappedLines
            Line collection layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedLines.from_table(
            self._df, x, y, name=name, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
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
        """
        Add a categorical marker plot.

        >>> ### Use "time" column as x-axis and "value" column as y-axis
        >>> canvas.cat(df).add_markers("time", "value")

        >>> ### Multiple markers colored by column "group"
        >>> canvas.cat(df).add_markers("time", "value", color="group")

        >>> ### Multiple markers with hatches determined by column "group"
        >>> canvas.cat(df).add_markers("time", "value", style="group")

        >>> ### Multiple markers with symbols determined by "group"
        >>> canvas.cat(df).add_markers("time", "value", symbol="group")

        Parameters
        ----------
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        size : str, optional
            Column name for marker size. Must be numerical.
        symbol : str or sequence of str, optional
            Column name(s) for symbols. Must be categorical.

        Returns
        -------
        WrappedMarkers
            Marker collection layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.from_table(
            self._df, x, y, name=name, color=color, hatch=hatch, size=size,
            symbol=symbol, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
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
        """
        Add a categorical bar plot.

        >>> ### Use "time" column as x-axis and "value" column as y-axis
        >>> canvas.cat(df).add_bar("time", "value")

        >>> ### Multiple bars colored by column "group"
        >>> canvas.cat(df).add_bar("time", "value", color="group")

        >>> ### Multiple bars with hatches determined by column "group"
        >>> canvas.cat(df).add_bar("time", "value", hatch="group")

        Parameters
        ----------
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        extent : float, optional
            Width of the bars. Usually in range (0, 1].

        Returns
        -------
        WrappedBars
            Bar collection layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedBars.from_table(
            self._df, x, y, name=name, color=color, hatch=hatch, extent=extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        if self._update_label:
            self._update_xy_label(x, y)
        return canvas.add_layer(layer)

    def add_line_hist(
        self,
        x: str,
        *,
        bins: int | ArrayLike1D = 10,
        range: tuple[float, float] | None = None,
        density: bool = False,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ):
        """
        Add lines representing histograms.

        >>> ### Use "value" column as x-axis
        >>> canvas.cat(df).add_line_hist("value", bins=8, density=True)

        >>> ### Multiple histograms colored by column "group"
        >>> canvas.cat(df).add_line_hist("value", color="group")

        Parameters
        ----------
        x : str
            Column name for x-axis.
        bins : int or array-like, default 10
            If an integer, the number of bins. If an array, the bin edges.
        range : (float, float), default None
            If provided, the lower and upper range of the bins.
        density : bool, default False
            If True, the total area of the histogram will be normalized to 1.
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        width : str, optional
            Column name for line width. Must be numerical.
        style : str or sequence of str, optional
            Column name(s) for styling the lines. Must be categorical.

        Returns
        -------
        WrappedLines
            Line collection layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedLines.build_hist(
            self._df, x, bins=bins, range=range, density=density, name=name,
            orient=orient, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        if self._update_label:
            y = "density" if density else "count"
            self._update_xy_label(x, y, orient)
        return canvas.add_layer(layer)

    def add_kde(
        self,
        value: str,
        *,
        band_width: float | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ):
        """
        Add lines representing kernel density estimation.

        >>> ### Use "value" column as x-axis
        >>> canvas.cat(df).add_kde("value")

        >>> ### Multiple KDEs colored by column "group"
        >>> canvas.cat(df).add_kde("value", color="group")

        Parameters
        ----------
        value : str
            Column name for x-axis.
        band_width : float, default None
            Bandwidth of the kernel density estimation. If None, use Scott's rule.
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        width : str, optional
            Column name for line width. Must be numerical.
        style : str or sequence of str, optional
            Column name(s) for styling the lines. Must be categorical.

        Returns
        -------
        WrappedLines
            Line collection layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedLines.build_kde(
            self._df, value, band_width=band_width, name=name,
            orient=orient, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        if self._update_label:
            self._update_xy_label(value, "density", orient)
        return canvas.add_layer(layer)

    ### 1-D categorical ###

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
        """
        Add a categorical violin plot.

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_violinplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_violinplot(offset, "weight", color="region")

        Parameters
        ----------
        offset : tuple of str
            Column name(s) for x-axis.
        value : str
            Column name for y-axis.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        extent : float, default 0.8
            Width of the violins. Usually in range (0, 1].
        shape : str, default "both"
            Shape of the violins. Can be "both", "left", or "right".
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".

        Returns
        -------
        WrappedViolinPlot
            Violin plot layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedViolinPlot.from_table(
            self._df, offset, value, name=name, color=color, hatch=hatch, extent=extent,
            shape=shape, orient=orient, backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color, orient, value)
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
        """
        Add a categorical box plot.

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_boxplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_boxplot(offset, "weight", color="region")

        Parameters
        ----------
        offset : tuple of str
            Column name(s) for x-axis.
        value : str
            Column name for y-axis.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".
        capsize : float, default 0.1
            Length of the caps as a fraction of the width of the box.
        extent : float, default 0.8
            Width of the violins. Usually in range (0, 1].

        Returns
        -------
        WrappedBoxPlot
            Box plot layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedBoxPlot.from_table(
            self._df, offset, value, name=name, color=color, hatch=hatch, orient=orient,
            capsize=capsize, extent=extent, backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color, orient, value)
        return canvas.add_layer(layer)

    def add_pointplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        name: str | None = None,
        orient: _Orientation = Orientation.VERTICAL,
        capsize: float = 0.1,
    ) -> _lt.WrappedPointPlot[_DF]:
        """
        Add a categorical point plot (markers with error bars).

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_pointplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_pointplot(offset, "weight", color="region")

        The default estimator and errors are mean and standard deviation. To change
        them, use `est_by_*` and `err_by_*` methods.

        >>> ### Use standard error x 2 (~95%) as error bars.
        >>> canvas.cat(df).add_pointplot("species", "weight").err_by_se(scale=2.0)

        Parameters
        ----------
        offset : tuple of str
            Column name(s) for x-axis.
        value : str
            Column name for y-axis.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".
        capsize : float, default 0.1
            Length of the caps as a fraction of the width of the box.

        Returns
        -------
        WrappedPointPlot
            Point plot layer.
        """
        canvas = self._canvas()
        layer = _lt.WrappedPointPlot.from_table(
            self._df, offset, value, name=name, color=color, hatch=hatch, orient=orient,
            capsize=capsize, backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color, orient, value)
        return canvas.add_layer(layer)

    def add_barplot(
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
    ) -> _lt.WrappedBarPlot[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedBarPlot.from_table(
            self._df, offset, value, name=name, color=color, hatch=hatch, orient=orient,
            capsize=capsize, extent=extent, backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color, orient, value)
        return canvas.add_layer(layer)

    def _post_add_boxlike(self, layer: _BoxLikeMixin, color, orient, value: str):
        canvas = self._canvas()
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        if self._update_label:
            pos, labels, offset_labels = layer._generate_labels()
            self._update_xy_ticks(pos, labels, orient=orient)
            self._update_xy_label(offset_labels, value, orient=orient)

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
        """
        Add a categorical strip plot.

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_stripplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_stripplot(offset, "weight", color="region")

        Parameters
        ----------
        offset : tuple of str
            Column name(s) for x-axis.
        value : str
            Column name for y-axis.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        symbol : str or sequence of str, optional
            Column name(s) for symbols. Must be categorical.
        size : str, optional
            Column name for marker size. Must be numerical.
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".
        extent : float, default 0.5
            Width of the violins. Usually in range (0, 1].
        seed : int, optional
            Random seed for jittering.

        Returns
        -------
        WrappedMarkerGroups
            Marker collection layer.
        """
        canvas = self._canvas()
        orient = Orientation.parse(orient)
        symbol = theme._default("markers.symbol", symbol)
        size = theme._default("markers.size", size)
        layer = _lt.WrappedMarkers.build_stripplot(
            self._df, offset, value, name=name, color=color, hatch=hatch, symbol=symbol,
            size=size, orient=orient, extent=extent, seed=seed,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
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
        """
        Add a categorical swarm plot.

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_swarmplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_swarmplot(offset, "weight", color="region")

        Parameters
        ----------
        offset : tuple of str
            Column name(s) for x-axis.
        value : str
            Column name for y-axis.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        symbol : str or sequence of str, optional
            Column name(s) for symbols. Must be categorical.
        size : str, optional
            Column name for marker size. Must be numerical.
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".
        extent : float, default 0.8
            Width of the violins. Usually in range (0, 1].
        sort : bool, default False
            Whether to sort the data by value.

        Returns
        -------
        WrappedMarkerGroups
            Marker collection layer.
        """
        canvas = self._canvas()
        symbol = theme._default("markers.symbol", symbol)
        size = theme._default("markers.size", size)
        layer = _lt.WrappedMarkers.build_swarmplot(
            self._df, offset, value, name=name, color=color, hatch=hatch, symbol=symbol,
            size=size, orient=orient, extent=extent, sort=sort,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
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
        """
        Add a categorical count plot.

        >>> ### Count for each category in column "species".
        >>> canvas.cat(df).add_countplot("species")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_countplot(offset, color="region")

        Parameters
        ----------
        offset : tuple of str
            Column name(s) for x-axis.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        name : str, optional
            Name of the layer.
        orient : str, default "vertical"
            Orientation of the violins. Can be "vertical" or "horizontal".
        extent : float, default 0.8
            Width of the violins. Usually in range (0, 1].

        Returns
        -------
        WrappedBars
            Bar collection layer.
        """
        canvas = self._canvas()
        orient = Orientation.parse(orient)
        layer = _lt.WrappedBars.build_count(
            self._df, offset, color=color, hatch=hatch, orient=orient, extent=extent,
            name=name, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        if self._update_label:
            self._update_xy_label(offset, "count", orient=orient)
        return canvas.add_layer(layer)

    ### Aggregation ###

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
        """
        Add line that connect the aggregated values.

        >>> canvas.cat(df).mean().add_line("time", "value")

        Parameters
        ----------
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        width : str, optional
            Column name for line width. Must be numerical.
        style : str or sequence of str, optional
            Column name(s) for styling the lines. Must be categorical.

        Returns
        -------
        WrappedLines
            Line collection layer.
        """
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
        if color is not None and not layer._color_by.is_const():
            layer.with_color(_color.value, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
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
        """
        Add markers that represent the aggregated values.

        >>> canvas.cat(df).mean().add_markers("time", "value")

        Parameters
        ----------
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        size : str, optional
            Column name for marker size. Must be numerical.
        symbol : str or sequence of str, optional
            Column name(s) for symbols. Must be categorical.

        Returns
        -------
        WrappedMarkers
            Marker collection layer.
        """
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
        if color is not None and not layer._color_by.is_const():
            layer.with_color(_color.value, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
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
