from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

from whitecanvas import theme
from whitecanvas.canvas.dataframe._base import AggMethods, BaseCatPlotter, CatIterator
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.types import ColorType, Hatch, Orientation, Symbol

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.layers.tabular._box_like import _BoxLikeMixin
    from whitecanvas.layers.tabular._dataframe import DataFrameWrapper

    NStr = str | Sequence[str]

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")


class _Aggregator(Generic[_C, _DF]):
    def __init__(self, method: str, plotter: OneAxisCatPlotter[_C, _DF] = None):
        self._method = method
        self._plotter = plotter

    def __get__(self, ins: _C, owner) -> Self:
        return _Aggregator(self._method, ins)

    def __repr__(self) -> str:
        return f"Aggregator<{self._method}>"

    def __call__(self) -> OneAxisCatAggPlotter[_C, _DF]:
        """Aggregate the values before plotting it."""
        plotter = self._plotter
        if plotter is None:
            raise TypeError("Cannot call this method from a class.")
        if self._method == "size":
            value = "size"
        elif plotter._value is None:
            raise ValueError("Value column is not specified.")
        else:
            value = plotter._value
        return OneAxisCatAggPlotter(
            plotter._canvas(),
            plotter._cat_iter,
            offset=plotter._offset,
            value=value,
            method=self._method,
            orient=plotter._orient,
        )


class _GroupAggregator(Generic[_C, _DF]):
    def __init__(self, method: str, plotter: OneAxisCatPlotter[_C, _DF] = None):
        self._method = method
        self._plotter = plotter

    def __get__(self, ins: _C, owner) -> Self:
        return _GroupAggregator(self._method, ins)

    def __repr__(self) -> str:
        return f"GroupAggregator<{self._method}>"

    def __call__(self, by: str | tuple[str, ...]) -> OneAxisCatPlotter[_C, _DF]:
        """Aggregate the values for each group before plotting it."""
        plotter = self._plotter
        if isinstance(by, str):
            by = (by,)
        elif len(by) == 0:
            raise ValueError("No column is specified for grouping.")
        return type(plotter)(
            plotter._canvas(),
            plotter._df.agg_by((*plotter._offset, *by), plotter._value, self._method),
            offset=plotter._offset,
            value=plotter._value,
            update_label=plotter._update_label,
        )


class OneAxisCatPlotter(BaseCatPlotter[_C, _DF]):
    _orient: Orientation

    def __init__(
        self,
        canvas: _C,
        df: _DF,
        offset: str | tuple[str, ...] | None,
        value: str | None,
        update_label: bool = False,
    ):
        super().__init__(canvas, df)
        if isinstance(offset, str):
            offset = (offset,)
        elif offset is None:
            offset = ()
        self._offset = offset
        self._cat_iter = CatIterator(self._df, offset)
        self._value = value
        self._update_label = update_label
        if update_label:
            if value is not None:
                self._update_axis_labels(value)
            pos, label = self._cat_iter.axis_ticks()
            if self._orient.is_vertical:
                canvas.x.ticks.set_labels(pos, label)
            else:
                canvas.y.ticks.set_labels(pos, label)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(offset={self._offset!r}, value={self._value!r}, "
            f"orient={self._orient!r})"
        )

    def _update_axis_labels(self, value_label: str) -> None:
        """Update the x and y labels using the column names"""
        canvas = self._canvas()
        offset_label = self._cat_iter.axis_label()
        if self._orient.is_vertical:
            canvas.x.label.text = offset_label
            canvas.y.label.text = value_label
        else:
            canvas.x.label.text = value_label
            canvas.y.label.text = offset_label

    def _get_value(self) -> str:
        if self._value is None:
            raise ValueError("Value column is not specified.")
        return self._value

    def _update_xy_ticks(self, pos, label):
        """Update the x or y ticks to categorical ticks"""
        canvas = self._canvas()
        if self._orient.is_vertical:
            canvas.x.ticks.set_labels(pos, label)
        else:
            canvas.y.ticks.set_labels(pos, label)

    ### 1-D categorical ###

    def add_violinplot(
        self,
        *,
        name: str | None = None,
        color: NStr | None = None,
        hatch: NStr | None = None,
        dodge: NStr | bool = True,
        extent: float = 0.8,
        shape: str = "both",
    ) -> _lt.DFViolinPlot[_DF]:
        """
        Add a categorical violin plot.

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat_x(df, x="species", y="weight").add_violinplot()

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_violinplot(offset, "weight", color="region")

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        extent : float, default 0.8
            Width of the violins. Usually in range (0, 1].
        shape : str, default "both"
            Shape of the violins. Can be "both", "left", or "right".


        Returns
        -------
        WrappedViolinPlot
            Violin plot layer.
        """
        canvas = self._canvas()
        layer = _lt.DFViolinPlot(
            self._cat_iter, self._get_value(), name=name, color=color, hatch=hatch,
            dodge=dodge, extent=extent, shape=shape, orient=self._orient,
            backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color)
        return canvas.add_layer(layer)

    def add_boxplot(
        self,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        dodge: NStr | bool = True,
        name: str | None = None,
        capsize: float = 0.1,
        extent: float = 0.8,
    ) -> _lt.DFBoxPlot[_DF]:
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
        layer = _lt.DFBoxPlot(
            self._cat_iter, self._get_value(), name=name, color=color, hatch=hatch,
            dodge=dodge, orient=self._orient, capsize=capsize, extent=extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color)
        return canvas.add_layer(layer)

    def add_pointplot(
        self,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        dodge: NStr | bool = True,
        name: str | None = None,
        capsize: float = 0.1,
    ) -> _lt.DFPointPlot[_DF]:
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
        layer = _lt.DFPointPlot(
            self._cat_iter, self._get_value(), name=name, color=color, hatch=hatch,
            dodge=dodge, orient=self._orient, capsize=capsize,
            backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color)
        return canvas.add_layer(layer)

    def add_barplot(
        self,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        dodge: NStr | bool = True,
        name: str | None = None,
        capsize: float = 0.1,
        extent: float = 0.8,
    ) -> _lt.DFBarPlot[_DF]:
        """
        Add a categorical bar plot (bars with error bars).

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_barplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_barplot(offset, "weight", color="region")

        The default estimator and errors are mean and standard deviation. To change
        them, use `est_by_*` and `err_by_*` methods.

        >>> ### Use standard error x 2 (~95%) as error bars.
        >>> canvas.cat(df).add_barplot("species", "weight").err_by_se(scale=2.0)

        Parameters
        ----------
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        name : str, optional
            Name of the layer.
        capsize : float, default 0.1
            Length of the caps as a fraction of the width of the box.
        extent : float, default 0.8
            Width of the violins. Usually in range (0, 1].

        Returns
        -------
        WrappedBarPlot
            Bar plot layer.
        """
        canvas = self._canvas()
        layer = _lt.DFBarPlot(
            self._cat_iter, self._get_value(), name=name, color=color, hatch=hatch,
            dodge=dodge, orient=self._orient, capsize=capsize, extent=extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        self._post_add_boxlike(layer, color)
        return canvas.add_layer(layer)

    def _post_add_boxlike(self, layer: _BoxLikeMixin, color):
        canvas = self._canvas()
        if color is not None and not layer._color_by.is_const():
            layer.with_color_palette(canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())

    def add_stripplot(
        self,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        symbol: NStr | None = None,
        size: str | None = None,
        dodge: NStr | bool = False,
        name: str | None = None,
        extent: float = 0.5,
        seed: int | None = 0,
    ) -> _lt.DFMarkerGroups[_DF]:
        """
        Add a categorical strip plot.

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_stripplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_stripplot(offset, "weight", color="region")

        Parameters
        ----------
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
        symbol = theme._default("markers.symbol", symbol)
        size = theme._default("markers.size", size)

        df = self._df
        splitby, dodge = _splitby_dodge(df, self._offset, color, hatch, dodge)
        _map = self._cat_iter.prep_position_map(splitby, dodge)
        _extent = self._cat_iter.zoom_factor(dodge) * extent
        xj = _jitter.UniformJitter(splitby, _map, extent=_extent, seed=seed)
        yj = _jitter.IdentityJitter(self._get_value()).check(df)
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFMarkerGroups(
            df, xj, yj, name=name, color=color, hatch=hatch, orient=self._orient,
            symbol=symbol, size=size, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_markers(
        self,
        *,
        name: str | None = None,
        color: NStr | None = None,
        hatch: NStr | None = None,
        symbol: NStr | None = None,
        size: str | None = None,
        dodge: NStr | bool = False,
    ) -> _lt.DFMarkerGroups[_DF]:
        return self.add_stripplot(
            color=color, hatch=hatch, symbol=symbol, size=size, dodge=dodge,
            extent=0, seed=0, name=name,
        )  # fmt: skip

    def add_swarmplot(
        self,
        *,
        color: NStr | None = None,
        hatch: NStr | None = None,
        symbol: NStr | None = None,
        size: str | None = None,
        dodge: NStr | bool = False,
        name: str | None = None,
        extent: float = 0.8,
        sort: bool = False,
    ) -> _lt.DFMarkerGroups[_DF]:
        """
        Add a categorical swarm plot.

        >>> ### Use "species" column as categories and "weight" column as values.
        >>> canvas.cat(df).add_swarmplot("species", "weight")

        >>> ### Color by column "region" with dodging.
        >>> offset = ["species", "region"]  # categories that offset will be added
        >>> canvas.cat(df).add_swarmplot(offset, "weight", color="region")

        Parameters
        ----------
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
        df = self._df
        splitby, dodge = _splitby_dodge(df, self._offset, color, hatch, dodge)
        _map = self._cat_iter.prep_position_map(splitby, dodge)
        _extent = self._cat_iter.zoom_factor(dodge) * extent

        val = self._get_value()
        if sort:
            df = df.sort(val)
        lims = df[val].min(), df[val].max()
        xj = _jitter.SwarmJitter(splitby, _map, val, limits=lims, extent=_extent)
        yj = _jitter.IdentityJitter(val).check(df)
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFMarkerGroups(
            df, xj, yj, name=name, color=color, hatch=hatch, orient=self._orient,
            symbol=symbol, size=size, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    mean = _Aggregator("mean")
    median = _Aggregator("median")
    min = _Aggregator("min")
    max = _Aggregator("max")
    std = _Aggregator("std")
    sum = _Aggregator("sum")
    count = _Aggregator("size")

    mean_for_each = _GroupAggregator("mean")
    median_for_each = _GroupAggregator("median")
    min_for_each = _GroupAggregator("min")
    max_for_each = _GroupAggregator("max")
    std_for_each = _GroupAggregator("std")
    sum_for_each = _GroupAggregator("sum")


class OneAxisCatAggPlotter(BaseCatPlotter[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        cat_iter: CatIterator[_DF],
        offset: str | tuple[str, ...],
        value: str,
        method: AggMethods,
        orient: Orientation,
    ):
        super().__init__(canvas, cat_iter._df)
        self._offset = offset
        self._value = value
        self._agg_method = method
        self._orient = orient
        self._cat_iter = cat_iter

    def add_line(
        self,
        *,
        name: str | None = None,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ) -> _lt.DFLines[_DF]:
        """
        Add line that connect the aggregated values.

        >>> canvas.cat(df).mean().add_line("time", "value")

        Parameters
        ----------
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
        df = self._df
        _joined = _shared.join_columns(self._offset, color, style, source=df)
        df_agg = self._aggregate(df, _joined, self._value)
        xj = _jitter.CategoricalJitter(self._offset, self._cat_iter.category_map())
        yj = _jitter.IdentityJitter(self._value).check(df_agg)
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFLines.from_table(
            df_agg, xj, yj, name=name, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_markers(
        self,
        *,
        name: str | None = None,
        color: NStr | ColorType | None = None,
        hatch: NStr | Hatch | None = None,
        size: str | float | None = None,
        symbol: NStr | Symbol | None = None,
    ) -> _lt.DFMarkers[_DF]:
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
        df = self._df
        _joined = _shared.join_columns(self._offset, color, hatch, symbol, source=df)
        df_agg = self._aggregate(df, _joined, self._value)
        xj = _jitter.CategoricalJitter(self._offset, self._cat_iter.category_map())
        yj = _jitter.IdentityJitter(self._value).check(df_agg)
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFMarkers(
            df_agg, xj, yj, name=name, color=color, hatch=hatch, size=size,
            symbol=symbol, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_bars(
        self,
        *,
        name: str | None = None,
        color: NStr | ColorType | None = None,
        hatch: NStr | Hatch | None = None,
        extent: float = 0.8,
    ) -> _lt.DFBars[_DF]:
        """
        Add bars that represent the aggregated values.

        >>> canvas.cat(df).mean().add_bars("time", "value")

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
        width : str, optional
            Column name for bar width. Must be numerical.

        Returns
        -------
        WrappedBars
            Bar collection layer.
        """
        canvas = self._canvas()
        df = self._df
        _joined = _shared.join_columns(self._offset, color, hatch, source=df)
        df_agg = self._aggregate(df, _joined, self._value)
        xj = _jitter.CategoricalJitter(self._offset, self._cat_iter.category_map())
        yj = _jitter.IdentityJitter(self._value).check(df_agg)
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFBars.from_table(
            df_agg, xj, yj, name=name, color=color, hatch=hatch, extent=extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def _aggregate(
        self,
        df: DataFrameWrapper,
        by: tuple[str, ...],
        on: str,
    ) -> DataFrameWrapper[_DF]:
        if self._agg_method == "size":
            return df.value_count(by)
        else:
            if on is None:
                raise ValueError("Value column is not specified.")
            return df.agg_by(by, on, self._agg_method)


class XCatPlotter(OneAxisCatPlotter[_C, _DF]):
    _orient = Orientation.VERTICAL


class YCatPlotter(OneAxisCatPlotter[_C, _DF]):
    _orient = Orientation.HORIZONTAL


def _splitby_dodge(
    source: DataFrameWrapper[_DF],
    offset: str | tuple[str, ...],
    color: str | tuple[str, ...] | None = None,
    hatch: str | tuple[str, ...] | None = None,
    dodge: str | tuple[str, ...] | bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(offset, str):
        offset = (offset,)
    if isinstance(dodge, bool):
        if dodge:
            _all = _shared.join_columns(color, hatch, source=source)
            dodge = tuple(c for c in _all if c not in offset)
        else:
            dodge = ()
    elif isinstance(dodge, str):
        dodge = (dodge,)
    else:
        dodge = tuple(dodge)
    splitby = _shared.join_columns(offset, dodge, source=source)
    return splitby, dodge