from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

import numpy as np

from whitecanvas import theme
from whitecanvas.canvas.dataframe._base import AggMethods, BaseCatPlotter, CatIterator
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.types import ColormapType, ColorType, Hatch, Orientation, Symbol

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.canvas.dataframe._stacked_cat import StackedCatPlotter
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
            plotter._df.agg_by((*plotter._offset, *by), [plotter._value], self._method),
            offset=plotter._offset,
            value=plotter._value,
            update_labels=plotter._update_labels,
        )


class OneAxisCatPlotter(BaseCatPlotter[_C, _DF]):
    _orient: Orientation

    def __init__(
        self,
        canvas: _C,
        df: _DF,
        offset: str | tuple[str, ...] | None,
        value: str | None,
        update_labels: bool = False,
    ):
        super().__init__(canvas, df)
        if isinstance(offset, str):
            offset = (offset,)
        elif offset is None:
            offset = ()
        else:
            if any(not isinstance(o, str) for o in offset):
                raise TypeError(
                    "Category column(s) must be specified by a string or a sequence "
                    f"of strings, got {offset!r}."
                )
        # check dtype
        for col in offset:
            arr = self._df[col]
            if arr.dtype.kind in "fc":
                raise ValueError(
                    f"Column {col!r} cannot be interpreted as a categorical column "
                    f"(dtype={arr.dtype!r})."
                )
        self._offset: tuple[str, ...] = offset
        self._cat_iter = CatIterator(self._df, offset)
        self._value = value
        self._update_labels = update_labels
        if update_labels:
            if value is not None:
                self._update_axis_labels(value)
            pos, label = self._cat_iter.axis_ticks()
            if self._orient.is_vertical:
                canvas.x.ticks.set_labels(pos, label)
                canvas.x.lim = (np.min(pos) - 0.5, np.max(pos) + 0.5)
            else:
                canvas.y.ticks.set_labels(pos, label)
                canvas.y.lim = (np.min(pos) - 0.5, np.max(pos) + 0.5)

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

    def stack(
        self,
        by: str | Sequence[str] | None = None,
    ) -> StackedCatPlotter[_C, _DF]:
        """
        Return a stacked plotter for data stacked by the given column(s).

        >>> canvas.cat_x(df, "time", "value").stack("region").add_bars()
        >>> canvas.cat_x(df, "time", "value").stack("region").add_bars(color="region")
        >>> canvas.cat_x(df, "time", "value").stack("region").add_bars(hatch="region")

        Parameters
        ----------
        by : str or sequence of str
            Column name(s) for stacking the data.
        """
        from whitecanvas.canvas.dataframe._stacked_cat import StackedCatPlotter

        return StackedCatPlotter(
            self._canvas(),
            df=self._df,
            offset=self._offset,
            value=self._value,
            orient=self._orient,
            stackby=by,
            update_labels=self._update_labels,
        )

    def melt(
        self,
        id_vars: str | Sequence[str] | None = None,
        value_vars: str | Sequence[str] | None = None,
        var_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        """
        Melt the dataframe to long format and update the x/y column names.

        This method should be used without setting x or y.

        >>> canvas.cat_x(df).melt().add_stripplot()

        Parameters
        ----------
        id_vars : str or sequence of str
            Column name(s) to use as identifier variables.
        value_vars : str or sequence of str
            Column name(s) to use as value variables.
        var_name : str, optional
            Name of the variable column. If None, it will be set to "variable".
        value_name : str, optional
            Name of the value column. If None, it will be set to "value".
        """
        if self._value is not None:
            raise ValueError("Cannot melt the dataframe with x or y set.")
        if id_vars is None:
            id_vars = []
        if value_vars is None:
            value_vars = self._df.columns
        if var_name is None:
            var_name = "variable"
        if value_name is None:
            value_name = "value"
        df = self._df.melt(id_vars, value_vars, var_name, value_name)
        out = type(self)(self._canvas(), df, var_name, value_name, self._update_labels)
        out._df = df
        return out

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
        >>> canvas.cat_x(df, "region", "weight").add_violinplot(dodge=True)

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
        DFViolinPlot
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
        >>> canvas.cat_x(df, x="species", y="weight").add_boxplot()

        >>> ### Color by column "region" with dodging.
        >>> canvas.cat_x(df, "region", "weight").add_boxplot(dodge=True)

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
        DFBoxPlot
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
        >>> canvas.cat_x(df, x="species", y="weight").add_pointplot()

        >>> ### Color by column "region" with dodging.
        >>> canvas.cat_x(df, "region", "weight").add_pointplot(dodge=True)

        The default estimator and errors are mean and standard deviation. To change
        them, use `est_by_*` and `err_by_*` methods.

        >>> ### Use standard error x 2 (~95%) as error bars.
        >>> canvas.cat_x(df, "species", "weight").add_pointplot().err_by_se(scale=2.0)

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

        Returns
        -------
        DFPointPlot
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
        >>> canvas.cat_x(df, x="species", y="weight").add_barplot()

        >>> ### Color by column "region" with dodging.
        >>> canvas.cat_x(df, "region", "weight").add_barplot(dodge=True)

        The default estimator and errors are mean and standard deviation. To change
        them, use `est_by_*` and `err_by_*` methods.

        >>> ### Use standard error x 2 (~95%) as error bars.
        >>> canvas.cat_x(df, "species", "weight").add_barplot().err_by_se(scale=2.0)

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
        DFBarPlot
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
            layer.update_color_palette(canvas._color_palette)
        elif color is None:
            layer.update_const(color=canvas._color_palette.next())

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
        >>> canvas.cat_x(df, x="species", y="weight").add_stripplot()

        >>> ### Color by column "region" with dodging.
        >>> canvas.cat_x(df, "region", "weight").add_stripplot(dodge=True)

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
        DFMarkerGroups
            Marker collection layer.
        """
        canvas = self._canvas()
        symbol = theme._default("markers.symbol", symbol)
        size = theme._default("markers.size", size)

        df = self._df
        splitby, dodge = _shared.norm_dodge_markers(
            df, self._offset, color, hatch, dodge
        )  # fmt: skip
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
            layer.update_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
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
        """Alias of `add_stripplot` with no jittering."""
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
        >>> canvas.cat_x(df, x="species", y="weight").add_swarmplot()

        >>> ### Color by column "region" with dodging.
        >>> canvas.cat_x(df, "region", "weight").add_swarmplot(dodge=True)

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
        DFMarkerGroups
            Marker collection layer.
        """
        canvas = self._canvas()
        symbol = theme._default("markers.symbol", symbol)
        size = theme._default("markers.size", size)
        df = self._df
        splitby, dodge = _shared.norm_dodge_markers(
            df, self._offset, color, hatch, dodge
        )  # fmt: skip
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
            layer.update_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_rugplot(
        self,
        *,
        name: str | None = None,
        color: NStr | None = None,
        width: float | None = None,
        style: NStr | None = None,
        dodge: NStr | bool = True,
        extent: float = 0.8,
    ) -> _lt.DFRugGroups[_DF]:
        canvas = self._canvas()
        width = theme._default("line.width", width)

        df = self._df
        splitby, dodge = _shared.norm_dodge(
            df, self._offset, color, style, dodge=dodge
        )  # fmt: skip
        _map = self._cat_iter.prep_position_map(splitby, dodge)
        _extent = self._cat_iter.zoom_factor(dodge) * extent
        jitter = _jitter.CategoricalJitter(splitby, _map)
        layer = _lt.DFRugGroups.from_table(
            df, jitter, self._get_value(), name=name, color=color, orient=self._orient,
            extent=_extent, width=width, style=style, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_hist_heatmap(
        self,
        cmap: ColormapType = "inferno",
        clim: tuple[float, float] | None = None,
    ) -> _lt.DFHeatmap[_DF]:
        # TODO: implement this
        raise NotImplementedError

    # aggregators and group aggregators
    mean = _Aggregator("mean")
    median = _Aggregator("median")
    min = _Aggregator("min")
    max = _Aggregator("max")
    std = _Aggregator("std")
    sum = _Aggregator("sum")
    count = _Aggregator("size")
    first = _Aggregator("first")

    mean_for_each = _GroupAggregator("mean")
    median_for_each = _GroupAggregator("median")
    min_for_each = _GroupAggregator("min")
    max_for_each = _GroupAggregator("max")
    std_for_each = _GroupAggregator("std")
    sum_for_each = _GroupAggregator("sum")
    first_for_each = _GroupAggregator("first")


class OneAxisCatAggPlotter(BaseCatPlotter[_C, _DF]):
    """Class for plotting aggregated values of a single categorical axis."""

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
        width: float | None = None,
        style: NStr | None = None,
        dodge: NStr | bool = False,
    ) -> _lt.DFLines[_DF]:
        """
        Add line that connect the aggregated values.

        >>> canvas.cat_x(df, "time", "value").mean().add_line()

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        width : float, optional
            Line width.
        style : str or sequence of str, optional
            Column name(s) for styling the lines. Must be categorical.

        Returns
        -------
        DFLines
            Line collection layer.
        """
        # TODO: support width: str
        canvas = self._canvas()
        df = self._df
        width = theme._default("line.width", width)

        _splitby, _dodge = _shared.norm_dodge(df, self._offset, color, dodge=dodge)
        df_agg = self._aggregate(df, _splitby, self._value)
        _pos_map = self._cat_iter.prep_position_map(_splitby, dodge=_dodge)

        xj = _jitter.CategoricalJitter(_splitby, _pos_map)
        yj = _jitter.IdentityJitter(self._value).check(df_agg)
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFLines.from_table(
            df_agg, xj, yj, name=name, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_markers(
        self,
        *,
        name: str | None = None,
        color: NStr | ColorType | None = None,
        hatch: NStr | Hatch | None = None,
        size: str | float | None = None,
        symbol: NStr | Symbol | None = None,
        dodge: NStr | bool = False,
    ) -> _lt.DFMarkers[_DF]:
        """
        Add markers that represent the aggregated values.

        >>> canvas.cat_x(df, "time", "value").mean().add_markers()

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
        DFMarkers
            Marker collection layer.
        """
        canvas = self._canvas()
        df = self._df
        _splitby, _dodge = _shared.norm_dodge(
            df, self._offset, color, hatch, symbol, dodge=dodge
        )  # fmt: skip
        df_agg = self._aggregate(df, _splitby, self._value)
        _pos_map = self._cat_iter.prep_position_map(_splitby, dodge=_dodge)

        xj = _jitter.CategoricalJitter(_splitby, _pos_map)
        yj = _jitter.IdentityJitter(self._value).check(df_agg)
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFMarkers(
            df_agg, xj, yj, name=name, color=color, hatch=hatch, size=size,
            symbol=symbol, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_bars(
        self,
        *,
        name: str | None = None,
        color: NStr | ColorType | None = None,
        hatch: NStr | Hatch | None = None,
        extent: float = 0.8,
        dodge: NStr | bool = True,
    ) -> _lt.DFBars[_DF]:
        """
        Add bars that represent the aggregated values.

        >>> canvas.cat_x(df, "time", "value").mean().add_bars()

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
        DFBars
            Bar collection layer.
        """
        canvas = self._canvas()
        df = self._df
        _splitby, _dodge = _shared.norm_dodge(
            df, self._offset, color, hatch, dodge=dodge
        )  # fmt: skip
        df_agg = self._aggregate(df, _splitby, self._value)
        _pos_map = self._cat_iter.prep_position_map(_splitby, dodge=_dodge)

        xj = _jitter.CategoricalJitter(_splitby, _pos_map)
        yj = _jitter.IdentityJitter(self._value).check(df_agg)

        _extent = self._cat_iter.zoom_factor(_dodge) * extent
        if not self._orient.is_vertical:
            xj, yj = yj, xj
        layer = _lt.DFBars.from_table(
            df_agg, xj, yj, name=name, color=color, hatch=hatch, extent=_extent,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(color, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def _aggregate(
        self,
        df: DataFrameWrapper[_DF],
        by: tuple[str, ...],
        on: str,
    ) -> DataFrameWrapper[_DF]:
        if self._agg_method == "size":
            return df.value_count(by)
        elif self._agg_method == "first":
            return df.value_first(by, on)
        else:
            if on is None:
                raise ValueError("Value column is not specified.")
            return df.agg_by(by, [on], self._agg_method)


class XCatPlotter(OneAxisCatPlotter[_C, _DF]):
    _orient = Orientation.VERTICAL


class YCatPlotter(OneAxisCatPlotter[_C, _DF]):
    _orient = Orientation.HORIZONTAL
