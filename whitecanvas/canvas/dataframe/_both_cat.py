from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

import numpy as np

from whitecanvas.canvas.dataframe._base import BaseCatPlotter, CatIterator
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular import _jitter
from whitecanvas.types import ColormapType

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.layers.tabular._dataframe import DataFrameWrapper

    NStr = str | Sequence[str]

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")


class _XYAggregator(Generic[_C, _DF]):
    def __init__(self, method: str, plotter: XYCatPlotter[_C, _DF] = None):
        self._method = method
        self._plotter = plotter

    def __get__(self, ins: _C, owner) -> Self:
        return _XYAggregator(self._method, ins)

    def __repr__(self) -> str:
        return f"XYAggregator<{self._method}>"

    def __call__(self) -> XYCatAggPlotter[_C, _DF]:
        """Aggregate the values before plotting it."""
        plotter = self._plotter
        if plotter is None:
            raise TypeError("Cannot call this method from a class.")
        if plotter._x is None or plotter._y is None:
            raise ValueError("Value column is not specified.")
        return XYCatAggPlotter(
            plotter._canvas(),
            plotter._cat_iter_x,
            plotter._cat_iter_y,
            x=plotter._x,
            y=plotter._y,
            method=self._method,
        )


class XYCatPlotter(BaseCatPlotter[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
        x: str | tuple[str, ...],
        y: str | tuple[str, ...],
        update_labels: bool = False,
    ):
        super().__init__(canvas, df)
        if isinstance(x, str):
            x = (x,)
        if isinstance(y, str):
            y = (y,)
        self._x: tuple[str, ...] = x
        self._y: tuple[str, ...] = y
        self._update_label = update_labels
        self._cat_iter_x = CatIterator(self._df, x)
        self._cat_iter_y = CatIterator(self._df, y)
        if update_labels:
            self._update_xy_label(x, y)
        self._update_axis_labels()

    def _update_xy_label(
        self,
        x: str | tuple[str, ...],
        y: str | tuple[str, ...],
    ) -> None:
        """Update the x and y labels using the column names"""
        canvas = self._canvas()
        if not isinstance(x, str):
            x = "/".join(x)
        if not isinstance(y, str):
            y = "/".join(y)
        canvas.x.label.text = x
        canvas.y.label.text = y

    def _update_axis_labels(self) -> None:
        """Update the x and y labels using the column names"""
        canvas = self._canvas()
        canvas.x.ticks.set_labels(*self._cat_iter_x.axis_ticks())
        canvas.y.ticks.set_labels(*self._cat_iter_y.axis_ticks())

    mean = _XYAggregator("mean")
    median = _XYAggregator("median")
    sum = _XYAggregator("sum")
    min = _XYAggregator("min")
    max = _XYAggregator("max")
    count = _XYAggregator("size")
    first = _XYAggregator("first")


class XYCatAggPlotter(BaseCatPlotter[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        cat_iter_x: CatIterator[_DF],
        cat_iter_y: CatIterator[_DF],
        x: str | tuple[str, ...],
        y: str | tuple[str, ...],
        method: str,
    ):
        super().__init__(canvas, cat_iter_x.df)
        self._cat_iter_x = cat_iter_x
        self._cat_iter_y = cat_iter_y
        self._x = x
        self._y = y
        self._agg_method = method

    def add_heatmap(
        self,
        value: str,
        *,
        cmap: ColormapType = "inferno",
        clim: tuple[float, float] | None = None,
        name: str | None = None,
        fill: float = 0,
    ) -> _lt.DFHeatmap[_DF]:
        """
        Add a heatmap whose color represents the value of the aggregated data.

        Parameters
        ----------
        value : str
            Column name to use as the value.
        cmap : colormap-like, default "inferno"
            Colormap to use for the heatmap.
        clim : (float, float), optional
            Color limits for the colormap. If not specified, the limits are calculated
            from the data min/max.
        name : str, optional
            Name of the layer.
        fill : float, optional
            Value to fill for the cells that do not have any data. This value will not
            be considered when calculating the color limits.

        Returns
        -------
        DFHeatmap
            Dataframe bound heatmap layer.
        """
        canvas = self._canvas()
        by_both = (*self._x, *self._y)
        nx = len(self._x)
        df_agg = self._aggregate(self._df, by_both, value)
        map_x = self._cat_iter_x.prep_position_map(self._x)
        map_y = self._cat_iter_y.prep_position_map(self._y)
        dtype = df_agg[value].dtype
        if dtype.kind not in "fiub":
            raise ValueError(f"Column {value!r} is not numeric.")
        arr = np.full((len(map_y), len(map_x)), fill, dtype=dtype)
        for sl, sub in df_agg.group_by(by_both):
            xval, yval = sl[:nx], sl[nx:]
            vals = sub[value]
            if vals.size == 1:
                arr[map_y[yval], map_x[xval]] = vals[0]
            else:
                raise ValueError(f"More than one value found for {sl!r}.")
        if clim is None:
            # `fill` may be outside the range of the data, so calculate clim here.
            clim = df_agg[value].min(), df_agg[value].max()
        layer = _lt.DFHeatmap.from_array(
            df_agg, arr, name=name, cmap=cmap, clim=clim, backend=canvas._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(layer)

    def add_markers(
        self,
        value: str,
        *,
        color: NStr | None = None,
        name: str | None = None,
        map_from: tuple[float, float] | None = None,
        map_to: tuple[float, float] = (3, 18),
    ) -> _lt.DFMarkerGroups[_DF]:
        """
        Add markers at the grid positions with the size representing the value.

        >>> canvas.cat_xy(df, "x", "y").count().add_markers("value")

        Parameters
        ----------
        value : str
            Column name to use as the value.
        color : str or sequence of str, optional
            Column name(s) to use as the color of the markers.
        name : str, optional
            Name of the layer.
        map_from : tuple of float, optional
            Limits of values that will be linearly mapped to the marker size. Data
            points outside this range will be clipped. If not specified, the min/max
            of the data will be used.
        map_to : tuple of float, optional
            Minimum and maximum size of the markers.

        Returns
        -------
        _lt.DFMarkerGroups[_DF]
            _description_
        """
        canvas = self._canvas()
        x, y = self._x, self._y
        xj = _jitter.CategoricalJitter(x, self._cat_iter_x.prep_position_map(x))
        yj = _jitter.CategoricalJitter(y, self._cat_iter_y.prep_position_map(y))
        return canvas.add_layer(
            _lt.DFMarkers(self._df, xj, yj, color=color, name=name).update_size(
                value, map_from=map_from, map_to=map_to
            )
        )

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
