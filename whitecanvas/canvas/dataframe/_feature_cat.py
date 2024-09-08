from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    Sequence,
    TypeVar,
)

from whitecanvas import theme
from whitecanvas.canvas.dataframe._base import BaseCatPlotter
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular import _jitter
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.types import HistBinType, KdeBandWidthType, Orientation

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas._base import CanvasBase

    NStr = str | Sequence[str]

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")


class _Aggregator(Generic[_C, _DF]):
    def __init__(self, method: str, plotter: CatPlotter[_C, _DF] = None):
        self._method = method
        self._plotter = plotter

    def __get__(self, ins: _C, owner) -> Self:
        return _Aggregator(self._method, ins)

    def __repr__(self) -> str:
        return f"Aggregator<{self._method}>"

    def __call__(self, by: str | tuple[str, ...]) -> CatPlotter[_C, _DF]:
        """Aggregate the values before plotting it."""
        plotter = self._plotter
        if isinstance(by, str):
            by = (by,)
        elif len(by) == 0:
            raise ValueError("No column is specified for grouping.")
        if plotter is None:
            raise TypeError("Cannot call this method from a class.")
        on = [plotter._x, plotter._y]
        df_agg = plotter._df.agg_by(by, on, self._method)
        return CatPlotter(
            plotter._canvas(),
            df_agg,
            x=plotter._x,
            y=plotter._y,
        )


class CatPlotter(BaseCatPlotter[_C, _DF]):
    """
    Categorical plotter that categorizes the data by features (color, style etc.)
    """

    def __init__(
        self,
        canvas: _C,
        df: _DF,
        x: str | None,
        y: str | None,
        update_labels: bool = False,
    ):
        super().__init__(canvas, df)
        self._x = x
        self._y = y
        self._update_label = update_labels
        if update_labels:
            self._update_xy_label(x, y)

    def _get_x(self) -> str:
        if self._x is None:
            raise ValueError("Column for x-axis is not set")
        return self._x

    def _get_y(self) -> str:
        if self._y is None:
            raise ValueError("Column for y-axis is not set")
        return self._y

    def _update_xy_label(self, x: str | None, y: str | None) -> None:
        """Update the x and y labels using the column names"""
        canvas = self._canvas()
        if isinstance(x, str):
            canvas.x.label.text = x
        if isinstance(y, str):
            canvas.y.label.text = y

    def along_x(self) -> CatPlotter[_C, _DF]:
        """Return the same plotter but with only x-axis set."""
        return self._copy_like(self._get_x(), None, self._update_label)

    def along_y(self) -> CatPlotter[_C, _DF]:
        """Return the same plotter but with only y-axis set."""
        return self._copy_like(None, self._get_y(), self._update_label)

    def _copy_like(self, x, y, update_label):
        out = self.__class__(self._canvas(), self._df, x, y, False)
        out._update_label = update_label
        return out

    def add_line(
        self,
        *,
        name: str | None = None,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ) -> _lt.DFLines[_DF]:
        """
        Add a categorical line plot.

        >>> ### Use "time" column as x-axis and "value" column as y-axis
        >>> canvas.cat(df, "time", "value").add_line()

        >>> ### Multiple lines colored by column "group"
        >>> canvas.cat(df, "time", "value").add_line(color="group")

        >>> ### Multiple lines styled by column "group"
        >>> canvas.cat(df, "time", "value").add_line(style="group")

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
        DFLines
            Line collection layer.
        """
        canvas = self._canvas()
        width = theme._default("line.width", width)
        layer = _lt.DFLines.from_table(
            self._df, self._get_x(), self._get_y(), name=name, color=color, width=width,
            style=style, backend=canvas._get_backend(),
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
        size: str | None = None,
        symbol: NStr | None = None,
    ) -> _lt.DFMarkers[_DF]:
        """
        Add a categorical marker plot.

        >>> ### Use "time" column as x-axis and "value" column as y-axis
        >>> canvas.cat(df, "time", "value").add_markers()

        >>> ### Multiple markers colored by column "group"
        >>> canvas.cat(df, "time", "value").add_markers(color="group")

        >>> ### Change marker size according to "weight" column
        >>> canvas.cat(df, "time", "value").add_markers(size="weight")

        >>> ### Multiple markers with hatches determined by column "group"
        >>> canvas.cat(df, "time", "value").add_markers(hatch="group")

        >>> ### Multiple markers with symbols determined by "group"
        >>> canvas.cat(df, "time", "value").add_markers(symbol="group")

        Parameters
        ----------
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
        xj = _jitter.IdentityJitter(self._get_x())
        yj = _jitter.IdentityJitter(self._get_y())
        layer = _lt.DFMarkers(
            self._df, xj, yj, name=name, color=color, hatch=hatch,
            size=size, symbol=symbol, backend=canvas._get_backend(),
        )  # fmt: skip
        if (
            color is not None
            and not layer._color_by.is_const()
            and isinstance(layer._color_by, _p.ColorPlan)
        ):
            layer.update_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_hist2d(
        self,
        *,
        name: str | None = None,
        color: str | Sequence[str] | None = None,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        rangex: tuple[float, float] | None = None,
        rangey: tuple[float, float] | None = None,
    ) -> _lt.DFMultiHeatmap[_DF]:
        """
        Add 2-D histogram of given x/y columns.

        >>> ### Use "tip" column as x-axis and "total_bill" column as y-axis
        >>> canvas.cat(df, "tip", "total_bill").add_hist2d()

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        color : str, optional
            Column name(s) for coloring the histogram.
        bins : int, array, str or tuple of them, default "auto"
            If int, the number of bins for both x and y. If tuple, the number of bins
            for x and y respectively.
        rangex : (float, float), optional
            Range of x values in which histogram will be built.
        rangey : (float, float), optional
            Range of y values in which histogram will be built.

        Returns
        -------
        DFMultiHeatmap
            Dataframe bound heatmap layer.
        """
        canvas = self._canvas()
        layer = _lt.DFMultiHeatmap.build_hist(
            self._df, self._get_x(), self._get_y(), color=color,name=name, bins=bins,
            range=(rangex, rangey), palette=canvas._color_palette,
            backend=canvas._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(layer)

    def add_kde2d(
        self,
        *,
        name: str | None = None,
        color: str | None = None,
        band_width: KdeBandWidthType = "scott",
    ) -> _lt.DFMultiHeatmap[_DF]:
        """
        Add 2-D kernel density estimation of given x/y columns.

        >>> ### Use "tip" column as x-axis and "total_bill" column as y-axis
        >>> canvas.cat(df, "tip", "total_bill").add_kde2d()

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        color : str, optional
            Column name(s) for coloring the densities.
        band_width : float, default None
            Bandwidth of the kernel density estimation. If None, use Scott's rule.

        Returns
        -------
        DFMultiHeatmap
            Dataframe bound heatmap layer.
        """
        canvas = self._canvas()
        layer = _lt.DFMultiHeatmap.build_kde(
            self._df, self._get_x(), self._get_y(), color=color, name=name,
            band_width=band_width, palette=canvas._color_palette,
            backend=canvas._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(layer)

    def add_pointplot(
        self,
        *,
        name: str | None = None,
        color: NStr | None = None,
        hatch: NStr | None = None,
        size: float | None = None,
        capsize: float = 0.15,
    ) -> _lt.DFPointPlot2D[_DF]:
        """
        Add 2-D point plot.

        >>> ### Use "time" column as x-axis and "value" column as y-axis
        >>> canvas.cat(df, "time", "value").add_pointplot()

        >>> ### Multiple point plots colored by column "group"
        >>> canvas.cat(df, "time", "value").add_pointplot(color="group")

        >>> ### Multiple point plots with hatches determined by column "group"
        >>> canvas.cat(df, "time", "value").add_pointplot(hatch="group")

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        color : str or sequence of str, optional
            Column name(s) for coloring the lines. Must be categorical.
        hatch : str or sequence of str, optional
            Column name(s) for hatches. Must be categorical.
        size : float, optional
            Size of the points.
        capsize : float, default 0.15
            Size of the cap on the error bars.

        Returns
        -------
        DFPointPlot2D
            Point plot layer.
        """
        canvas = self._canvas()
        layer = _lt.DFPointPlot2D(
            self._df, self._get_x(), self._get_y(), name=name, color=color,
            hatch=hatch, size=size, capsize=capsize, backend=canvas._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(layer)

    def add_hist(
        self,
        *,
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        kind: str = "count",
        shape: str = "bars",
        name: str | None = None,
        color: NStr | None = None,
        width: float | None = None,
        style: NStr | None = None,
    ) -> _lt.DFHistograms[_DF]:
        """
        Add histograms.

        >>> ### Use "value" column as x-axis
        >>> canvas.cat(df, x="value").add_hist()

        >>> ### Multiple KDEs colored by column "group"
        >>> canvas.cat(df, x="value).add_hist(color="group")

        Parameters
        ----------
        bins : int, array-like or str, default "auto"
            If int, the number of bins. If array, the bin edges. If str, the method to
            calculate the number of bins.
        limits : (float, float), optional
            Lower and upper limits of the bins.
        kind : {"count", "density", "probability", "frequency"}, default "count"
            Kind of histogram to draw.
        shape : {"bars", "steps", "lines"}, default "bars"
            Shape of the histogram.
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
        DFHistograms
            Histogram layer.
        """
        canvas = self._canvas()
        width = theme._default("line.width", width)
        x0, orient = self._column_and_orient()
        layer = _lt.DFHistograms.from_table(
            self._df, x0, bins=bins, limits=limits, kind=kind, shape=shape, name=name,
            orient=orient, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        if self._update_label:
            if orient.is_vertical:
                canvas.y.label.text = kind
            else:
                canvas.x.label.text = kind

        return canvas.add_layer(layer)

    def add_kde(
        self,
        *,
        band_width: KdeBandWidthType = "scott",
        name: str | None = None,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ) -> _lt.DFKde[_DF]:
        """
        Add lines representing kernel density estimation.

        >>> ### Use "value" column as x-axis
        >>> canvas.cat(df, x="value").add_kde()

        >>> ### Multiple KDEs colored by column "group"
        >>> canvas.cat(df, x="value).add_kde(color="group")

        Parameters
        ----------
        band_width : float, default None
            Bandwidth of the kernel density estimation. If None, use Scott's rule.
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
        DFKde
            KDE layer.
        """
        canvas = self._canvas()
        width = theme._default("line.width", width)
        x0, orient = self._column_and_orient()
        layer = _lt.DFKde.from_table(
            self._df, x0, band_width=band_width, name=name,
            orient=orient, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        if self._update_label:
            ax_label = "density"
            if orient.is_vertical:
                canvas.y.label.text = ax_label
            else:
                canvas.x.label.text = ax_label
        return canvas.add_layer(layer)

    def add_rug(
        self,
        *,
        name: str | None = None,
        color: NStr | None = None,
        width: NStr | float | None = None,
        style: NStr | None = None,
        low: float = 0.0,
        high: float | None = None,
    ) -> _lt.DFRug[_DF]:
        """
        Add a rug plot representing 1D distribution.

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
        low : float, default 0.0
            Lower bound of each rug lines.
        high : float, default 0.0
            Higher bound of each rug lines. Automatically determined by the canvas size
            by default.

        Returns
        -------
        DFRug
            Rug plot layer.
        """
        canvas = self._canvas()
        width = theme._default("line.width", width)
        x0, orient = self._column_and_orient()
        if high is None:
            if orient.is_vertical:
                hmin, hmax = sorted(canvas.y.lim)
            else:
                hmin, hmax = sorted(canvas.x.lim)
            high = low + (hmax - hmin) * 0.05
        layer = _lt.DFRug.from_table(
            self._df, x0, name=name, orient=orient, color=color, width=width,
            style=style, low=low, high=high, backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.update_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def _column_and_orient(self) -> tuple[str, Orientation]:
        if self._x is None and self._y is None:
            raise ValueError("Column for either x- or y-axis must be set")
        elif self._x is not None and self._y is not None:
            raise ValueError("Only one of x- or y-axis can be set")
        elif self._x is not None:
            return self._x, Orientation.VERTICAL
        else:
            return self._y, Orientation.HORIZONTAL

    # aggregators and group aggregators
    mean_for_each = _Aggregator("mean")
    median_for_each = _Aggregator("median")
