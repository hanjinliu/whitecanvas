from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Sequence,
    TypeVar,
)

from whitecanvas.canvas.dataframe._base import BaseCatPlotter
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular import _jitter
from whitecanvas.types import ArrayLike1D, ColormapType, Orientation

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

    NStr = str | Sequence[str]

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")


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
        update_label: bool = False,
    ):
        super().__init__(canvas, df)
        self._x = x
        self._y = y
        self._update_label = update_label
        if update_label:
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
        layer = _lt.DFLines.from_table(
            self._df, self._get_x(), self._get_y(), name=name, color=color, width=width,
            style=style, backend=canvas._get_backend(),
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
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        return canvas.add_layer(layer)

    def add_hist2d(
        self,
        *,
        cmap: ColormapType = "inferno",
        name: str | None = None,
        bins: int | tuple[int, int] = 10,
        rangex: tuple[float, float] | None = None,
        rangey: tuple[float, float] | None = None,
        density: bool = False,
    ):
        """
        Add 2-D histogram of given x/y columns.

        >>> ### Use "tip" column as x-axis and "total_bill" column as y-axis
        >>> canvas.cat(df, "tip", "total_bill").add_hist2d()

        Parameters
        ----------
        cmap : colormap-like, default "inferno"
            Colormap to use for the heatmap.
        name : str, optional
            Name of the layer.
        bins : int or tuple[int, int], default 10
            If int, the number of bins for both x and y. If tuple, the number of bins
            for x and y respectively.
        rangex : (float, float), optional
            Range of x values in which histogram will be built.
        rangey : (float, float), optional
            Range of y values in which histogram will be built.
        density : bool, default False
            If True, the result is the value of the probability density function at the
            bin, normalized such that the integral over the range is 1.

        Returns
        -------
        DFHeatmap
            Dataframe bound heatmap layer.
        """
        canvas = self._canvas()
        layer = _lt.DFHeatmap.build_hist(
            self._df, self._get_x(), self._get_y(), cmap=cmap, name=name, bins=bins,
            range=(rangex, rangey), density=density, backend=canvas._get_backend(),
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
    ):
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
        bins: int | ArrayLike1D = 10,
        range: tuple[float, float] | None = None,
        density: bool = False,
        name: str | None = None,
        color: NStr | None = None,
        hatch: NStr | None = None,
    ):
        # TODO: implement this
        raise NotImplementedError

    def add_hist_line(
        self,
        *,
        bins: int | ArrayLike1D = 10,
        range: tuple[float, float] | None = None,
        density: bool = False,
        name: str | None = None,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ):
        """
        Add lines representing histograms.

        >>> ### Use "value" column as x-axis
        >>> canvas.cat(df, x="value").add_line_hist(bins=8, density=True)

        >>> ### Multiple histograms colored by column "group"
        >>> canvas.cat(df, x="value").add_line_hist(color="group")

        Parameters
        ----------
        bins : int or array-like, default 10
            If an integer, the number of bins. If an array, the bin edges.
        range : (float, float), default None
            If provided, the lower and upper range of the bins.
        density : bool, default False
            If True, the total area of the histogram will be normalized to 1.
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
        x0, orient = self._column_and_orient()
        layer = _lt.DFLines.build_hist(
            self._df, x0, bins=bins, range=range, density=density, name=name,
            orient=orient, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        if self._update_label:
            ax_label = "density" if density else "count"
            if orient.is_vertical:
                canvas.y.label.text = ax_label
            else:
                canvas.x.label.text = ax_label
        return canvas.add_layer(layer)

    def add_kde(
        self,
        *,
        band_width: float | None = None,
        name: str | None = None,
        color: NStr | None = None,
        width: str | None = None,
        style: NStr | None = None,
    ):
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
        WrappedLines
            Line collection layer.
        """
        canvas = self._canvas()
        x0, orient = self._column_and_orient()
        layer = _lt.DFLines.build_kde(
            self._df, x0, band_width=band_width, name=name,
            orient=orient, color=color, width=width, style=style,
            backend=canvas._get_backend(),
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.with_color(layer._color_by.by, palette=canvas._color_palette)
        elif color is None:
            layer.with_color(canvas._color_palette.next())
        if self._update_label:
            ax_label = "density"
            if orient.is_vertical:
                canvas.y.label.text = ax_label
            else:
                canvas.x.label.text = ax_label
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


class CatAggPlotter:
    ...
