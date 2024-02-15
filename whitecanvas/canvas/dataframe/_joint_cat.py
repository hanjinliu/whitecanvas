from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Sequence,
    TypeVar,
)

from whitecanvas.canvas.dataframe._feature_cat import CatPlotter
from whitecanvas.layers import tabular as _lt
from whitecanvas.layers.tabular import _jitter
from whitecanvas.types import ColormapType, HistBinType

if TYPE_CHECKING:
    from whitecanvas.canvas import JointGrid

    NStr = str | Sequence[str]

_C = TypeVar("_C", bound="JointGrid")
_DF = TypeVar("_DF")


class JointCatPlotter(CatPlotter[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
        x: str | None,
        y: str | None,
        update_labels: bool = False,
    ):
        super().__init__(canvas, df, x, y, update_labels=update_labels)

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
        grid = self._canvas()
        main = grid.main_canvas
        xj = _jitter.IdentityJitter(self._get_x())
        yj = _jitter.IdentityJitter(self._get_y())
        layer = _lt.DFMarkers(
            self._df, xj, yj, name=name, color=color, hatch=hatch,
            size=size, symbol=symbol, backend=grid._backend,
        )  # fmt: skip
        if color is not None and not layer._color_by.is_const():
            layer.update_color(layer._color_by.by, palette=main._color_palette)
        elif color is None:
            layer.update_color(main._color_palette.next())
        main.add_layer(layer)
        for _x_plt in grid._iter_x_plotters():
            xlayer = _x_plt.add_layer_for_cat_markers(
                self._df, self._get_x(), color=color, hatch=hatch, backend=grid._backend
            )  # fmt: skip
            grid.x_canvas.add_layer(xlayer)
        for _y_plt in grid._iter_y_plotters():
            ylayer = _y_plt.add_layer_for_cat_markers(
                self._df, self._get_y(), color=color, hatch=hatch, backend=grid._backend
            )
            grid.y_canvas.add_layer(ylayer)
        return layer

    def add_hist2d(
        self,
        *,
        cmap: ColormapType = "inferno",
        name: str | None = None,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        rangex: tuple[float, float] | None = None,
        rangey: tuple[float, float] | None = None,
        density: bool = False,
    ) -> _lt.DFHeatmap[_DF]:
        """
        Add 2-D histogram of given x/y columns.

        Parameters
        ----------
        cmap : colormap-like, default "inferno"
            Colormap to use for the heatmap.
        name : str, optional
            Name of the layer.
        bins : int, array, str or tuple of them, default "auto"
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
        grid = self._canvas()
        main = grid.main_canvas
        layer = _lt.DFHeatmap.build_hist(
            self._df, self._get_x(), self._get_y(), cmap=cmap, name=name, bins=bins,
            range=(rangex, rangey), density=density, backend=grid._backend,
        )  # fmt: skip
        main.add_layer(layer)
        for _x_plt in grid._iter_x_plotters():
            xlayer = _x_plt.add_layer_for_cat_hist2d(
                self._df, self._get_x(), bins=bins, limits=rangex, backend=grid._backend
            )  # fmt: skip
            grid.x_canvas.add_layer(xlayer)
        for _y_plt in grid._iter_y_plotters():
            ylayer = _y_plt.add_layer_for_cat_hist2d(
                self._df, self._get_y(), bins=bins, limits=rangey, backend=grid._backend
            )  # fmt: skip
            grid.y_canvas.add_layer(ylayer)
        return layer
