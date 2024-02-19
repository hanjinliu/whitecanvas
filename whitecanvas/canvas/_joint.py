from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
)

from whitecanvas import layers as _l
from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.canvas._grid import CanvasGrid
from whitecanvas.canvas._linker import link_axes
from whitecanvas.layers import group as _lg
from whitecanvas.layers import tabular as _lt
from whitecanvas.types import (
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    HistBinType,
    HistogramKind,
    HistogramShape,
    KdeBandWidthType,
    LegendLocation,
    LegendLocationStr,
    Orientation,
    Symbol,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas import Canvas
    from whitecanvas.canvas import _namespaces as _ns
    from whitecanvas.canvas.dataframe import JointCatPlotter
    from whitecanvas.layers import _mixin
    from whitecanvas.layers.tabular._dataframe import DataFrameWrapper

    NStr = str | Sequence[str]

_DF = TypeVar("_DF")


_0_or_1 = Literal[0, 1]


class JointGrid(CanvasGrid):
    """
    Grid with a main (joint) canvas and two marginal canvases.

    The marginal canvases shares the x-axis and y-axis with the main canvas.
    """

    def __init__(
        self,
        loc: tuple[_0_or_1, _0_or_1] = (1, 0),
        palette: str | ColormapType | None = None,
        ratio: int = 4,
        backend: Backend | str | None = None,
    ):
        widths = [1, 1]
        heights = [1, 1]
        rloc, cloc = loc
        if rloc not in (0, 1) or cloc not in (0, 1):
            raise ValueError(f"Invalid location {loc!r}.")
        widths[rloc] = heights[cloc] = ratio
        super().__init__(widths, heights, backend=Backend(backend))
        self._main_canvas = self.add_canvas(rloc, cloc, palette=palette)
        self._x_canvas = self.add_canvas(1 - rloc, cloc)
        self._y_canvas = self.add_canvas(rloc, 1 - cloc)

        # flip the axes if needed
        if rloc == 0:
            self._x_canvas.y.flipped = True
            self._x_namespace_canvas = self._x_canvas
            self._main_canvas.x.ticks.visible = False
            self._title_namespace_canvas = self._main_canvas
        else:
            self._x_namespace_canvas = self._main_canvas
            self._x_canvas.x.ticks.visible = False
            self._title_namespace_canvas = self._x_canvas
        if cloc == 0:
            self._ynamespace_canvas = self._main_canvas
            self._y_canvas.y.ticks.visible = False
        else:
            self._y_canvas.x.flipped = True
            self._ynamespace_canvas = self._y_canvas
            self._main_canvas.y.ticks.visible = False

        self._backend_object._plt_set_spacings(10, 10)

        # link axes
        self._x_linker = link_axes([self._main_canvas.x, self._x_canvas.x])
        self._y_linker = link_axes([self._main_canvas.y, self._y_canvas.y])

        # joint plotter
        self._x_plotters = []
        self._y_plotters = []

    def _iter_x_plotters(self) -> Iterator[MarginalPlotter]:
        if len(self._x_plotters) == 0:
            yield MarginalHistPlotter(Orientation.VERTICAL)
        else:
            yield from self._x_plotters

    def _iter_y_plotters(self) -> Iterator[MarginalPlotter]:
        if len(self._y_plotters) == 0:
            yield MarginalHistPlotter(Orientation.HORIZONTAL)
        else:
            yield from self._y_plotters

    @property
    def x_canvas(self) -> Canvas:
        """The canvas at the x-axis."""
        return self._x_canvas

    @property
    def y_canvas(self) -> Canvas:
        """The canvas at the y-axis."""
        return self._y_canvas

    @property
    def main_canvas(self) -> Canvas:
        """The main (joint) canvas."""
        return self._main_canvas

    @property
    def x(self) -> _ns.XAxisNamespace:
        """The x-axis namespace of the joint grid."""
        return self._x_namespace_canvas.x

    @property
    def y(self) -> _ns.YAxisNamespace:
        """The y-axis namespace of the joint grid."""
        return self._ynamespace_canvas.y

    @property
    def title(self) -> _ns.TitleNamespace:
        """Title namespace of the joint grid."""
        return self._title_namespace_canvas.title

    def cat(
        self,
        data: _DF,
        x: str | None = None,
        y: str | None = None,
        *,
        update_labels: bool = True,
    ) -> JointCatPlotter[Self, _DF]:
        """Create a joint categorical canvas."""
        from whitecanvas.canvas.dataframe import JointCatPlotter

        return JointCatPlotter(self, data, x, y, update_labels=update_labels)

    def _link_marginal_to_main(self, layer: _l.Layer, main: _l.Layer) -> None:
        # TODO: this is not the only thing to be done
        main.events.visible.connect_setattr(layer, "visible")

    def add_legend(
        self,
        layers: Sequence[_l.Layer] | None = None,
        location: LegendLocation | LegendLocationStr = "top_right",
    ):
        self.main_canvas.add_legend(layers, location=location)
        return None

    def add_markers(
        self,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        *,
        name: str | None = None,
        symbol: Symbol | str | None = None,
        size: float | None = None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> _l.Markers[_mixin.ConstFace, _mixin.ConstEdge, float]:
        color = self._main_canvas._generate_colors(color)
        out = self._main_canvas.add_markers(
            xdata, ydata, name=name, symbol=symbol, size=size, color=color,
            alpha=alpha, hatch=hatch,
        )  # fmt: skip
        for _x_plt in self._iter_x_plotters():
            xlayer = _x_plt.add_layer_for_markers(
                xdata, color, hatch, backend=self._backend
            )
            self.x_canvas.add_layer(xlayer)
            self._link_marginal_to_main(xlayer, out)
        for _y_plt in self._iter_y_plotters():
            ylayer = _y_plt.add_layer_for_markers(
                ydata, color, hatch, backend=self._backend
            )
            self.y_canvas.add_layer(ylayer)
            self._link_marginal_to_main(ylayer, out)
        self._autoscale_layers()
        return out

    def with_hist_x(
        self,
        *,
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        kind: str | HistogramKind = HistogramKind.density,
        shape: str | HistogramShape = HistogramShape.bars,
    ) -> Self:
        """
        Configure the x-marginal canvas to have a histogram.

        Parameters
        ----------
        bins : int or 1D array-like, default "auto"
            Bins of the histogram. This parameter will directly be passed
            to `np.histogram`.
        limits : (float, float), optional
            Limits in which histogram will be built. This parameter will equivalent to
            the `range` paraneter of `np.histogram`.
        shape : {"step", "polygon", "bars"}, default "bars"
            Shape of the histogram. This parameter defines how to convert the data into
            the line nodes.
        kind : {"count", "density", "probability", "frequency", "percent"}, optional
            Kind of the histogram.
        """
        self._x_plotters.append(
            MarginalHistPlotter(
                Orientation.VERTICAL, bins=bins, limits=limits, kind=kind, shape=shape
            )
        )
        return self

    def with_hist_y(
        self,
        *,
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        kind: str | HistogramKind = HistogramKind.density,
        shape: str | HistogramShape = HistogramShape.bars,
    ) -> Self:
        """
        Configure the y-marginal canvas to have a histogram.

        Parameters
        ----------
        bins : int or 1D array-like, default "auto"
            Bins of the histogram. This parameter will directly be passed
            to `np.histogram`.
        limits : (float, float), optional
            Limits in which histogram will be built. This parameter will equivalent to
            the `range` paraneter of `np.histogram`.
        shape : {"step", "polygon", "bars"}, default "bars"
            Shape of the histogram. This parameter defines how to convert the data into
            the line nodes.
        kind : {"count", "density", "probability", "frequency", "percent"}, optional
            Kind of the histogram.
        """
        self._y_plotters.append(
            MarginalHistPlotter(
                Orientation.HORIZONTAL, bins=bins, limits=limits, kind=kind, shape=shape
            )
        )
        return self

    def with_hist(
        self,
        *,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        limits: tuple[float, float] | None = None,
        kind: str | HistogramKind = HistogramKind.density,
        shape: str | HistogramShape = HistogramShape.bars,
    ) -> Self:
        """
        Configure both of the marginal canvases to have histograms.

        Parameters
        ----------
        bins : int or 1D array-like, default "auto"
            Bins of the histogram. This parameter will directly be passed
            to `np.histogram`.
        limits : (float, float), optional
            Limits in which histogram will be built. This parameter will equivalent to
            the `range` paraneter of `np.histogram`.
        shape : {"step", "polygon", "bars"}, default "bars"
            Shape of the histogram. This parameter defines how to convert the data into
            the line nodes.
        kind : {"count", "density", "probability", "frequency", "percent"}, optional
            Kind of the histogram.
        """
        if isinstance(bins, tuple):
            bins_x, bins_y = bins
        else:
            bins_x = bins_y = bins
        self.with_hist_x(bins=bins_x, limits=limits, kind=kind, shape=shape)
        self.with_hist_y(bins=bins_y, limits=limits, kind=kind, shape=shape)
        return self

    def with_kde_x(
        self,
        *,
        width: float | None = None,
        band_width: KdeBandWidthType = "scott",
        fill_alpha: float = 0.2,
    ) -> Self:
        """
        Configure the x-marginal canvas to have a kernel density estimate (KDE) plot.

        Parameters
        ----------
        width : float, optional
            Width of the line. Use theme default if not specified.
        band_width : "scott", "silverman" or float, default "scott"
            Bandwidth of the kernel.
        fill_alpha : float, default 0.2
            Alpha value of the fill color.
        """
        width = theme._default("line.width", width)
        self._x_plotters.append(
            MarginalKdePlotter(
                Orientation.VERTICAL,
                width=width,
                band_width=band_width,
                fill_alpha=fill_alpha,
            )
        )
        return self

    def with_kde_y(
        self,
        *,
        width: float | None = None,
        band_width: KdeBandWidthType = "scott",
        fill_alpha: float = 0.2,
    ) -> Self:
        """
        Configure the y-marginal canvas to have a kernel density estimate (KDE) plot.

        Parameters
        ----------
        width : float, optional
            Width of the line. Use theme default if not specified.
        band_width : "scott", "silverman" or float, default "scott"
            Bandwidth of the kernel.
        fill_alpha : float, default 0.2
            Alpha value of the fill color.
        """
        width = theme._default("line.width", width)
        self._y_plotters.append(
            MarginalKdePlotter(
                Orientation.HORIZONTAL,
                width=width,
                band_width=band_width,
                fill_alpha=fill_alpha,
            )
        )
        return self

    def with_kde(
        self,
        *,
        width: float | None = None,
        band_width: KdeBandWidthType = "scott",
        fill_alpha: float = 0.2,
    ) -> Self:
        """
        Configure both of the marginal canvases to have KDE plots.

        Parameters
        ----------
        width : float, optional
            Width of the line. Use theme default if not specified.
        band_width : "scott", "silverman" or float, default "scott"
            Bandwidth of the kernel.
        fill_alpha : float, default 0.2
            Alpha value of the fill color.
        """
        self.with_kde_x(width=width, band_width=band_width, fill_alpha=fill_alpha)
        self.with_kde_y(width=width, band_width=band_width, fill_alpha=fill_alpha)
        return self

    def with_rug_x(self, *, width: float | None = None) -> Self:
        """
        Configure the x-marginal canvas to have a rug plot.

        Parameters
        ----------
        width : float, optional
            Width of the line. Use theme default if not specified.
        """
        width = theme._default("line.width", width)
        self._x_plotters.append(MarginalRugPlotter(Orientation.VERTICAL, width=width))
        return self

    def with_rug_y(self, *, width: float | None = None) -> Self:
        """
        Configure the y-marginal canvas to have a rug plot.

        Parameters
        ----------
        width : float, optional
            Width of the line. Use theme default if not specified.
        """
        width = theme._default("line.width", width)
        self._y_plotters.append(MarginalRugPlotter(Orientation.HORIZONTAL, width=width))
        return self

    def with_rug(self, *, width: float | None = None) -> Self:
        """
        Configure both of the marginal canvases to have rug plots.

        Parameters
        ----------
        width : float, optional
            Width of the line. Use theme default if not specified.
        """
        self.with_rug_x(width=width)
        self.with_rug_y(width=width)
        return self

    def _autoscale_layers(self):
        for layer in self.x_canvas.layers:
            if isinstance(layer, (_l.Rug, _lt.DFRug)):
                ylow, yhigh = self.x_canvas.y.lim
                layer.update_length((yhigh - ylow) * 0.1)
        for layer in self.y_canvas.layers:
            if isinstance(layer, (_l.Rug, _lt.DFRug)):
                xlow, xhigh = self.y_canvas.x.lim
                layer.update_length((xhigh - xlow) * 0.1)


class MarginalPlotter(ABC):
    def __init__(self, orient: str | Orientation):
        self._orient = Orientation.parse(orient)

    @abstractmethod
    def add_layer_for_markers(
        self,
        data: ArrayLike1D,
        color: ColorType,
        hatch: Hatch = Hatch.SOLID,
        backend: str | Backend | None = None,
    ) -> _l.Layer:
        ...

    @abstractmethod
    def add_layer_for_cat_markers(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: NStr | None = None,
        hatch: NStr | None = None,
        backend: str | Backend | None = None,
    ) -> _l.Layer:
        ...

    @abstractmethod
    def add_layer_for_cat_hist2d(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: str | None = None,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        limits: tuple[float, float] | None = None,
        backend: str | Backend | None = None,
    ) -> _l.Layer:
        ...


class MarginalHistPlotter(MarginalPlotter):
    def __init__(
        self,
        orient: str | Orientation,
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        kind: str | HistogramKind = "density",
        shape: str | HistogramShape = "bars",
    ):
        super().__init__(orient)
        self._bins = bins
        self._limits = limits
        self._kind = HistogramKind(kind)
        self._shape = HistogramShape(shape)

    def add_layer_for_markers(
        self,
        data: ArrayLike1D,
        color: ColorType,
        hatch: Hatch = Hatch.SOLID,
        backend: str | Backend | None = None,
    ) -> _lg.Histogram:
        return _lg.Histogram.from_array(
            data,
            shape=self._shape,
            kind=self._kind,
            color=color,
            orient=self._orient,
            bins=self._bins,
            limits=self._limits,
            backend=backend,
        )

    def add_layer_for_cat_markers(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: NStr | None = None,
        hatch: NStr | None = None,
        backend: str | Backend | None = None,
    ) -> _lt.DFHistograms[_DF]:
        return _lt.DFHistograms.from_table(
            df, value, orient=self._orient, color=color, hatch=hatch,
            bins=self._bins, limits=self._limits, kind=self._kind, shape=self._shape,
            backend=backend,
        )  # fmt: skip

    def add_layer_for_cat_hist2d(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: str | None = None,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        limits: tuple[float, float] | None = None,
        backend: str | Backend | None = None,
    ) -> _lt.DFHistograms[_DF]:
        if self._bins != "auto":
            bins = self._bins
        if self._limits is not None:
            limits = self._limits
        return _lt.DFHistograms.from_table(
            df, value, orient=self._orient, color=color, bins=bins, limits=limits,
            kind=self._kind, shape=self._shape, backend=backend,
        )  # fmt: skip


class MarginalKdePlotter(MarginalPlotter):
    def __init__(
        self,
        orient: str | Orientation,
        width: float = 1.0,
        band_width: KdeBandWidthType = "scott",
        fill_alpha: float = 0.2,
    ):
        super().__init__(orient)
        self._width = width
        self._band_width = band_width
        self._fill_alpha = fill_alpha

    def add_layer_for_markers(
        self,
        data: ArrayLike1D,
        color: ColorType,
        hatch: Hatch = Hatch.SOLID,
        backend: str | Backend | None = None,
    ) -> _lg.Kde:
        out = _lg.Kde.from_array(
            data, color=color, orient=self._orient, band_width=self._band_width,
            width=self._width, backend=backend,
        )  # fmt: skip
        out.fill_alpha = self._fill_alpha
        return out

    def add_layer_for_cat_markers(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: NStr | None = None,
        hatch: NStr | None = None,
        backend: str | Backend | None = None,
    ) -> _lt.DFKde[_DF]:
        out = _lt.DFKde.from_table(
            df, value, orient=self._orient, color=color, hatch=hatch,
            width=self._width, backend=backend,
        )  # fmt: skip
        for layer in out.base:
            layer.fill_alpha = self._fill_alpha
        return out

    def add_layer_for_cat_hist2d(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: str | None = None,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        limits: tuple[float, float] | None = None,
        backend: str | Backend | None = None,
    ) -> _lt.DFKde[_DF]:
        out = _lt.DFKde.from_table(
            df, value, orient=self._orient, color=color, band_width=self._band_width,
            width=self._width, backend=backend,
        )  # fmt: skip
        for layer in out.base:
            layer.fill_alpha = self._fill_alpha
        return out


class MarginalRugPlotter(MarginalPlotter):
    def __init__(
        self,
        orient: str | Orientation,
        width: float = 1.0,
        length: float = 0.1,
    ):
        super().__init__(orient)
        self._width = width
        self._length = length

    def add_layer_for_markers(
        self,
        data: ArrayLike1D,
        color: ColorType,
        hatch: Hatch = Hatch.SOLID,
        backend: str | Backend | None = None,
    ) -> _l.Rug:
        return _l.Rug(
            data, high=self._length, color=color, orient=self._orient,
            width=self._width, backend=backend
        )  # fmt: skip

    def add_layer_for_cat_markers(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: NStr | None = None,
        hatch: NStr | None = None,
        backend: str | Backend | None = None,
    ) -> _lt.DFRug[_DF]:
        return _lt.DFRug.from_table(
            df,
            value,
            high=self._length,
            orient=self._orient,
            color=color,
            width=self._width,
            backend=backend,
        )

    def add_layer_for_cat_hist2d(
        self,
        df: DataFrameWrapper[_DF],
        value: str,
        color: str | None = None,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        limits: tuple[float, float] | None = None,
        backend: str | Backend | None = None,
    ) -> _lt.DFRug[_DF]:
        return _lt.DFRug.from_table(
            df,
            value,
            high=self._length,
            orient=self._orient,
            color=color,
            width=self._width,
            backend=backend,
        )


# class MarginalBoxPlotter
# class MarginalViolinPlotter
