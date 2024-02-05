from __future__ import annotations

from enum import Enum
from typing import overload

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers._primitive import Band, Line
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.types import ArrayLike1D, ColorType, LineStyle, Orientation
from whitecanvas.utils.hist import get_hist_edges, histograms
from whitecanvas.utils.normalize import as_array_1d


class HistogramShape(Enum):
    step = "step"
    polygon = "polygon"
    bars = "bars"


class HistogramKind(Enum):
    count = "count"
    density = "density"
    frequency = "frequency"
    percent = "percent"


class LineFillBase(LayerContainer):
    def __init__(self, line: Line, fill: Band, name: str | None = None):
        super().__init__([line, fill], name=name)

    @property
    def line(self) -> Line:
        """The line layer."""
        return self._children[0]

    @property
    def fill(self) -> Band:
        """The fill layer."""
        return self._children[1]

    @property
    def orient(self) -> Orientation:
        """Orientation of the line and fill layers."""
        return self.fill.orient

    @property
    def color(self) -> NDArray[np.float32]:
        """Color of the layer."""
        return self.line.color

    @color.setter
    def color(self, color: ColorType):
        self.line.color = color
        self.fill.face.update(color=color, alpha=0.2)
        self.fill.edge.width = 0.0


class Histogram(LineFillBase):
    def __init__(
        self,
        data: NDArray[np.number],
        edges: NDArray[np.number],
        limits: tuple[float, float] | None,
        line: Line,
        fill: Band,
        shape: HistogramShape = HistogramShape.bars,
        kind: HistogramKind = HistogramKind.count,
        name: str | None = None,
    ):
        if name is None:
            name = "histogram"
        super().__init__(line, fill, name=name)
        self._data = data
        self._shape = shape
        self._kind = kind
        self._edges = edges
        self._limits = limits

    @property
    def data(self) -> NDArray[np.number]:
        """The data used to plot the histogram."""
        return self._data

    @data.setter
    def data(self, data: NDArray[np.number]):
        data = as_array_1d(data)
        xdata, ydata = self._calculate_xy(
            data, self._edges, self._shape, self._kind, self._limits, clip=True
        )  # fmt: skip
        self._update_internal(xdata, ydata)
        self._data = data

    def _update_internal(self, xdata: NDArray[np.number], ydata: NDArray[np.number]):
        if self.orient.is_vertical:
            self.line.data = xdata, ydata
        else:
            self.line.data = ydata, xdata
        self.fill.data = xdata, _prep_bottom(ydata), ydata

    @property
    def shape(self) -> HistogramShape:
        """The shape of the histogram."""
        return self._shape

    @shape.setter
    def shape(self, shape: str | HistogramShape):
        shape = HistogramShape(shape)
        xdata, ydata, _ = self._calculate_xy(
            self._data, self._edges, shape, self._kind, self._limits
        )  # fmt: skip
        self._update_internal(xdata, ydata)
        self._shape = shape

    @property
    def kind(self) -> HistogramKind:
        """The kind of the histogram."""
        return self._kind

    @kind.setter
    def kind(self, kind: str | HistogramKind):
        kind = HistogramKind(kind)
        xdata, ydata, _ = self._calculate_xy(
            self._data, self._edges, self._shape, kind, self._limits
        )  # fmt: skip
        self._update_internal(xdata, ydata)
        self._kind = kind

    @property
    def limits(self) -> tuple[float, float] | None:
        """The limits of the histogram."""
        return self._limits

    @limits.setter
    def limits(self, limits: tuple[float, float] | None):
        xdata, ydata, _ = self._calculate_xy(
            self._data, self._edges, self._shape, self._kind, limits
        )
        self._update_internal(xdata, ydata)
        self._limits = limits

    @property
    def edges(self) -> NDArray[np.number]:
        """The edges of the histogram."""
        return self._edges

    @edges.setter
    def edges(self, edges: NDArray[np.number]):
        edges = as_array_1d(edges)
        xdata, ydata, _ = self._calculate_xy(
            self._data, edges, self._shape, self._kind, self._limits
        )
        self._update_internal(xdata, ydata)
        self._edges = edges

    @overload
    def update_edges(self, bins: int, limits: tuple[float, float] | None = None):
        ...

    @overload
    def update_edges(self, edges: NDArray[np.number]):
        ...

    def update_edges(self, bins, limits=None):
        """
        Update the edges of the histogram.

        >>> hist.update_edges(20, limits=(0, 10))  # uniform bins
        >>> hist.update_edges([0, 2, 3, 5])  # non-uniform bins
        """
        if limits is not None and not isinstance(bins, (int, np.number)):
            raise TypeError("bins must be an integer when limits are specified.")
        edges = get_hist_edges([self._data], bins, limits)
        self.edges = edges

    @classmethod
    def from_array(
        cls,
        data: NDArray[np.number],
        shape: HistogramShape = HistogramShape.bars,
        kind: HistogramKind = HistogramKind.count,
        name: str | None = None,
        bins: int = 10,
        limits: tuple[float, float] | None = None,
        color: ColorType = "black",
        style: str | LineStyle = LineStyle.SOLID,
        width: float = 1.0,
        orient: str | Orientation = "vertical",
        backend: str | Backend | None = None,
    ) -> Histogram:
        """Create a histogram from an array."""
        shape = HistogramShape(shape)
        kind = HistogramKind(kind)
        ori = Orientation.parse(orient)
        xdata, ydata, edges = cls._calculate_xy(data, bins, shape, kind, limits)
        if ori.is_vertical:
            line = Line(
                xdata, ydata, color=color, style=style, width=width, backend=backend
            )  # fmt: skip
        else:
            line = Line(
                ydata, xdata, color=color, style=style, width=width, backend=backend
            )
        fill = Band(
            xdata, _prep_bottom(ydata), ydata, color=color, alpha=0.2, orient=ori,
            backend=backend,
        )  # fmt: skip
        return cls(data, edges, limits, line, fill, shape, kind, name=name)

    @staticmethod
    def _calculate_xy(
        data,
        bins: int | ArrayLike1D,
        shape: HistogramShape,
        kind: HistogramKind,
        limits: tuple[float, float] | None = None,
        clip: bool = True,
    ) -> tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number]]:
        if clip and limits is not None:
            data = np.clip(data, *limits)
        hist = histograms([data], bins, limits)
        shape = HistogramShape(shape)
        kind = HistogramKind(kind)
        if kind is HistogramKind.count:
            heights = hist.counts[0]
        elif kind is HistogramKind.density:
            heights = hist.density()[0]
        elif kind is HistogramKind.frequency:
            heights = hist.frequency()[0]
        elif kind is HistogramKind.percent:
            heights = hist.percent()[0]
        else:
            raise ValueError(f"Unknown kind {kind!r}.")

        if shape is HistogramShape.step:
            xdata = np.repeat(hist.edges, 2)
            ydata = np.concatenate([[0], np.repeat(heights, 2), [0]])
        elif shape is HistogramShape.polygon:
            centers = hist.centers()
            xdata = np.concatenate([[centers[0]], centers, [centers[-1]]])
            ydata = np.concatenate([[0], heights, [0]])
        elif shape is HistogramShape.bars:
            edges = hist.edges
            xdata = np.repeat(edges, 3)[1:-1]
            ydata = np.zeros_like(xdata)
            ydata[1::3] = ydata[2::3] = heights
        else:
            raise ValueError(f"Unknown shape {shape!r}.")
        return xdata, ydata, hist.edges


def _prep_bottom(ydata: NDArray[np.number]) -> NDArray[np.number]:
    return np.full_like(ydata, 0)


class Kde(LineFillBase):
    def __init__(
        self,
        data: NDArray[np.number],
        band_width: float,
        line: Line,
        fill: Band,
        name: str | None = None,
        bottom: float = 0.0,
        scale: float = 1.0,
    ):
        if name is None:
            name = "kde"
        super().__init__(line, fill, name=name)
        self._data = data
        self._bottom = bottom
        self._band_width = band_width
        self._scale = scale

    @property
    def data(self) -> NDArray[np.number]:
        """The data used to plot the histogram."""
        return self._data

    @data.setter
    def data(self, data: NDArray[np.number]):
        data = as_array_1d(data)
        xdata, ydata = self._calculate_params(
            data, self._band_width, self._bottom, self._scale
        )  # fmt: skip
        self._update_internal(xdata, ydata, self._bottom)
        self._data = data

    def _update_internal(
        self, xdata: NDArray[np.number], ydata: NDArray[np.number], bottom: float
    ):
        if self.orient.is_vertical:
            self.line.data = xdata, ydata
        else:
            self.line.data = ydata, xdata
        self.fill.data = xdata, np.full_like(xdata, bottom), ydata

    @property
    def band_width(self) -> float:
        """The band width of the kernel density estimation."""
        return self._band_width

    @band_width.setter
    def band_width(self, band_width: float):
        xdata, ydata, bw = self._calculate_params(
            self._data, band_width, self._bottom, self._scale
        )  # fmt: skip
        self._update_internal(xdata, ydata, self._bottom)
        self._band_width = bw

    @property
    def bottom(self) -> float:
        """The bottom value of the fill."""
        return self._bottom

    @bottom.setter
    def bottom(self, bottom: float):
        xdata, ydata, _ = self._calculate_params(
            self._data, self._band_width, bottom, self._scale
        )  # fmt: skip
        self._update_internal(xdata, ydata, bottom)
        self._bottom = bottom

    @property
    def scale(self) -> float:
        """The scale of the kernel density estimation."""
        return self._scale

    @scale.setter
    def scale(self, scale: float):
        xdata, ydata, _ = self._calculate_params(
            self._data, self._band_width, self._bottom, scale
        )  # fmt: skip
        self._update_internal(xdata, ydata, self._bottom)
        self._scale = scale

    @classmethod
    def from_array(
        cls,
        data: ArrayLike1D,
        bottom: float = 0.0,
        scale: float = 1.0,
        *,
        name: str | None = None,
        band_width: float | None = None,
        color: ColorType = "blue",
        style: str | LineStyle = LineStyle.SOLID,
        width: float = 1.0,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: Backend | str | None = None,
    ):
        data = as_array_1d(data)
        x, y1, bw = cls._calculate_params(data, band_width, bottom, scale)
        if orient.is_vertical:
            line = Line(x, y1, color=color, style=style, width=width, backend=backend)
        else:
            line = Line(y1, x, color=color, style=style, width=width, backend=backend)
        fill = Band(
            x, np.full_like(x, bottom), y1, color=color, alpha=0.2, orient=orient,
            backend=backend,
        )  # fmt: skip
        return Kde(data, bw, line, fill, name=name, bottom=bottom, scale=scale)

    @staticmethod
    def _calculate_params(
        data: NDArray[np.number],
        band_width: float,
        bottom: float = 0.0,
        scale: float = 1.0,
    ) -> tuple[NDArray[np.number], NDArray[np.number], float]:
        from whitecanvas.utils.kde import gaussian_kde

        data = as_array_1d(data)
        kde = gaussian_kde(data, bw_method=band_width)

        sigma = np.sqrt(kde.covariance[0, 0])
        pad = sigma * 2.5
        x = np.linspace(data.min() - pad, data.max() + pad, 100)
        y1 = kde(x) * scale + bottom
        return x, y1, kde.factor
