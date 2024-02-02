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


class Histogram(LayerContainer):
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
        super().__init__([line, fill], name=name)
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
        xdata, ydata = _calculate_xy(
            data, self._edges, self._shape, self._kind, self._limits, clip=True
        )  # fmt: skip
        self._update_internal(xdata, ydata)
        self._data = data

    def _update_internal(self, xdata: NDArray[np.number], ydata: NDArray[np.number]):
        if self.orient.is_vertical:
            self.line.data = xdata, ydata
        else:
            self.line.data = ydata, xdata
        self.fill.data = xdata, np.zeros_like(ydata), ydata

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
        return self.fill.orient

    @property
    def shape(self) -> HistogramShape:
        """The shape of the histogram."""
        return self._shape

    @shape.setter
    def shape(self, shape: str | HistogramShape):
        shape = HistogramShape(shape)
        xdata, ydata, _ = _calculate_xy(
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
        xdata, ydata, _ = _calculate_xy(
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
        xdata, ydata, _ = _calculate_xy(
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
        xdata, ydata, _ = _calculate_xy(
            self._data, edges, self._shape, self._kind, self._limits
        )
        self._update_internal(xdata, ydata)
        self._edges = edges

    @property
    def color(self) -> NDArray[np.float32]:
        return self.line.color

    @color.setter
    def color(self, color: ColorType):
        self.line.color = color
        self.fill.face.update(color=color, alpha=0.2)

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
        xdata, ydata, edges = _calculate_xy(data, bins, shape, kind, limits)
        if ori.is_vertical:
            line = Line(
                xdata, ydata, color=color, style=style, width=width, backend=backend
            )  # fmt: skip
        else:
            line = Line(
                ydata, xdata, color=color, style=style, width=width, backend=backend
            )
        fill = Band(
            xdata, np.zeros_like(ydata), ydata, color=color, alpha=0.2, orient=ori,
            backend=backend,
        )  # fmt: skip
        return cls(data, edges, limits, line, fill, shape, kind, name=name)


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
