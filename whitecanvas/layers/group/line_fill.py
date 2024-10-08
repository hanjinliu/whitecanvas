from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers._deserialize import construct_layers
from whitecanvas.layers._primitive import Band, Line
from whitecanvas.layers._primitive.line import _SingleLine
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    HistBinType,
    HistogramKind,
    HistogramShape,
    KdeBandWidthType,
    LineStyle,
    Orientation,
    OrientationLike,
    XYData,
)
from whitecanvas.utils.hist import get_hist_edges, histograms
from whitecanvas.utils.normalize import as_array_1d

if TYPE_CHECKING:
    from typing_extensions import Self

_L = TypeVar("_L", bound=_SingleLine)


class LineFillBase(LayerContainer, Generic[_L]):
    _ATTACH_TO_AXIS = True

    def __init__(self, line: _L, fill: Band, name: str | None = None):
        super().__init__([line, fill], name=name)
        self._fill_alpha = 0.2

    @property
    def line(self) -> _L:
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
        self.fill.face.update(color=color, alpha=self._fill_alpha)
        self.fill.edge.update(color=color, alpha=1.0)

    @property
    def fill_alpha(self) -> float:
        """The alpha value applied to the fill region compared to the line."""
        return self._fill_alpha

    @fill_alpha.setter
    def fill_alpha(self, alpha: float):
        self._fill_alpha = alpha
        self.fill.face.alpha = alpha
        self.fill.edge.alpha = alpha

    def _as_legend_item(self) -> _legend.BarLegendItem:
        edge = _legend.EdgeInfo(self.line.color, self.line.width, self.line.style)
        return _legend.BarLegendItem(self.fill.face, edge)


class Histogram(LineFillBase[Line]):
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
        xdata, ydata, _ = self._calculate_xy(
            data, self._edges, self._shape, self._kind, self._limits, clip=True
        )  # fmt: skip
        self._update_internal(xdata, ydata)
        self._data = data

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        data = d["data"]
        edges = d["edges"]
        limits = d.get("limits")
        children = construct_layers(d["children"], backend=backend)
        shape = HistogramShape(d.get("shape", "bars"))
        kind = HistogramKind(d.get("kind", "count"))
        return cls(data, edges, limits, *children, shape, kind, name=d.get("name"))

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "data": self.data,
            "edges": self.edges,
            "limits": self.limits,
            "shape": self.shape.value,
            "kind": self.kind.value,
        }

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
    def update_edges(
        self, bins: HistBinType, limits: tuple[float, float] | None = None
    ): ...

    @overload
    def update_edges(self, edges: NDArray[np.number]): ...

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
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        color: ColorType = "black",
        style: str | LineStyle = LineStyle.SOLID,
        width: float = 1.0,
        orient: OrientationLike = "vertical",
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
        bins: HistBinType,
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
        elif kind is HistogramKind.probability:
            heights = hist.probability()[0]
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
    return np.full(ydata.size, 0)


class Kde(LineFillBase[Line]):
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

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        children = construct_layers(d["children"], backend=backend)
        return cls(
            d["data"],
            d["band_width"],
            *children,
            bottom=d.get("bottom", 0.0),
            scale=d.get("scale", 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "data": self.data,
            "band_width": self.band_width,
            "bottom": self.bottom,
            "scale": self.scale,
        }

    def _update_internal(
        self, xdata: NDArray[np.number], ydata: NDArray[np.number], bottom: float
    ):
        if self.orient.is_vertical:
            self.line.data = xdata, ydata
        else:
            self.line.data = ydata, xdata
        self.fill.data = xdata, np.full(xdata.size, bottom), ydata

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
        band_width: KdeBandWidthType = "scott",
        color: ColorType = "blue",
        style: str | LineStyle = LineStyle.SOLID,
        width: float = 1.0,
        orient: OrientationLike = "vertical",
        backend: Backend | str | None = None,
    ):
        data = as_array_1d(data)
        x, y1, bw = cls._calculate_params(data, band_width, bottom, scale)
        ori = Orientation.parse(orient)
        if ori.is_vertical:
            line = Line(x, y1, color=color, style=style, width=width, backend=backend)
        else:
            line = Line(y1, x, color=color, style=style, width=width, backend=backend)
        fill = Band(
            x, np.full(x.size, bottom), y1, color=color, alpha=0.2, orient=ori,
            backend=backend,
        )  # fmt: skip
        return Kde(data, bw, line, fill, name=name, bottom=bottom, scale=scale)

    @staticmethod
    def _calculate_params(
        data: NDArray[np.number],
        band_width: KdeBandWidthType,
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


class Area(LineFillBase[Line]):
    @classmethod
    def from_arrays(
        cls,
        x: ArrayLike1D,
        y: ArrayLike1D,
        bottom: ArrayLike1D | float = 0.0,
        orient: OrientationLike = "vertical",
        name: str | None = None,
        backend: Backend | str | None = None,
    ) -> Area:
        if name is None:
            name = "area"
        ori = Orientation.parse(orient)
        line = Line(x, y + bottom, name="line", backend=backend)
        fill = Band(x, bottom, y + bottom, name="fill", orient=ori, backend=backend)
        return cls(line, fill, name=name)

    @property
    def data(self) -> XYData:
        """The data used to plot the histogram."""
        x, y0, y1 = self.fill.data
        return XYData(x, y1 - y0)

    @data.setter
    def data(self, data: XYData):
        x, y = data
        bottom = self.fill.data.y0
        self.line.data = x, y + bottom
        self.fill.data = x, bottom, y + bottom

    @property
    def bottom(self) -> ArrayLike1D:
        """The bottom value of the fill."""
        return self.fill.data.y0

    @property
    def orient(self) -> Orientation:
        """Orientation of the line and fill layers."""
        return self.fill.orient
