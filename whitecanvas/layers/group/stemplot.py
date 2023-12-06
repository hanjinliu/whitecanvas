from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import Orientation, XYData, ArrayLike1D
from whitecanvas.layers._primitive import Markers, MultiLine
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.utils.normalize import normalize_xy
from whitecanvas.backend import Backend


class StemPlot(ListLayerGroup):
    def __init__(
        self,
        markers: Markers,
        lines: MultiLine,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
    ):
        self._orient = Orientation.parse(orient)
        super().__init__([markers, lines], name=name)

    def _default_ordering(self, n: int) -> list[int]:
        assert n == 2
        return [1, 0]

    @property
    def data(self) -> XYData:
        """XYData as (x, height)."""
        mdata = self.markers.data
        if self.orient.is_vertical:
            xdata = mdata.x
            ydata = mdata.y - self.bottom
        else:
            xdata = mdata.x - self.bottom
            ydata = mdata.y
        return XYData(xdata, ydata)

    @property
    def bottom(self) -> NDArray[np.floating]:
        """Bottom of the stem."""
        if self.orient.is_vertical:
            return np.array([d.y[0] for d in self.lines.data])
        else:
            return np.array([d.x[0] for d in self.lines.data])

    @property
    def top(self) -> NDArray[np.floating]:
        """Top of the stem."""
        if self.orient.is_vertical:
            return self.markers.data.y
        else:
            return self.markers.data.x

    @property
    def markers(self) -> Markers:
        """Markers layer."""
        return self._children[0]

    @property
    def lines(self) -> MultiLine:
        """Lines for the stems."""
        return self._children[1]

    @property
    def orient(self) -> Orientation:
        """Orientation of the stem plot."""
        return self._orient

    def bbox_hint(self) -> NDArray[np.float64]:
        xmin, xmax, ymin, ymax = self.markers.bbox_hint()
        # NOTE: min/max is nan-safe (returns nan)
        if self.orient.is_vertical:
            ymin = min(ymin, 0)
            ymax = max(ymax, 0)
        else:
            xmin = min(xmin, 0)
            xmax = max(xmax, 0)
        return xmin, xmax, ymin, ymax

    @classmethod
    def from_arrays(
        cls,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        bottom: ArrayLike1D | float = 0,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> StemPlot:
        xdata, ydata = normalize_xy(xdata, ydata)
        top = ydata + bottom
        return Markers(xdata, top, name=name, backend=backend).with_stem(
            orient, bottom=bottom
        )
