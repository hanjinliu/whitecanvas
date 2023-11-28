from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import Orientation
from whitecanvas.layers.primitive import Markers, Line
from whitecanvas.layers.group._collections import ListLayerGroup


class StemPlot(ListLayerGroup):
    def __init__(
        self,
        markers: Markers,
        lines: Line,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
    ):
        super().__init__([markers, lines], name=name)
        self._orient = Orientation.parse(orient)

    @property
    def markers(self) -> Markers:
        return self._children[0]

    @property
    def lines(self) -> Line:
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
