from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import Orientation
from whitecanvas.layers.primitive import Markers, MultiLine
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
    def markers(self) -> Markers:
        return self._children[0]

    @property
    def lines(self) -> MultiLine:
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
        xdata,
        top,
        bottom=None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> StemPlot:
        xdata, top = normalize_xy(xdata, top)
        return Markers(xdata, top, name=name, backend=backend).with_stem(
            orient, bottom=bottom
        )
