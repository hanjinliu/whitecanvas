from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, LineStyle, _Void
from whitecanvas.layers.primitive import Bars, MultiLine, Markers
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.utils.normalize import as_array_1d


class BoxPlot(ListLayerGroup):
    """
    A group for boxplot.

    Children layers are:
    - Bars (boxes)
    - MultiLine (whiskers)
    - MultiLine (median line)
    - Markers (mean values)
    - Markers (outliers)

     ──┬──  <-- max
       │
    ┌──┴──┐ <-- 75% quantile
    │  o  │ <-- mean
    ╞═════╡ <-- median
    └──┬──┘ <-- 25% quantile
       │
     ──┴──  <-- min
    """

    def __init__(
        self,
        boxes: Bars,
        whiskers: MultiLine,
        median_line: MultiLine,
        means: Markers,
        outliers: Markers,
        *,
        name: str | None = None,
        box_width: float = 0.3,
    ):
        ...
