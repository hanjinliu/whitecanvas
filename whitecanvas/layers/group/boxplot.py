from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, LineStyle, Orientation
from whitecanvas.layers.primitive import HeteroBars, MultiLine, Markers
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.utils.normalize import as_color_array, norm_color


class BoxPlot(ListLayerGroup):
    """
    A group for boxplot.

    Children layers are:
    - HeteroBars (boxes)
    - MultiLine (whiskers)
    - MultiLine (median line)
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
        boxes: HeteroBars,
        whiskers: MultiLine,
        medians: MultiLine,
        # outliers: Markers | None = None,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
    ):
        super().__init__([boxes, whiskers, medians], name=name)
        self._orient = Orientation.parse(orient)

    @property
    def boxes(self) -> HeteroBars:
        return self._children[0]

    @property
    def whiskers(self) -> MultiLine:
        return self._children[1]

    @property
    def medians(self) -> MultiLine:
        return self._children[2]

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike],
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        box_width: float = 0.5,
        capsize: float = 0.3,
        face_color: ColorType | list[ColorType] = "blue",
        edge_color: ColorType = "black",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: str | Backend | None = None,
    ):
        x, data = check_array_input(x, data)
        ori = Orientation.parse(orient)
        face_color = as_color_array(face_color, len(x))
        edge_color = norm_color(edge_color)
        agg_values: list[NDArray[np.number]] = []
        for d in data:
            agg_values.append(np.quantile(d, [0, 0.25, 0.5, 0.75, 1]))
        agg_arr = np.stack(agg_values, axis=1)
        box = HeteroBars(
            x, agg_arr[3], agg_arr[1], name=name, orient=ori, bar_width=box_width,
            pattern=pattern, color=face_color, alpha=alpha, backend=backend,
        ).with_edge(color=edge_color)  # fmt: skip
        if ori.is_vertical:
            segs = _xyy_to_segments(
                x, agg_arr[0], agg_arr[1], agg_arr[3], agg_arr[4], capsize
            )
            medsegs = [
                [(x0 - box_width / 2, y0), (x0 + box_width / 2, y0)]
                for x0, y0 in zip(x, agg_arr[2])
            ]
        else:
            segs = _yxx_to_segments(
                x, agg_arr[0], agg_arr[1], agg_arr[3], agg_arr[4], capsize
            )
            medsegs = [
                [(x0, y0 - box_width / 2), (x0, y0 + box_width / 2)]
                for x0, y0 in zip(x, agg_arr[2])
            ]
        whiskers = MultiLine(
            segs, name=name, style=LineStyle.SOLID, alpha=alpha, backend=backend,
            color=edge_color,
        )  # fmt: skip
        medians = MultiLine(
            medsegs, name="medians", color=edge_color, alpha=alpha, backend=backend,
        )  # fmt: skip

        return cls(box, whiskers, medians, name=name, orient=ori)

    @property
    def orient(self) -> Orientation:
        return self._orient


def _xyy_to_segments(
    x: ArrayLike,
    y0: ArrayLike,
    y1: ArrayLike,
    y2: ArrayLike,
    y3: ArrayLike,
    capsize: float,
):
    """
    ──┬──  <-- y3
      │    <-- y2

      │    <-- y1
    ──┴──  <-- y0
      ↑
      x
    """
    v0 = np.stack([x, y0], axis=1)
    v1 = np.stack([x, y1], axis=1)
    v2 = np.stack([x, y2], axis=1)
    v3 = np.stack([x, y3], axis=1)
    segments_0 = [[s0, s1] for s0, s1 in zip(v0, v1)]
    segments_1 = [[s2, s3] for s2, s3 in zip(v2, v3)]
    if capsize > 0:
        _c = np.array([capsize / 2, 0])
        cap0 = [[s0 - _c, s0 + _c] for s0 in v0]
        cap1 = [[s3 - _c, s3 + _c] for s3 in v3]
    else:
        cap0 = []
        cap1 = []
    return segments_0 + segments_1 + cap0 + cap1


def _yxx_to_segments(
    y: ArrayLike,
    x0: ArrayLike,
    x1: ArrayLike,
    x2: ArrayLike,
    x3: ArrayLike,
    capsize: float,
):
    """
    |        |
     ───  ───  <-- y
    |        |
    ↑  ↑  ↑  ↑
    x0 x1 x2 x3
    """
    v0 = np.stack([x0, y], axis=1)
    v1 = np.stack([x1, y], axis=1)
    v2 = np.stack([x2, y], axis=1)
    v3 = np.stack([x3, y], axis=1)
    segments_0 = [[s0, s1] for s0, s1 in zip(v0, v1)]
    segments_1 = [[s2, s3] for s2, s3 in zip(v2, v3)]

    if capsize > 0:
        _c = np.array([0, capsize / 2])
        cap0 = [[s0 - _c, s0 + _c] for s0 in v0]
        cap1 = [[s3 - _c, s3 + _c] for s3 in v3]
    else:
        cap0 = []
        cap1 = []
    return segments_0 + segments_1 + cap0 + cap1