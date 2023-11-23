from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from matplotlib.collections import LineCollection
from whitecanvas.protocols import ErrorbarProtocol, check_protocol
from whitecanvas.types import LineStyle, Orientation


@check_protocol(ErrorbarProtocol)
class Errorbars(LineCollection):
    def __init__(self, t, y0, y1, orient: Orientation):
        if orient.is_vertical:
            starts = np.stack([t, y0], axis=1)
            ends = np.stack([t, y1], axis=1)
        else:
            starts = np.stack([y0, t], axis=1)
            ends = np.stack([y1, t], axis=1)
        segments = [[start, end] for start, end in zip(starts, ends)]
        super().__init__(segments)
        self._capsize = 0
        self._nbars = len(segments)

    def _plt_set_segments(self, segments, cap0, cap1):
        assert len(cap0) in (0, self._nbars)
        assert len(cap1) in (0, self._nbars)
        self.set_segments(segments + cap0 + cap1)

    def _plt_get_segments(self):  # bars, lowercaps, uppercaps
        segs = self.get_segments()
        nsegs = len(segs)
        if nsegs == self._nbars:
            return segs, [], []
        else:
            return segs[:nsegs], segs[nsegs : nsegs * 2], segs[nsegs * 2 :]

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.get_visible()

    def _plt_set_visible(self, visible: bool):
        self.set_visible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

    ##### OrientedXYYDataProtocol #####
    def _plt_get_data_all(self):
        seg, _, _ = self._plt_get_segments()
        if len(seg) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        seg_arr = np.stack(seg, axis=0)
        x0 = seg_arr[:, 0, 0]
        x1 = seg_arr[:, 1, 0]
        y0 = seg_arr[:, 0, 1]
        y1 = seg_arr[:, 1, 1]
        return x0, x1, y0, y1

    def _plt_get_vertical_data(self):
        x0, _, y0, y1 = self._plt_get_data_all()
        return x0, y0, y1

    def _plt_get_horizontal_data(self):
        x0, x1, y0, _ = self._plt_get_data_all()
        return y0, x0, x1

    def _plt_set_vertical_data(self, x, y0, y1):
        _, cap0, cap1 = self._plt_get_segments()
        starts = np.stack([x, y0], axis=1)
        ends = np.stack([x, y1], axis=1)
        segments = [[start, end] for start, end in zip(starts, ends)]
        if self._capsize > 0:
            _c = np.array([self._capsize / 2, 0])
            cap0 = [[start - _c, start + _c] for start in starts]
            cap1 = [[end - _c, end + _c] for end in ends]
        else:
            cap0 = []
            cap1 = []
        self._plt_set_segments(segments, cap0, cap1)

    def _plt_set_horizontal_data(self, y, x0, x1):
        _, cap0, cap1 = self._plt_get_segments()
        starts = np.stack([x0, y], axis=1)
        ends = np.stack([x1, y], axis=1)
        segments = [[start, end] for start, end in zip(starts, ends)]
        if self._capsize > 0:
            _c = np.array([0, self._capsize / 2])
            cap0 = [[start - _c, start + _c] for start in starts]
            cap1 = [[end - _c, end + _c] for end in ends]
        else:
            cap0 = []
            cap1 = []
        self._plt_set_segments(segments, cap0, cap1)

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> float:
        return self.get_linewidth()[0]

    def _plt_set_edge_width(self, width: float):
        self.set_linewidth(width)

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle(self.get_linestyle()[0])

    def _plt_set_edge_style(self, style: LineStyle):
        self.set_linestyle(style.value)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_color()[0]

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_color(color)

    def _plt_get_antialias(self) -> bool:
        return self.get_antialiased()[0]

    def _plt_set_antialias(self, antialias: bool):
        self.set_antialiased(antialias)

    ##### ErrorbarProtocol #####

    def _plt_get_capsize(self) -> float:
        return self._capsize

    def _plt_set_capsize(self, capsize: float, orient: Orientation):
        self._capsize = capsize
        if orient.is_vertical:
            self._plt_set_vertical_data(*self._plt_get_vertical_data())
        else:
            self._plt_set_horizontal_data(*self._plt_get_horizontal_data())
