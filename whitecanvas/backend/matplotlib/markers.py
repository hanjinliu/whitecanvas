from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib.markers as mmarkers
import matplotlib.transforms as mtransforms
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Symbol, FacePattern, LineStyle


def _get_path(symbol: Symbol):
    marker_obj = mmarkers.MarkerStyle(symbol.value)
    return marker_obj.get_path().transformed(marker_obj.get_transform())


@check_protocol(MarkersProtocol)
class Markers(PathCollection):
    def __init__(self, xdata, ydata):
        offsets = np.stack([xdata, ydata], axis=1)
        self._symbol = Symbol.CIRCLE
        super().__init__(
            (_get_path(self._symbol),),
            sizes=[6] * len(offsets),
            offsets=offsets,
            offset_transform=plt.gca().transData,
        )
        self.set_transform(mtransforms.IdentityTransform())
        self._edge_style = LineStyle.SOLID

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.get_visible()

    def _plt_set_visible(self, visible: bool):
        self.set_visible(visible)

    def _plt_get_zorder(self) -> int:
        return self.get_zorder()

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        offsets = self.get_offsets()
        return offsets[:, 0], offsets[:, 1]

    def _plt_set_data(self, xdata, ydata):
        data = np.stack([xdata, ydata], axis=1)
        self.set_offsets(data)

    ##### HasSymbol protocol #####

    def _plt_get_symbol(self) -> Symbol:
        return self._symbol

    def _plt_set_symbol(self, symbol: Symbol):
        path = _get_path(symbol)
        self.set_paths([path] * len(self.get_offsets()))
        self._symbol = symbol

    def _plt_get_symbol_size(self) -> float:
        return float(np.sqrt(self.get_sizes()[0]))

    def _plt_set_symbol_size(self, size: float):
        self.set_sizes([size**2] * len(self.get_offsets()))

    ##### HasFaces protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self.get_facecolor()[0]

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self.set_facecolor([color] * len(self.get_offsets()))

    def _plt_get_face_pattern(self) -> FacePattern:
        return FacePattern(self.get_hatch())

    def _plt_set_face_pattern(self, pattern: FacePattern):
        self.set_hatch(pattern.value)

    ##### HasEdges protocol #####

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_edgecolor()[0]

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_edgecolor([color] * len(self.get_offsets()))

    def _plt_get_edge_style(self) -> LineStyle:
        return self._edge_style

    def _plt_set_edge_style(self, style: LineStyle):
        self.set_linestyle([style.value] * len(self.get_offsets()))
        self._edge_style = style

    def _plt_get_edge_width(self) -> float:
        return self.get_linewidth()[0]

    def _plt_set_edge_width(self, width: float):
        self.set_linewidth([width] * len(self.get_offsets()))
