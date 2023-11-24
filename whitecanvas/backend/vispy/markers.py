from __future__ import annotations

from vispy.scene import visuals

import numpy as np
from numpy.typing import NDArray
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Symbol, LineStyle, FacePattern


@check_protocol(MarkersProtocol)
class Markers(visuals.Markers):
    def __init__(self, xdata, ydata):
        pos = np.stack([xdata, ydata], axis=1)
        super().__init__(pos=pos, edge_width=0, face_color="blue")

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self._data["a_position"]

    def _plt_set_data(self, xdata, ydata):
        self.set_data(pos=np.stack([xdata, ydata], axis=1))

    ##### HasSymbol protocol #####
    def _plt_get_symbol(self) -> Symbol:
        return self.symbol[0]

    def _plt_set_symbol(self, symbol: Symbol):
        self.symbol = symbol.value

    def _plt_get_symbol_size(self) -> float:
        return self._data["a_size"][0]

    def _plt_set_symbol_size(self, size: float):
        self.set_data(size=size)

    ##### HasFace protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self._data["a_bg_color"][0]

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self.set_data(face_color=color)

    def _plt_get_face_pattern(self) -> FacePattern:
        return FacePattern.SOLID

    def _plt_set_face_pattern(self, pattern: FacePattern):
        pass  # TODO

    ##### HasEdges protocol #####
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self._data["a_fg_color"][0]

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_data(edge_color=color)

    def _plt_get_edge_width(self) -> float:
        return self._data["a_edgewidth"][0]

    def _plt_set_edge_width(self, width: float):
        self.set_data(edge_width=width)

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle.SOLID

    def _plt_set_edge_style(self, style: LineStyle):
        pass
