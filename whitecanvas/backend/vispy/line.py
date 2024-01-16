from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from vispy.scene import visuals

from whitecanvas.backend import _not_implemented
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol


@check_protocol(LineProtocol)
class MonoLine(visuals.Line):
    def __init__(self, xdata, ydata):
        data = np.stack([xdata, ydata], axis=1)
        super().__init__(data, antialias=True)
        self.unfreeze()

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        pos = self.pos
        return pos[:, 0], pos[:, 1]

    def _plt_set_data(self, xdata, ydata):
        self.set_data(np.stack([xdata, ydata], axis=1))

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> float:
        return self.width

    def _plt_set_edge_width(self, width: float):
        self.set_data(width=width)

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_style()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.color

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_data(color=color)

    def _plt_get_antialias(self) -> bool:
        return self.antialias

    def _plt_set_antialias(self, antialias: bool):
        self.antialias = antialias


@check_protocol(MultiLineProtocol)
class MultiLine(visuals.Compound):
    def __init__(self, data: list[NDArray[np.float32]]):
        items = []
        for seg in data:
            item = visuals.Line(seg, width=1, antialias=True)
            items.append(item)
        self._data = data
        self._antialias = True
        super().__init__(items)
        self.unfreeze()

    @property
    def _lines(self) -> list[visuals.Line]:
        return self._subvisuals

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self._data

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        ndata = len(data)
        nitem = len(self._lines)
        if ndata < nitem:
            for item in self._lines[ndata:]:
                self.remove_subvisual(item)
        else:
            for _ in range(ndata - nitem):
                item = visuals.Line()
                self.add_subvisual(item)
        for item, seg in zip(self._lines, data):
            item.set_data(seg)

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> float:
        if len(self._lines) == 0:
            return 0.0
        return self._lines[0].width

    def _plt_set_edge_width(self, width: float):
        for item in self._lines:
            item.set_data(width=width)

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_style()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        if len(self._lines) == 0:
            return np.zeros((4,), dtype=np.float32)
        return self._lines[0].color

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        for item in self._lines:
            item.set_data(color=color)

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        for item in self._lines:
            item.antialias = antialias
        self._antialias = antialias
