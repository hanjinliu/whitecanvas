from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from vispy.scene import visuals
from whitecanvas.protocols import LineProtocol, check_protocol
from whitecanvas.types import LineStyle


@check_protocol(LineProtocol)
class MonoLine(visuals.Line):
    def __init__(self, xdata, ydata):
        data = np.stack([xdata, ydata], axis=1)
        super().__init__(data, antialias=True)

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    def _plt_set_zorder(self, zorder: int):
        pass

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

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle.SOLID

    def _plt_set_edge_style(self, style: LineStyle):
        pass  # TODO

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.color

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self.set_data(color=color)

    def _plt_get_antialias(self) -> bool:
        return self.antialias

    def _plt_set_antialias(self, antialias: bool):
        self.antialias = antialias


@check_protocol(LineProtocol)
class MultiLine(visuals.Compound):
    def __init__(self, data: list[NDArray[np.float32]]):
        items = []
        for seg in data:
            item = visuals.Line(seg, width=1, antialias=True)
            items.append(item)
        self._data = data
        self._antialias = True
        self._width = np.ones(len(data))
        self._color = np.ones((len(data), 4), dtype=np.float32)
        super().__init__(items)

    @property
    def _lines(self) -> list[visuals.Line]:
        return self._subvisuals

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    def _plt_set_zorder(self, zorder: int):
        pass

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
    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.array([item.width for item in self._lines], dtype=np.float32)

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if np.isscalar(width):
            width = np.full(len(self._data), width)
        for w, item in zip(width, self._lines):
            item.set_data(width=w)

    def _plt_get_edge_style(self) -> LineStyle:
        return np.array([LineStyle.SOLID] * len(self._data), dtype=object)

    def _plt_set_edge_style(self, style):
        pass

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.stack([item.color for item in self._lines], axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = np.tile(color, (len(self._data), 1))
        for col, item in zip(color, self._lines):
            item.set_data(color=col)

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        for item in self._lines:
            item.antialias = antialias
        self._antialias = antialias
