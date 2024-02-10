from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from vispy.scene import visuals

from whitecanvas.backend import _not_implemented
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.utils.normalize import as_array_1d, as_color_array
from whitecanvas.utils.type_check import is_real_number


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

    def _plt_set_hover_text(self, text: list[str]):
        # TODO: not used yet
        self._hover_texts = text

    def _plt_connect_pick_event(self, callback):
        pass


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
        self._data = data

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        if len(self._lines) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.array([line.width for line in self._lines], dtype=np.float32)

    def _plt_set_edge_width(self, width):
        if is_real_number(width):
            for item in self._lines:
                item.set_data(width=width)
        else:
            widths = as_array_1d(width)
            if widths.size != len(self._lines):
                raise ValueError(
                    f"width must be a scalar or an array of length {len(self._lines)}"
                )
            for item, w in zip(self._lines, widths):
                item.set_data(width=w)

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_style()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        if len(self._lines) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack([line.color for line in self._lines], axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        col = as_color_array(color, len(self._lines))
        for item, c in zip(self._lines, col):
            item.set_data(color=c)

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        for item in self._lines:
            item.antialias = antialias
        self._antialias = antialias

    def _plt_set_hover_text(self, text: list[str]):
        # TODO: not used yet
        self._hover_texts = text

    def _plt_connect_pick_event(self, callback):
        pass
