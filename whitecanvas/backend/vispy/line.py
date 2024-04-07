from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from vispy.scene import visuals

from whitecanvas.backend import _not_implemented
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.utils.normalize import as_color_array
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


def _make_connection(data: NDArray[np.number]):
    if len(data) == 0:
        return np.empty((0, 2), dtype=np.uint32)
    connections = []
    npoints = 0
    for seg in data:
        conn = np.column_stack(
            [
                np.arange(npoints, npoints + len(seg) - 1, dtype=np.uint32),
                np.arange(npoints + 1, npoints + len(seg), dtype=np.uint32),
            ]
        )
        connections.append(conn)
        npoints += len(seg)
    return np.concatenate(connections)


def _safe_concat(data: list[NDArray[np.floating]]):
    if len(data) == 0:
        return None
    return np.concatenate(data, axis=0)


@check_protocol(MultiLineProtocol)
class MultiLine(visuals.Line):
    def __init__(self, data: list[NDArray[np.float32]]):
        self._data_raw = data
        self._seg_colors = np.ones((len(data), 4), dtype=np.float32)
        super().__init__(
            pos=_safe_concat(data),
            antialias=True,
            connect=_make_connection(data),
        )
        self.unfreeze()

    def _plt_get_ndata(self):
        return len(self._data_raw)

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self._data_raw

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        connections = _make_connection(data)
        self._data_raw = data
        self.set_data(pos=_safe_concat(data), connect=connections)

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        ndata = len(self._data_raw)
        return np.full(ndata, self.width, dtype=np.float32)

    def _plt_set_edge_width(self, width):
        if is_real_number(width):
            self.set_data(width=width)
        else:
            candidates = np.unique(width)
            if len(candidates) == 1:
                self.set_data(width=width[0])
            elif len(candidates) == 0:
                return
            else:
                warnings.warn(
                    "vispy backend does not support different edge width for each line "
                    "segment. The first value is used for all line",
                    UserWarning,
                    stacklevel=2,
                )
                self.set_data(width=width[0])

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_styles()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self._seg_colors

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        ndata = len(self._data_raw)
        colors = as_color_array(color, ndata)
        raw_colors = []
        for color, seg in zip(colors, self._data_raw):
            raw_colors.append(np.repeat(color[np.newaxis], len(seg), axis=0))
        self.set_data(color=_safe_concat(raw_colors))
        self._seg_colors = colors

    def _plt_get_antialias(self) -> bool:
        return self.antialias

    def _plt_set_antialias(self, antialias: bool):
        self.antialias = antialias

    def _plt_set_hover_text(self, text: list[str]):
        # TODO: not used yet
        self._hover_texts = text

    def _plt_connect_pick_event(self, callback):
        pass
