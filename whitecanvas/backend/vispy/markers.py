from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from vispy.scene import visuals

from whitecanvas.backend import _not_implemented

# from vispy.visuals.filters.markers import MarkerPickingFilter
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Symbol
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number


@check_protocol(MarkersProtocol)
class Markers(visuals.Markers):
    def __init__(self, xdata, ydata):
        pos = np.stack([xdata, ydata], axis=1)
        super().__init__(pos=pos, edge_width=0, face_color="blue")
        self.unfreeze()
        self._hover_texts: list[str] | None = None
        # self._picking_filter = MarkerPickingFilter()
        # self.attach(self._picking_filter)

    def _plt_get_ndata(self):
        return len(self._data["a_position"])

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        data = self._data["a_position"]
        return data[:, 0], data[:, 1]

    def _plt_set_data(self, xdata, ydata):
        ndata = self._plt_get_ndata()
        size = self._plt_get_symbol_size()
        edge_width = self._plt_get_edge_width()
        face_color = self._plt_get_face_color()
        edge_color = self._plt_get_edge_color()
        if xdata.size > ndata:
            size = np.concatenate([size, np.full(xdata.size - ndata, size[-1])])
            edge_width = np.concatenate(
                [edge_width, np.full(xdata.size - ndata, edge_width[-1])]
            )
            face_color = np.concatenate(
                [face_color, np.full((xdata.size - ndata, 4), face_color[-1])]
            )
            edge_color = np.concatenate(
                [edge_color, np.full((xdata.size - ndata, 4), edge_color[-1])]
            )
        elif xdata.size < ndata:
            size = size[: xdata.size]
            edge_width = edge_width[: xdata.size]
            face_color = face_color[: xdata.size]
            edge_color = edge_color[: xdata.size]
        self.set_data(
            pos=np.stack([xdata, ydata], axis=1),
            size=size,
            edge_width=edge_width,
            face_color=face_color,
            edge_color=edge_color,
            symbol=self.symbol[0],
        )

    ##### HasSymbol protocol #####
    def _plt_get_symbol(self) -> Symbol:
        if self._data["a_position"].shape[0] == 0:
            return Symbol.CIRCLE
        sym = self.symbol[0]
        if sym == "clobber":
            return Symbol.TRIANGLE_LEFT
        elif sym == "diamond":
            return Symbol.DIAMOND
        elif sym == "-":
            return Symbol.HBAR
        elif sym is None:
            return Symbol.CIRCLE
        return Symbol(sym)

    def _plt_set_symbol(self, symbol: Symbol):
        if symbol is Symbol.DIAMOND:
            self.symbol = "diamond"
        elif symbol is Symbol.HBAR:
            self.symbol = "-"
        elif symbol is Symbol.TRIANGLE_LEFT:
            # NOTE: vispy does not have "<"
            self.symbol = "clobber"
        else:
            self.symbol = symbol.value

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return self._data["a_size"]

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if is_real_number(size):
            size = np.full(self._plt_get_ndata(), size)
        if size.shape[0] == 0:
            return
        self.set_data(
            pos=self._data["a_position"],
            size=size,
            edge_width=self._plt_get_edge_width(),
            face_color=self._plt_get_face_color(),
            edge_color=self._plt_get_edge_color(),
            symbol=self.symbol,
        )

    ##### HasFace protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self._data["a_bg_color"]

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, self._plt_get_ndata())
        if color.shape[0] == 0:
            return
        self.set_data(
            pos=self._data["a_position"],
            size=self._plt_get_symbol_size(),
            edge_width=self._plt_get_edge_width(),
            face_color=color,
            edge_color=self._plt_get_edge_color(),
            symbol=self.symbol,
        )

    _plt_get_face_hatch, _plt_set_face_hatch = _not_implemented.face_hatches()

    ##### HasEdges protocol #####
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self._data["a_fg_color"]

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, self._plt_get_ndata())
        if color.shape[0] == 0:
            return
        self.set_data(
            pos=self._data["a_position"],
            size=self._plt_get_symbol_size(),
            edge_width=self._plt_get_edge_width(),
            face_color=self._plt_get_face_color(),
            edge_color=color,
            symbol=self.symbol,
        )

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self._data["a_edgewidth"]

    def _plt_set_edge_width(self, width: float):
        if isinstance(width, float):
            width = np.full(self._plt_get_ndata(), width)
        if width.shape[0] == 0:
            return
        self.set_data(
            pos=self._data["a_position"],
            size=self._plt_get_symbol_size(),
            edge_width=width,
            face_color=self._plt_get_face_color(),
            edge_color=self._plt_get_edge_color(),
            symbol=self.symbol,
        )

    _plt_get_edge_style, _plt_set_edge_style = _not_implemented.edge_styles()

    def _plt_connect_pick_event(self, callback):
        # TODO: implement this
        # https://github.com/napari/napari/blob/main/napari/layers/points/points.py#L1617
        pass

    def _plt_set_hover_text(self, text: list[str]):
        # TODO: not used yet
        self._hover_texts = text

    def _compute_bounds(self, axis, view):
        # override to fix the bounds computation
        pos = self._data["a_position"]
        if pos is None or pos.size == 0:
            return None
        if pos.shape[1] > axis:
            return (pos[:, axis].min(), pos[:, axis].max())
        else:
            return (0, 0)
