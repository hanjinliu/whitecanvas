from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import Hatch, LineStyle
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number


class BaseMockLayer:
    def __init__(self):
        self._visible = True

    def _plt_get_visible(self):
        return self._visible

    def _plt_set_visible(self, visible):
        self._visible = visible


class MockHasData(BaseMockLayer):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def _plt_get_data(self) -> Sequence[NDArray[np.float32]]:
        return self._data

    def _plt_set_data(self, *data: Sequence[NDArray[np.float32]]):
        self._data = data


class MockHasFaces:
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        if not hasattr(self, "_face_color"):
            self._face_color = np.zeros(4, dtype=np.float32)
        return self._face_color

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self._face_color = color

    def _plt_get_face_hatch(self) -> Hatch:
        if not hasattr(self, "_face_hatch"):
            self._face_hatch = Hatch.SOLID
        return self._face_hatch

    def _plt_set_face_hatch(self, pattern: Hatch):
        self._face_hatch = pattern


class MockHasEdges:
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        if not hasattr(self, "_edge_color"):
            self._edge_color = np.zeros(4, dtype=np.float32)
        return self._edge_color

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._edge_color = color

    def _plt_get_edge_style(self) -> LineStyle:
        if not hasattr(self, "_edge_style"):
            self._edge_style = LineStyle.SOLID
        return self._edge_style

    def _plt_set_edge_style(self, style: LineStyle):
        self._edge_style = style

    def _plt_get_edge_width(self) -> float:
        if not hasattr(self, "_edge_width"):
            self._edge_width = 1.0
        return self._edge_width

    def _plt_set_edge_width(self, width: float):
        self._edge_width = width


class MockHasMultiFaces:
    def _plt_get_ndata(self) -> int:
        raise NotImplementedError

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        if not hasattr(self, "_face_color"):
            ndata = self._plt_get_ndata()
            self._face_color = np.zeros((ndata, 4), dtype=np.float32)
        return self._face_color

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        self._face_color = as_color_array(color, self._plt_get_ndata())

    def _plt_get_face_hatch(self) -> list[Hatch]:
        if not hasattr(self, "_face_hatch"):
            self._face_hatch = [Hatch.SOLID] * self._plt_get_ndata()
        return self._face_hatch

    def _plt_set_face_hatch(self, pattern: Hatch | Sequence[Hatch]):
        if isinstance(pattern, (str, Hatch)):
            pattern = [Hatch(pattern)] * self._plt_get_ndata()
        else:
            pattern = [Hatch(p) for p in pattern]
        self._face_hatch = pattern


class MockHasMultiEdges:
    def _plt_get_ndata(self) -> int:
        raise NotImplementedError

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        if not hasattr(self, "_edge_color"):
            ndata = self._plt_get_ndata()
            self._edge_color = np.zeros((ndata, 4), dtype=np.float32)
        return self._edge_color

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._edge_color = as_color_array(color, self._plt_get_ndata())

    def _plt_get_edge_style(self) -> list[LineStyle]:
        if not hasattr(self, "_edge_style"):
            self._edge_style = [LineStyle.SOLID] * self._plt_get_ndata()
        return self._edge_style

    def _plt_set_edge_style(self, style: LineStyle | Sequence[LineStyle]):
        if isinstance(style, (str, LineStyle)):
            style = [LineStyle(style)] * self._plt_get_ndata()
        else:
            style = [LineStyle(s) for s in style]
        self._edge_style = style

    def _plt_get_edge_width(self) -> float:
        if not hasattr(self, "_edge_width"):
            self._edge_width = np.ones(self._plt_get_ndata(), dtype=np.float32)
        return self._edge_width

    def _plt_set_edge_width(self, width: float | Sequence[float]):
        if is_real_number(width):
            width = np.full(self._plt_get_ndata(), width, dtype=np.float32)
        else:
            width = np.asarray(width)
        self._edge_width = width


class MockHasMouseEvents:
    def _plt_set_hover_text(self, text: list[str]) -> None:
        pass

    def _plt_connect_pick_event(self, callback) -> None:
        pass
