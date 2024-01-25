from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from vispy.scene import visuals

from whitecanvas.backend import _not_implemented
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.types import Alignment, LineStyle
from whitecanvas.utils.normalize import as_color_array


@check_protocol(TextProtocol)
class Texts(visuals.Compound):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        super().__init__(
            [SingleText(x0, y0, text0) for x0, y0, text0 in zip(x, y, text)]
        )
        self.unfreeze()

    @property
    def subvisuals(self) -> list[SingleText]:
        return self._subvisuals

    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### TextProtocol #####

    def _plt_get_text(self) -> list[str]:
        return [t.text for t in self.subvisuals]

    def _plt_set_text(self, text: list[str]):
        for t, text0 in zip(self.subvisuals, text):
            t.text = text0

    def _plt_get_ndata(self) -> int:
        return len(self.subvisuals)

    def _plt_get_text_color(self):
        return np.array([t.color for t in self.subvisuals])

    def _plt_set_text_color(self, color):
        color = as_color_array(color, self._plt_get_ndata())
        for t, color0 in zip(self.subvisuals, color):
            t.color = color0

    def _plt_get_text_size(self) -> float:
        return [t.font_size for t in self.subvisuals]

    def _plt_set_text_size(self, size: float | NDArray[np.floating]):
        if isinstance(size, (int, float, np.number)):
            size = np.full(self._plt_get_ndata(), size)
        for t, size0 in zip(self.subvisuals, size):
            t.font_size = size0

    def _plt_get_text_position(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        pos = np.stack([np.array(t.pos[0, 1:]) for t in self.subvisuals], axis=0)
        return pos[:, 0], pos[:, 1]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        for t, x0, y0 in zip(self.subvisuals, *position):
            t.pos = x0, y0

    def _plt_get_text_anchor(self) -> list[Alignment]:
        return [t._alignment for t in self.subvisuals]

    def _plt_set_text_anchor(self, anc: Alignment | list[Alignment]):
        if isinstance(anc, Alignment):
            anc = [anc] * self._plt_get_ndata()
        for t, anc0 in zip(self.subvisuals, anc):
            va, ha = anc0.split()
            t.anchors = va.value, ha.value
            t._alignment = anc0

    def _plt_get_text_rotation(self) -> float:
        return np.array([t.rotation[0] for t in self.subvisuals])

    def _plt_set_text_rotation(self, rotation: float | NDArray[np.floating]):
        if isinstance(rotation, (int, float, np.number)):
            rotation = np.full(self._plt_get_ndata(), rotation)
        for t, rotation0 in zip(self.subvisuals, rotation):
            t.rotation = rotation0

    def _plt_get_text_fontfamily(self) -> list[str]:
        return [t.face for t in self.subvisuals]

    def _plt_set_text_fontfamily(self, fontfamily: str | list[str]):
        if isinstance(fontfamily, str):
            fontfamily = [fontfamily] * self._plt_get_ndata()
        for t, fontfamily0 in zip(self.subvisuals, fontfamily):
            t.face = fontfamily0

    def _plt_get_face_color(self):
        return np.zeros((self._plt_get_ndata(), 4))

    def _plt_set_face_color(self, color):
        pass

    _plt_get_face_hatch, _plt_set_face_hatch = _not_implemented.face_patterns()

    def _plt_get_edge_color(self):
        return np.zeros((self._plt_get_ndata(), 4))

    def _plt_set_edge_color(self, color):
        pass

    def _plt_get_edge_width(self) -> float:
        return np.zeros(self._plt_get_ndata())

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        pass

    def _plt_get_edge_style(self) -> LineStyle:
        return [LineStyle.SOLID] * self._plt_get_ndata()

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        pass


class SingleText(visuals.Text):
    def __init__(self, x: float, y: float, text: str):
        super().__init__(text=text, anchor_x="left", anchor_y="bottom")
        self._plt_set_text_position([x, y])
        self.unfreeze()
        self._alignment = Alignment.BOTTOM_LEFT

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### TextProtocol #####

    def _plt_get_text(self) -> str:
        return self.text

    def _plt_set_text(self, text: str):
        self.text = text

    def _plt_get_text_color(self):
        return self.color

    def _plt_set_text_color(self, color):
        self.color = color

    def _plt_get_text_size(self) -> float:
        return self.font_size

    def _plt_set_text_size(self, size: float):
        self.font_size = size

    def _plt_get_text_position(self) -> tuple[float, float]:
        return tuple(self.pos[0, 1:])

    def _plt_set_text_position(self, position: tuple[float, float]):
        self.pos = position

    def _plt_get_text_anchor(self) -> Alignment:
        return self._alignment

    def _plt_set_text_anchor(self, anc: Alignment):
        va, ha = anc.split()
        self.anchors = va.value, ha.value
        self._alignment = anc

    def _plt_get_text_rotation(self) -> float:
        return self.rotation[0]

    def _plt_set_text_rotation(self, rotation: float):
        self.rotation = rotation

    def _plt_get_text_fontfamily(self) -> str:
        return self.face

    def _plt_set_text_fontfamily(self, fontfamily: str):
        self.face = fontfamily
