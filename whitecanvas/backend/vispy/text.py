from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from vispy.scene import visuals

from whitecanvas.backend import _not_implemented
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.types import Alignment, LineStyle
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number

FONT_SIZE_FACTOR = 2.0


@check_protocol(TextProtocol)
class Texts(visuals.Text):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        if x.size > 0:
            pos = np.stack([x, y, np.zeros(x.size)], axis=1)
        else:
            pos = np.array([[0, 0, 0]], dtype=np.float32)
            text = [""]
        super().__init__(text, pos=pos)
        self.unfreeze()
        self._alignment = Alignment.BOTTOM_LEFT
        # NOTE: vispy does not support empty text layer. Here we use "_is_empty" to
        # specify whether the layer is empty, and pretend to be empty.
        self._is_empty = x.size == 0

    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### TextProtocol #####

    def _plt_get_text(self) -> list[str]:
        if self._is_empty:
            return []
        return self.text

    def _plt_set_text(self, text: list[str]):
        if self._is_empty:
            self.text = [""]
        else:
            self.text = text

    def _plt_get_ndata(self) -> int:
        if self._is_empty:
            return 0
        return len(self.text)

    def _plt_get_text_color(self):
        return self.color.rgba

    def _plt_set_text_color(self, color):
        col = as_color_array(color, self._plt_get_ndata())
        if not self._is_empty:
            self.color = col

    def _plt_get_text_size(self) -> NDArray[np.floating]:
        return np.full(self._plt_get_ndata(), self.font_size * FONT_SIZE_FACTOR)

    def _plt_set_text_size(self, size: float | NDArray[np.floating]):
        if is_real_number(size):
            self.font_size = size / FONT_SIZE_FACTOR
        else:
            candidates = np.unique(size)
            if candidates.size == 1:
                self.font_size = candidates[0] / FONT_SIZE_FACTOR
            elif candidates.size == 0:
                pass
            else:
                warnings.warn(
                    "vispy Text layer does not support different font sizes. Set to "
                    "the average size.",
                    UserWarning,
                    stacklevel=4,
                )
                self.font_size = np.mean(size) / FONT_SIZE_FACTOR

    def _plt_get_text_position(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        if self._is_empty:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        return self.pos[:, 0], self.pos[:, 1]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        x, y = position
        if x.size == 0:
            self.pos = np.array([[0, 0, 0]], dtype=np.float32)
        else:
            self.pos = np.stack([x, y, np.zeros(x.size)], axis=1)
        self._is_empty = x.size == 0

    def _plt_get_text_anchor(self) -> Alignment:
        return self._alignment

    def _plt_set_text_anchor(self, anc: Alignment):
        va, ha = anc.split()
        self.anchors = va.value, ha.value
        self._alignment = anc

    def _plt_get_text_rotation(self) -> NDArray[np.floating]:
        return -self.rotation  # the +/- is reversed compared to other backends

    def _plt_set_text_rotation(self, rotation: float | NDArray[np.floating]):
        if self._is_empty:
            if is_real_number(rotation):
                self.rotation = np.array([-rotation])
            else:
                if rotation.size != 0:
                    raise ValueError(f"zero text but got {rotation.size} inputs.")
        else:
            if is_real_number(rotation):
                rotation = np.full(self._plt_get_ndata(), rotation)
            self.rotation = -rotation

    def _plt_get_text_fontfamily(self) -> str:
        return self.face

    def _plt_set_text_fontfamily(self, fontfamily: str):
        self.face = fontfamily

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
