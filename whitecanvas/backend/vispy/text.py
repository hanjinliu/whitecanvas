from __future__ import annotations
import numpy as np

from vispy.scene import visuals

from whitecanvas.types import Alignment, FacePattern, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol


@check_protocol(TextProtocol)
class Text(visuals.Text):
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

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.zeros(4)

    def _plt_set_face_color(self, color):
        pass

    def _plt_get_face_pattern(self) -> FacePattern:
        return FacePattern.SOLID

    def _plt_set_face_pattern(self, pattern: FacePattern):
        pass

    ##### HasEdges #####

    def _plt_get_edge_color(self):
        return np.zeros(4)

    def _plt_set_edge_color(self, color):
        pass

    def _plt_get_edge_width(self) -> float:
        return 0.0

    def _plt_set_edge_width(self, width: float):
        pass

    def _plt_get_edge_style(self) -> LineStyle:
        return LineStyle.SOLID

    def _plt_set_edge_style(self, style: LineStyle):
        pass
