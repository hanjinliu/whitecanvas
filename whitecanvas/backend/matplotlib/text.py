from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text as mplText

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.types import Alignment, FacePattern, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol


@check_protocol(TextProtocol)
class Text(mplText, MplLayer):
    def __init__(self, x: float, y: float, text: str):
        super().__init__(
            x,
            y,
            text=text,
            verticalalignment='baseline',
            horizontalalignment='left',
            transform=plt.gca().transData,
            clip_on=False,
        )
        self._bbox_props = {}

    ##### TextProtocol #####

    def _plt_get_text(self) -> str:
        return self.get_text()

    def _plt_set_text(self, text: list[str]):
        self.set_text(text)

    def _plt_get_text_color(self):
        return np.array(self.get_color())

    def _plt_set_text_color(self, color):
        self.set_color(color)

    def _plt_get_text_size(self) -> float:
        return self.get_fontsize()

    def _plt_set_text_size(self, size: float):
        self.set_fontsize(size)

    def _plt_get_text_position(self) -> tuple[float, float]:
        return self.get_position()

    def _plt_set_text_position(self, position: tuple[float, float]):
        self.set_position(position)

    def _plt_get_text_anchor(self) -> Alignment:
        va = self.get_verticalalignment()
        ha = self.get_horizontalalignment()
        v = _VERTICAL_ALIGNMENTS[va]
        h = _HORIZONTAL_ALIGNMENTS[ha]
        return Alignment.merge(v, h)

    def _plt_set_text_anchor(self, anc: Alignment):
        """Set the text position."""
        v, h = anc.split()
        va = _VERTICAL_ALIGNMENTS_INV[v]
        ha = _HORIZONTAL_ALIGNMENTS_INV[h]
        self.set_verticalalignment(va)
        self.set_horizontalalignment(ha)

    def _plt_get_text_rotation(self) -> float:
        return self.get_rotation()

    def _plt_set_text_rotation(self, rotation: float):
        self.set_rotation(rotation)

    def _plt_get_text_fontfamily(self) -> str:
        return self.get_fontfamily()

    def _plt_set_text_fontfamily(self, fontfamily: str):
        self.set_fontfamily(fontfamily)

    ##### HasFaces #####

    def _plt_get_face_color(self):
        patch = self.get_bbox_patch()
        if patch is None:
            return np.array([0, 0, 0, 0])

        return np.array(patch.get_facecolor())

    def _plt_set_face_color(self, color):
        self._bbox_props['facecolor'] = color
        self.set_bbox(self._bbox_props)

    def _plt_get_face_pattern(self) -> FacePattern:
        patch = self.get_bbox_patch()
        if patch is None:
            return FacePattern.SOLID
        return FacePattern(patch.get_hatch())

    def _plt_set_face_pattern(self, pattern: FacePattern):
        if pattern is FacePattern.SOLID:
            ptn = None
        else:
            ptn = pattern.value
        self._bbox_props['hatch'] = ptn
        self.set_bbox(self._bbox_props)

    def _plt_get_edge_color(self):
        patch = self.get_bbox_patch()
        if patch is None:
            return np.array([0, 0, 0, 0])
        return np.array(patch.get_edgecolor())

    def _plt_set_edge_color(self, color):
        self._bbox_props['edgecolor'] = color
        self.set_bbox(self._bbox_props)

    def _plt_get_edge_width(self) -> float:
        patch = self.get_bbox_patch()
        if patch is None:
            return 0
        return patch.get_linewidth()

    def _plt_set_edge_width(self, width: float):
        self._bbox_props['linewidth'] = width
        self.set_bbox(self._bbox_props)

    def _plt_get_edge_style(self) -> LineStyle:
        patch = self.get_bbox_patch()
        if patch is None:
            return LineStyle.SOLID
        return LineStyle(patch.get_linestyle())

    def _plt_set_edge_style(self, style: LineStyle):
        self._bbox_props['linestyle'] = style.value
        self.set_bbox(self._bbox_props)


_VERTICAL_ALIGNMENTS = {
    "bottom": Alignment.BOTTOM,
    "baseline": Alignment.BOTTOM,
    "center": Alignment.CENTER,
    "center_baseline": Alignment.CENTER,
    "top": Alignment.TOP,
}
_VERTICAL_ALIGNMENTS_INV = {v: k for k, v in _VERTICAL_ALIGNMENTS.items()}
_HORIZONTAL_ALIGNMENTS = {
    "left": Alignment.LEFT,
    "center": Alignment.CENTER,
    "right": Alignment.RIGHT,
}
_HORIZONTAL_ALIGNMENTS_INV = {v: k for k, v in _HORIZONTAL_ALIGNMENTS.items()}
