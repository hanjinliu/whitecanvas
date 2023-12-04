from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.text import Text as mplText
from matplotlib.artist import Artist

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.types import Alignment, FacePattern, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol


class Texts(Artist, MplLayer):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        super().__init__()
        self._children = []
        for x0, y0, text0 in zip(x, y, text):
            self._children.append(
                mplText(
                    x0, y0, text0, verticalalignment="baseline",
                    horizontalalignment="left", clip_on=False,
                    color=np.array([0, 0, 0, 1], dtype=np.float32),
                )  # fmt: skip
            )

    def draw(self, renderer):
        for child in self.get_children():
            child.draw(renderer)

    def get_children(self) -> list[mplText]:
        return self._children

    def set_visible(self, visible: bool):
        for child in self.get_children():
            child.set_visible(visible)
        super().set_visible(visible)

    def set_transform(self, transform):
        for child in self.get_children():
            child.set_transform(transform)
        super().set_transform(transform)

    def _plt_get_text(self) -> list[str]:
        return [child.get_text() for child in self.get_children()]

    def _plt_set_text(self, text: list[str]):
        for child, text0 in zip(self.get_children(), text):
            child.set_text(str(text0))

    def _plt_get_text_color(self):
        return np.stack([child.get_color() for child in self.get_children()], axis=0)

    def _plt_set_text_color(self, color):
        for child, color0 in zip(self.get_children(), color):
            child.set_color(color0)

    def _plt_get_text_size(self) -> NDArray[np.float32]:
        return np.array(
            [child.get_fontsize() for child in self.get_children()], dtype=np.float32
        )

    def _plt_set_text_size(self, size: NDArray[np.float32]):
        for child, size0 in zip(self.get_children(), size):
            child.set_fontsize(size0)

    def _plt_get_text_position(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        ar = np.array([child.get_position() for child in self.get_children()])
        return ar[:, 0], ar[:, 1]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.float32], NDArray[np.float32]]
    ):
        x, y = position
        for child, x0, y0 in zip(self.get_children(), x, y):
            child.set_position((x0, y0))

    def _plt_get_text_anchor(self) -> Alignment:
        out = []
        for child in self.get_children():
            va = child.get_verticalalignment()
            ha = child.get_horizontalalignment()
            if (aln := _ALIGNMENTS.get((va, ha))) is None:
                v = _VERTICAL_ALIGNMENTS[va]
                h = _HORIZONTAL_ALIGNMENTS[ha]
                aln = Alignment.merge(v, h)
                _ALIGNMENTS[(va, ha)] = aln
                _ALIGNMENTS_INV[aln] = (va, ha)
            out.append(aln)
        return out

    def _plt_set_text_anchor(self, anc: Alignment | list[Alignment]):
        """Set the text position."""
        if isinstance(anc, Alignment):
            v, h = anc.split()
            va = _VERTICAL_ALIGNMENTS_INV[v]
            ha = _HORIZONTAL_ALIGNMENTS_INV[h]
            for child in self.get_children():
                child.set_verticalalignment(va)
                child.set_horizontalalignment(ha)
        else:
            for child, anc0 in zip(self.get_children(), anc):
                v, h = anc0.split()
                va = _VERTICAL_ALIGNMENTS_INV[v]
                ha = _HORIZONTAL_ALIGNMENTS_INV[h]
                child.set_verticalalignment(va)
                child.set_horizontalalignment(ha)

    def _plt_get_text_rotation(self) -> NDArray[np.float32]:
        return np.array(
            [child.get_rotation() for child in self.get_children()], dtype=np.float32
        )

    def _plt_set_text_rotation(self, rotation: NDArray[np.float32]):
        for child, rotation0 in zip(self.get_children(), rotation):
            child.set_rotation(rotation0)

    def _plt_get_text_fontfamily(self) -> list[str]:
        return [child.get_fontfamily() for child in self.get_children()]

    def _plt_set_text_fontfamily(self, fontfamily: list[str]):
        for child, fontfamily0 in zip(self.get_children(), fontfamily):
            child.set_fontfamily(fontfamily0)

    ##### HasFaces #####

    @staticmethod
    def _set_bbox_props(child: mplText, **bbox_props):
        if child.get_bbox_patch() is None:
            child.set_bbox(bbox_props)
        else:
            child.get_bbox_patch().update(bbox_props)

    def _plt_get_face_color(self):
        out = []
        for child in self.get_children():
            patch = child.get_bbox_patch()
            if patch is None:
                out.append(np.array([0, 0, 0, 0]))
            else:
                out.append(np.asarray(patch.get_facecolor()))
        return np.stack(out, axis=0)

    def _plt_set_face_color(self, color):
        for child, color0 in zip(self.get_children(), color):
            self._set_bbox_props(child, facecolor=color0)

    def _plt_get_face_pattern(self) -> FacePattern:
        out = []
        for child in self.get_children():
            patch = child.get_bbox_patch()
            if patch is None:
                out.append(FacePattern.SOLID)
            else:
                out.append(FacePattern(patch.get_hatch()))
        return out

    def _plt_set_face_pattern(self, pattern: FacePattern):
        if isinstance(pattern, FacePattern):
            if pattern is FacePattern.SOLID:
                ptn = [None] * len(self.get_children())
            else:
                ptn = [pattern.value] * len(self.get_children())
        else:
            ptn = [p.value if p is not FacePattern.SOLID else None for p in pattern]
        for child, ptn0 in zip(self.get_children(), ptn):
            self._set_bbox_props(child, hatch=ptn0)

    def _plt_get_edge_color(self):
        out = []
        for child in self.get_children():
            patch = child.get_bbox_patch()
            if patch is None:
                out.append(np.array([0, 0, 0, 0]))
            else:
                out.append(np.asarray(patch.get_edgecolor()))
        return np.stack(out, axis=0)

    def _plt_set_edge_color(self, color):
        for child, color0 in zip(self.get_children(), color):
            self._set_bbox_props(child, edgecolor=color0)

    def _plt_get_edge_width(self) -> float:
        out = []
        for child in self.get_children():
            patch = child.get_bbox_patch()
            if patch is None:
                out.append(0)
            else:
                out.append(patch.get_linewidth())
        return np.array(out, dtype=np.float32)

    def _plt_set_edge_width(self, width: float):
        for child, width0 in zip(self.get_children(), width):
            self._set_bbox_props(child, linewidth=width0)

    def _plt_get_edge_style(self) -> LineStyle:
        out = []
        for child in self.get_children():
            patch = child.get_bbox_patch()
            if patch is None:
                out.append(LineStyle.SOLID)
            else:
                out.append(LineStyle(patch.get_linestyle()))
        return out

    def _plt_set_edge_style(self, style: LineStyle):
        if isinstance(style, LineStyle):
            if style is LineStyle.SOLID:
                sty = [None] * len(self.get_children())
            else:
                sty = [style.value] * len(self.get_children())
        else:
            sty = [s.value if s is not LineStyle.SOLID else None for s in style]
        for child, sty0 in zip(self.get_children(), sty):
            self._set_bbox_props(child, linestyle=sty0)


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

_ALIGNMENTS: dict[tuple[str, str], Alignment] = {}
_ALIGNMENTS_INV: dict[Alignment, tuple[str, str]] = {}
