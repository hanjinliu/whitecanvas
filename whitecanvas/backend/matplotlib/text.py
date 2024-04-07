from __future__ import annotations

import numpy as np
from matplotlib.artist import Artist
from matplotlib.text import Text as mplText
from numpy.typing import NDArray

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.types import Alignment, Hatch, LineStyle
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number


@check_protocol(TextProtocol)
class Texts(Artist, MplLayer):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        super().__init__()
        self._children: list[mplText] = []
        for x0, y0, text0 in zip(x, y, text):
            self._children.append(
                mplText(
                    x0, y0, text0, verticalalignment="baseline",
                    horizontalalignment="left", clip_on=True,
                    color=np.array([0, 0, 0, 1], dtype=np.float32),
                )  # fmt: skip
            )
        self._font_family = "Arial"
        self._align = Alignment.BOTTOM_LEFT
        self._remove_method = _remove_method

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
        color = as_color_array(color, len(self.get_children()))
        for child, color0 in zip(self.get_children(), color):
            child.set_color(color0)

    def _plt_get_text_size(self) -> NDArray[np.float32]:
        return np.array(
            [child.get_fontsize() for child in self.get_children()], dtype=np.float32
        )

    def _plt_set_text_size(self, size: float | NDArray[np.float32]):
        if is_real_number(size):
            _size = np.full(len(self.get_children()), size, dtype=np.float32)
        else:
            _size = size
        for child, size0 in zip(self.get_children(), _size):
            child.set_fontsize(size0)

    def _plt_get_text_position(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        ar = np.array([child.get_position() for child in self.get_children()])
        if ar.size == 0:
            return np.array([]), np.array([])
        return ar[:, 0], ar[:, 1]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.float32], NDArray[np.float32]]
    ):
        x, y = position
        if x.size > len(self._children):
            for _ in range(x.size - len(self._children)):
                self._children.append(
                    mplText(
                        0, 0, "", verticalalignment="baseline",
                        horizontalalignment="left", clip_on=True,
                        color=np.array([0, 0, 0, 1], dtype=np.float32),
                    )  # fmt: skip
                )
        elif x.size < len(self._children):
            for c in self._children[x.size :]:
                c.remove()
            self._children = self._children[: x.size]
        for child, x0, y0 in zip(self.get_children(), x, y):
            child.set_position((x0, y0))

    def _plt_get_text_anchor(self) -> Alignment:
        return self._align

    def _plt_set_text_anchor(self, anc: Alignment):
        """Set the text position."""
        v, h = anc.split()
        va = _VERTICAL_ALIGNMENTS_INV[v]
        ha = _HORIZONTAL_ALIGNMENTS_INV[h]
        for child in self.get_children():
            child.set_verticalalignment(va)
            child.set_horizontalalignment(ha)
        self._align = anc

    def _plt_get_text_rotation(self) -> NDArray[np.float32]:
        return np.array(
            [child.get_rotation() for child in self.get_children()], dtype=np.float32
        )

    def _plt_set_text_rotation(self, rotation: NDArray[np.float32]):
        for child, rotation0 in zip(self.get_children(), rotation):
            child.set_rotation(rotation0)

    def _plt_get_text_fontfamily(self) -> str:
        return self._font_family

    def _plt_set_text_fontfamily(self, fontfamily: str):
        for child in self.get_children():
            child.set_fontfamily(fontfamily)
        self._font_family = fontfamily

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
        color = as_color_array(color, len(self.get_children()))
        for child, color0 in zip(self.get_children(), color):
            self._set_bbox_props(child, facecolor=color0)

    def _plt_get_face_hatch(self) -> Hatch:
        out = []
        for child in self.get_children():
            patch = child.get_bbox_patch()
            if patch is None:
                out.append(Hatch.SOLID)
            else:
                out.append(Hatch(patch.get_hatch() or ""))
        return out

    def _plt_set_face_hatch(self, pattern: Hatch):
        if isinstance(pattern, Hatch):
            if pattern is Hatch.SOLID:
                ptn = [None] * len(self.get_children())
            else:
                ptn = [pattern.value] * len(self.get_children())
        else:
            ptn = [p.value if p is not Hatch.SOLID else None for p in pattern]
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
        color = as_color_array(color, len(self.get_children()))
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

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if is_real_number(width):
            width = np.full(len(self.get_children()), width, dtype=np.float32)
        for child, width0 in zip(self.get_children(), width):
            self._set_bbox_props(child, linewidth=width0)

    def _plt_get_edge_style(self) -> LineStyle:
        out = []
        for child in self.get_children():
            patch = child.get_bbox_patch()
            if patch is None:
                out.append(LineStyle.SOLID)
            else:
                style = patch.get_linestyle()
                if style == "solid":
                    out.append(LineStyle.SOLID)
                else:
                    out.append(LineStyle(style))
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


def _remove_method(this: Texts):
    for child in this.get_children():
        child.remove()
