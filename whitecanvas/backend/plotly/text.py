from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import Alignment, Hatch, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.utils.normalize import arr_color, as_color_array, rgba_str_color
from ._base import PlotlyLayer


@check_protocol(TextProtocol)
class Texts(PlotlyLayer):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        ntexts = len(text)
        self._props = {
            "x": x,
            "y": y,
            "mode": "text",
            "text": text,
            "textposition": ["bottom left"] * ntexts,
            "textfont": {
                "family": ["Arial"] * ntexts,
                "size": np.full(ntexts, 10),
                "color": ["rgba(0, 0, 0, 255)"] * ntexts,
            },
            "type": "scatter",
            "showlegend": False,
            # "angle": np.zeros(ntexts),
            "visible": True,
        }
        self._angle = np.zeros(ntexts)

    ##### TextProtocol #####

    def _plt_get_text(self) -> str:
        return self._props["text"]

    def _plt_set_text(self, text: str):
        self._props["text"] = text

    def _plt_get_text_color(self):
        return np.stack(
            [arr_color(c) for c in self._props["textfont"]["color"]],
            axis=0,
        )

    def _plt_set_text_color(self, color):
        color = as_color_array(color, len(self._props["text"]))
        self._props["textfont"]["color"] = [rgba_str_color(c) for c in color]

    def _plt_get_text_size(self) -> NDArray[np.floating]:
        return self._props["textfont"]["size"]

    def _plt_set_text_size(self, size: NDArray[np.floating]):
        if np.isscalar(size):
            size = np.full(len(self._props["text"]), size)
        self._props["textfont"]["size"] = size

    def _plt_get_text_position(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return self._props["x"], self._props["y"]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        self._props["x"], self._props["y"] = position

    def _plt_get_text_anchor(self) -> list[Alignment]:
        return [_from_plotly_alignment(p) for p in self._props["textposition"]]

    def _plt_set_text_anchor(self, anc: Alignment | list[Alignment]):
        if isinstance(anc, Alignment):
            self._props["textposition"] = [_to_plotly_alignment(anc)] * len(
                self._props["textposition"]
            )
        else:
            self._props["textposition"] = [_to_plotly_alignment(a) for a in anc]

    def _plt_get_text_rotation(self):
        return self._angle

    def _plt_set_text_rotation(self, rotation: float):
        if np.isscalar(rotation):
            rotation = np.full(len(self._props["text"]), rotation)
        self._angle = rotation

    def _plt_get_text_fontfamily(self) -> list[str]:
        return self._props["textfont"]["family"]

    def _plt_set_text_fontfamily(self, fontfamily: list[str]):
        if isinstance(fontfamily, str):
            fontfamily = [fontfamily] * len(self._props["textfont"]["family"])
        self._props["textfont"]["family"] = fontfamily

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.zeros((len(self._props["text"]), 4))

    def _plt_set_face_color(self, color):
        pass

    def _plt_get_face_pattern(self) -> Hatch:
        return [Hatch.SOLID] * len(self._props["text"])

    def _plt_set_face_pattern(self, pattern: Hatch):
        pass

    ##### HasEdges #####

    def _plt_get_edge_color(self):
        return np.zeros((len(self._props["text"]), 4))

    def _plt_set_edge_color(self, color):
        pass

    def _plt_get_edge_width(self) -> float:
        return np.full(len(self._props["text"]), 0.0)

    def _plt_set_edge_width(self, width: float):
        pass

    def _plt_get_edge_style(self) -> LineStyle:
        return [LineStyle.SOLID] * len(self._props["text"])

    def _plt_set_edge_style(self, style: LineStyle):
        pass


def _from_plotly_alignment(s: str) -> Alignment:
    return _ALIGNMENT_MAP.get(s.lower(), Alignment.CENTER)


def _to_plotly_alignment(a: Alignment) -> str:
    return _ALIGNMENT_MAP_INV.get(a, "center")


_ALIGNMENT_MAP = {
    "top": Alignment.BOTTOM,
    "bottom": Alignment.TOP,
    "left": Alignment.RIGHT,
    "right": Alignment.LEFT,
    "center": Alignment.CENTER,
    "top left": Alignment.BOTTOM_RIGHT,
    "top right": Alignment.BOTTOM_LEFT,
    "bottom left": Alignment.TOP_RIGHT,
    "bottom right": Alignment.TOP_LEFT,
}

_ALIGNMENT_MAP_INV = {v: k for k, v in _ALIGNMENT_MAP.items()}
