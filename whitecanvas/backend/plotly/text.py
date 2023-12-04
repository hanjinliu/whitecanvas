from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import Alignment, FacePattern, LineStyle
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
            "mode": "markers",
            "marker": {
                "color": "rgba(0, 0, 0, 0)",
                "size": 10,
                "symbol": "circle",
            },
            "text": text,
            "textposition": ["bottom left"] * ntexts,
            "textfont_color": ["black"] * ntexts,
            "textfont_size": np.full(ntexts, 10),
            "textfont_family": ["Arial"] * ntexts,
            "type": "scatter",
            "showlegend": False,
            "visible": True,
        }
        # TODO: plotly does not support text rotation.
        # It seems that rotation (and background) can be implemented with
        # fig.add_annotation, but since it is a layout instead of a trace,
        # we need additional work to make it work.
        self._rotation = 0

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### TextProtocol #####

    def _plt_get_text(self) -> str:
        return self._props["text"]

    def _plt_set_text(self, text: str):
        self._props["text"] = text

    def _plt_get_text_color(self):
        return arr_color(self._props["textfont_color"])

    def _plt_set_text_color(self, color):
        color = as_color_array(color, len(self._props["text"]))
        self._props["textfont_color"] = [rgba_str_color(c) for c in color]

    def _plt_get_text_size(self) -> NDArray[np.floating]:
        return self._props["textfont_size"]

    def _plt_set_text_size(self, size: NDArray[np.floating]):
        if np.isscalar(size):
            size = np.full(len(self._props["text"]), size)
        self._props["textfont_size"] = size

    def _plt_get_text_position(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return self._props["x"], self._props["y"]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        self._props["x"], self._props["y"] = position

    def _plt_get_text_anchor(self) -> list[Alignment]:
        return [_norm_alignment(p) for p in self._props["textposition"]]

    def _plt_set_text_anchor(self, anc: Alignment | list[Alignment]):
        if isinstance(anc, Alignment):
            self._props["textposition"] = [anc.value.replace("_", " ")] * len(
                self._props["textposition"]
            )
        else:
            self._props["textposition"] = [a.value.replace("_", " ") for a in anc]

    def _plt_get_text_rotation(self) -> float:
        return self._rotation

    def _plt_set_text_rotation(self, rotation: float):
        if np.isscalar(rotation):
            rotation = np.full(len(self._props["text"]), rotation)
        self._rotation = rotation

    def _plt_get_text_fontfamily(self) -> list[str]:
        return self._props["textfont_family"]

    def _plt_set_text_fontfamily(self, fontfamily: list[str]):
        if isinstance(fontfamily, str):
            fontfamily = [fontfamily] * len(self._props["textfont_family"])
        self._props["textfont_family"] = fontfamily

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.zeros((len(self._props["text"]), 4))

    def _plt_set_face_color(self, color):
        pass

    def _plt_get_face_pattern(self) -> FacePattern:
        return [FacePattern.SOLID] * len(self._props["text"])

    def _plt_set_face_pattern(self, pattern: FacePattern):
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


def _norm_alignment(s: str) -> Alignment:
    if s == "center":
        return Alignment.CENTER
    va, ha = s.lower().split(" ")
    return Alignment(f"{va}_{ha}")
