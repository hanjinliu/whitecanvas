from __future__ import annotations
import numpy as np

from whitecanvas.types import Alignment, FacePattern, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.utils.normalize import arr_color, rgba_str_color
from ._base import PlotlyLayer


@check_protocol(TextProtocol)
class Text(PlotlyLayer):
    def __init__(self, x: float, y: float, text: str):
        self._props = {
            "x": [x],
            "y": [y],
            "mode": "markers",
            "marker": {
                "color": "rgba(0, 0, 0, 0)",
                "size": 10,
                "symbol": "circle",
            },
            "text": [text],
            "textposition": "bottom left",
            "textfont_color": "black",
            "textfont_size": 10,
            "textfont_family": "Arial",
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
        return self._props["text"][0]

    def _plt_set_text(self, text: str):
        self._props["text"][0] = text

    def _plt_get_text_color(self):
        return arr_color(self._props["textfont_color"])

    def _plt_set_text_color(self, color):
        self._props["textfont_color"] = rgba_str_color(color)

    def _plt_get_text_size(self) -> float:
        return self._props["textfont_size"]

    def _plt_set_text_size(self, size: float):
        self._props["textfont_size"] = size

    def _plt_get_text_position(self) -> tuple[float, float]:
        return self._props["x"][0], self._props["y"][0]

    def _plt_set_text_position(self, position: tuple[float, float]):
        self._props["x"][0], self._props["y"][0] = position

    def _plt_get_text_anchor(self) -> Alignment:
        return _norm_alignment(self._props["textposition"])

    def _plt_set_text_anchor(self, anc: Alignment):
        self._props["textposition"] = anc.value.replace("_", " ")

    def _plt_get_text_rotation(self) -> float:
        return self._rotation

    def _plt_set_text_rotation(self, rotation: float):
        self._rotation = rotation

    def _plt_get_text_fontfamily(self) -> str:
        return self._props["textfont_family"]

    def _plt_set_text_fontfamily(self, fontfamily: str):
        self._props["textfont_family"] = fontfamily

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


def _norm_alignment(s: str) -> Alignment:
    if s == "center":
        return Alignment.CENTER
    va, ha = s.lower().split(" ")
    return Alignment(f"{va}_{ha}")
