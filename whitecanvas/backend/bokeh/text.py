from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from cmap import Color

from whitecanvas.types import Alignment, FacePattern, LineStyle
from whitecanvas.protocols import TextProtocol, check_protocol

import bokeh.models as bk_models
from ._base import (
    BokehLayer,
    to_bokeh_line_style,
    from_bokeh_line_style,
    to_bokeh_hatch,
    from_bokeh_hatch,
)


@check_protocol(TextProtocol)
class Text(BokehLayer[bk_models.Text]):
    def __init__(self, x: float, y: float, text: str):
        self._data = bk_models.ColumnDataSource(data=dict(x=[x], y=[y], text=[text]))
        self._model = bk_models.Text(x="x", y="y", text="text")

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self._model.visible

    def _plt_set_visible(self, visible: bool):
        self._model.visible = visible

    ##### TextProtocol #####

    def _plt_get_text(self) -> str:
        return self._model.text[0]

    def _plt_set_text(self, text: str):
        self._model.text = [text]

    def _plt_get_text_color(self):
        return np.array(self._model.text_color)

    def _plt_set_text_color(self, color):
        self._model.text_color = Color(color).hex

    def _plt_get_text_size(self) -> float:
        return int(self._model.text_font_size.rstrip("pt"))

    def _plt_set_text_size(self, size: float):
        self._model.text_font_size = f"{size}pt"

    def _plt_get_text_position(self) -> tuple[float, float]:
        return self._model.x[0], self._model.y[0]

    def _plt_set_text_position(self, position: tuple[float, float]):
        x, y = position
        self._model.x = [x]
        self._model.y = [y]

    def _plt_get_text_anchor(self) -> Alignment:
        anc = self._model.anchor
        return Alignment(anc)

    def _plt_set_text_anchor(self, anc: Alignment):
        self._model.anchor = anc.value

    def _plt_get_text_rotation(self) -> float:
        return self._model.angle

    def _plt_set_text_rotation(self, rotation: float):
        self._model.angle = rotation

    def _plt_get_text_fontfamily(self) -> str:
        return self._model.text_font

    def _plt_set_text_fontfamily(self, fontfamily: str):
        self._model.text_font = fontfamily

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.array(self._model.background_fill_color)

    def _plt_set_face_color(self, color):
        self._model.background_fill_color = Color(color).hex

    def _plt_get_face_pattern(self) -> FacePattern:
        return from_bokeh_hatch(self._model.background_hatch_pattern)

    def _plt_set_face_pattern(self, pattern: FacePattern):
        self._model.background_hatch_pattern = to_bokeh_hatch(pattern)

    def _plt_get_edge_color(self):
        return np.array(self._model.border_line_color)

    def _plt_set_edge_color(self, color):
        self._model.border_line_color = Color(color).hex

    def _plt_get_edge_width(self) -> float:
        return self._model.border_line_width

    def _plt_set_edge_width(self, width: float):
        self._model.border_line_width = width

    def _plt_get_edge_style(self) -> LineStyle:
        return from_bokeh_line_style(self._model.border_line_dash)

    def _plt_set_edge_style(self, style: LineStyle):
        self._model.border_line_dash = to_bokeh_line_style(style)
