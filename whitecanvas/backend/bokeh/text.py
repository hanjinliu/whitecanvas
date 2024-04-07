from __future__ import annotations

import bokeh.models as bk_models
import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend.bokeh._base import (
    BokehLayer,
    from_bokeh_hatch,
    from_bokeh_line_style,
    to_bokeh_hatch,
    to_bokeh_line_style,
)
from whitecanvas.protocols import TextProtocol, check_protocol
from whitecanvas.types import Alignment, Hatch, LineStyle
from whitecanvas.utils.normalize import arr_color, as_color_array, hex_color
from whitecanvas.utils.type_check import is_real_number

# column names
TEXT = "text"
TEXT_SIZE = "text_font_size"
TEXT_COLOR = "text_color"
TEXT_ANGLE = "angle"
BG_COLOR = "background_fill_color"
BG_HATCH = "background_hatch_pattern"
BD_COLOR = "border_line_color"
BD_WIDTH = "border_line_width"
BD_STYLE = "border_line_dash"

INVISIBLE = "#00000000"


@check_protocol(TextProtocol)
class Texts(BokehLayer[bk_models.Text]):
    def __init__(
        self, x: NDArray[np.floating], y: NDArray[np.floating], text: list[str]
    ):
        ntexts = len(text)
        self._data = bk_models.ColumnDataSource(
            data={
                "x": x,
                "y": y,
                TEXT: text,
                TEXT_SIZE: ["12pt"] * ntexts,
                TEXT_COLOR: ["black"] * ntexts,
                TEXT_ANGLE: [0] * ntexts,
                BG_COLOR: [INVISIBLE] * ntexts,
                BG_HATCH: [""] * ntexts,
                BD_COLOR: [INVISIBLE] * ntexts,
                BD_WIDTH: [0] * ntexts,
                BD_STYLE: ["solid"] * ntexts,
            }
        )
        self._model = bk_models.Text(
            x="x",
            y="y",
            text=TEXT,
            text_font="helvetica",
            text_font_size=TEXT_SIZE,
            text_color=TEXT_COLOR,
            text_align="left",
            angle=TEXT_ANGLE,
            background_fill_color=BG_COLOR,
            background_hatch_pattern=BG_HATCH,
            border_line_color=BD_COLOR,
            border_line_width=BD_WIDTH,
            border_line_dash=BD_STYLE,
        )
        self._visible = True

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible
        if visible:
            self._model.text_color = INVISIBLE
            self._model.background_fill_color = INVISIBLE
            self._model.border_line_color = INVISIBLE
        else:
            self._model.text_color = TEXT_COLOR
            self._model.background_fill_color = BG_COLOR
            self._model.border_line_color = BD_COLOR

    ##### TextProtocol #####

    def _plt_get_text(self) -> list[str]:
        return self._data.data[TEXT]

    def _plt_set_text(self, text: list[str]):
        self._data.data[TEXT] = text

    def _plt_get_text_color(self):
        return np.stack([arr_color(c) for c in self._data.data[TEXT_COLOR]])

    def _plt_set_text_color(self, color):
        color = as_color_array(color, len(self._data.data[TEXT]))
        self._data.data[TEXT_COLOR] = [hex_color(c) for c in color]

    def _plt_get_text_size(self) -> float:
        return np.array(
            [float(s.rstrip("pt")) for s in self._data.data[TEXT_SIZE]],
            dtype=np.float32,
        )

    def _plt_set_text_size(self, size: float):
        if is_real_number(size):
            size = np.full(len(self._data.data[TEXT]), size)
        self._data.data[TEXT_SIZE] = [f"{round(s, 1)}pt" for s in size]

    def _plt_get_text_position(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return self._data.data["x"], self._data.data["y"]

    def _plt_set_text_position(
        self, position: tuple[NDArray[np.floating], NDArray[np.floating]]
    ):
        x, y = position
        cur_data = self._data.data.copy()
        cur_data["x"], cur_data["y"] = x, y
        cur_size = len(cur_data[TEXT])
        if x.size > cur_size:
            _n = x.size - cur_size
            _concat = np.concatenate
            cur_data[TEXT] = _concat([cur_data[TEXT], [""] * _n])
            cur_data[TEXT_SIZE] = _concat([cur_data[TEXT_SIZE], ["12pt"] * _n])
            cur_data[TEXT_COLOR] = _concat([cur_data[TEXT_COLOR], ["black"] * _n])
            cur_data[TEXT_ANGLE] = _concat([cur_data[TEXT_ANGLE], [0] * _n])
            cur_data[BG_COLOR] = _concat([cur_data[BG_COLOR], [INVISIBLE] * _n])
            cur_data[BG_HATCH] = _concat([cur_data[BG_HATCH], [""] * _n])
            cur_data[BD_COLOR] = _concat([cur_data[BD_COLOR], [INVISIBLE] * _n])
            cur_data[BD_WIDTH] = _concat([cur_data[BD_WIDTH], [0] * _n])
            cur_data[BD_STYLE] = _concat([cur_data[BD_STYLE], ["solid"] * _n])
        elif x.size < cur_size:
            cur_data[TEXT] = cur_data[TEXT][: x.size]
            cur_data[TEXT_SIZE] = cur_data[TEXT_SIZE][: x.size]
            cur_data[TEXT_COLOR] = cur_data[TEXT_COLOR][: x.size]
            cur_data[TEXT_ANGLE] = cur_data[TEXT_ANGLE][: x.size]
            cur_data[BG_COLOR] = cur_data[BG_COLOR][: x.size]
            cur_data[BG_HATCH] = cur_data[BG_HATCH][: x.size]
            cur_data[BD_COLOR] = cur_data[BD_COLOR][: x.size]
            cur_data[BD_WIDTH] = cur_data[BD_WIDTH][: x.size]
            cur_data[BD_STYLE] = cur_data[BD_STYLE][: x.size]
        self._data.data = cur_data

    def _plt_get_text_anchor(self) -> Alignment:
        return Alignment(self._model.text_align)

    def _plt_set_text_anchor(self, anc: Alignment):
        self._model.text_align = anc.value

    def _plt_get_text_rotation(self) -> NDArray[np.floating]:
        return np.array(self._data.data[TEXT_ANGLE], dtype=np.float32)

    def _plt_set_text_rotation(self, rotation: float | NDArray[np.floating]):
        if is_real_number(rotation):
            rotation = np.full(len(self._data.data[TEXT]), rotation)
        self._data.data[TEXT_ANGLE] = rotation

    def _plt_get_text_fontfamily(self) -> str:
        return self._model.text_font

    def _plt_set_text_fontfamily(self, fontfamily: str):
        self._model.text_font = fontfamily

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.stack([arr_color(c) for c in self._data.data[BG_COLOR]])

    def _plt_set_face_color(self, color):
        color = as_color_array(color, len(self._data.data[TEXT]))
        self._data.data[BG_COLOR] = [hex_color(c) for c in color]

    def _plt_get_face_hatch(self) -> list[Hatch]:
        return [from_bokeh_hatch(h) for h in self._data.data[BG_HATCH]]

    def _plt_set_face_hatch(self, pattern: Hatch | list[Hatch]):
        if isinstance(pattern, Hatch):
            pattern = [pattern] * len(self._data.data[TEXT])
        self._data.data[BG_HATCH] = [to_bokeh_hatch(p) for p in pattern]

    def _plt_get_edge_color(self):
        return np.stack([arr_color(c) for c in self._data.data[BD_COLOR]])

    def _plt_set_edge_color(self, color):
        color = as_color_array(color, len(self._data.data[TEXT]))
        self._data.data[BD_COLOR] = [hex_color(c) for c in color]

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.array(self._data.data[BD_WIDTH], dtype=np.float32)

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if is_real_number(width):
            width = np.full(len(self._data.data[TEXT]), width)
        self._data.data[BD_WIDTH] = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_bokeh_line_style(s) for s in self._data.data[BD_STYLE]]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * len(self._data.data[TEXT])
        self._data.data[BD_STYLE] = [to_bokeh_line_style(s) for s in style]
