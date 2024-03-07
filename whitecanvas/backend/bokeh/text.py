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
                "text": text,
                "text_font": ["Arial"] * ntexts,
                "text_font_size": ["12pt"] * ntexts,
                "text_align": ["left"] * ntexts,
                "text_color": ["black"] * ntexts,
                "angle": [0] * ntexts,
                "background_fill_color": ["#00000000"] * ntexts,
                "background_hatch_pattern": [""] * ntexts,
                "border_line_color": ["#00000000"] * ntexts,
                "border_line_width": [0] * ntexts,
                "border_line_dash": ["solid"] * ntexts,
            }
        )
        self._model = bk_models.Text(
            x="x",
            y="y",
            text="text",
            text_font="text_font",
            text_font_size="text_font_size",
            text_color="text_color",
            text_align="text_align",
            angle="angle",
            background_fill_color="background_fill_color",
            background_hatch_pattern="background_hatch_pattern",
            border_line_color="border_line_color",
            border_line_width="border_line_width",
            border_line_dash="border_line_dash",
        )
        self._visible = True

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible
        if visible:
            self._model.text_color = "#00000000"
            self._model.background_fill_color = "#00000000"
            self._model.border_line_color = "#00000000"
        else:
            self._model.text_color = "text_color"
            self._model.background_fill_color = "background_fill_color"
            self._model.border_line_color = "border_line_color"

    ##### TextProtocol #####

    def _plt_get_text(self) -> list[str]:
        return self._data.data["text"]

    def _plt_set_text(self, text: list[str]):
        self._data.data["text"] = text

    def _plt_get_text_color(self):
        return np.stack([arr_color(c) for c in self._data.data["text_color"]])

    def _plt_set_text_color(self, color):
        color = as_color_array(color, len(self._data.data["text"]))
        self._data.data["text_color"] = [hex_color(c) for c in color]

    def _plt_get_text_size(self) -> float:
        return np.array(
            [float(s.rstrip("pt")) for s in self._data.data["text_font_size"]],
            dtype=np.float32,
        )

    def _plt_set_text_size(self, size: float):
        if is_real_number(size):
            size = np.full(len(self._data.data["text"]), size)
        self._data.data["text_font_size"] = [f"{round(s, 1)}pt" for s in size]

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
        cur_size = len(cur_data["text"])
        if x.size > cur_size:
            _n = x.size - cur_size
            cur_data["text"] = np.concatenate([cur_data["text"], [""] * _n])
            cur_data["text_font"] = np.concatenate(
                [cur_data["text_font"], ["Arial"] * _n]
            )
            cur_data["text_font_size"] = np.concatenate(
                [cur_data["text_font_size"], ["12pt"] * _n]
            )
            cur_data["text_align"] = np.concatenate(
                [cur_data["text_align"], ["left"] * _n]
            )
            cur_data["text_color"] = np.concatenate(
                [cur_data["text_color"], ["black"] * _n]
            )
            cur_data["angle"] = np.concatenate([cur_data["angle"], [0] * _n])
            cur_data["background_fill_color"] = np.concatenate(
                [cur_data["background_fill_color"], ["#00000000"] * _n]
            )
            cur_data["background_hatch_pattern"] = np.concatenate(
                [cur_data["background_hatch_pattern"], [""] * _n]
            )
            cur_data["border_line_color"] = np.concatenate(
                [cur_data["border_line_color"], ["#00000000"] * _n]
            )
            cur_data["border_line_width"] = np.concatenate(
                [cur_data["border_line_width"], [0] * _n]
            )
            cur_data["border_line_dash"] = np.concatenate(
                [cur_data["border_line_dash"], ["solid"] * _n]
            )
        elif x.size < cur_size:
            cur_data["text"] = cur_data["text"][: x.size]
            cur_data["text_font"] = cur_data["text_font"][: x.size]
            cur_data["text_font_size"] = cur_data["text_font_size"][: x.size]
            cur_data["text_align"] = cur_data["text_align"][: x.size]
            cur_data["text_color"] = cur_data["text_color"][: x.size]
            cur_data["angle"] = cur_data["angle"][: x.size]
            cur_data["background_fill_color"] = cur_data["background_fill_color"][
                : x.size
            ]
            cur_data["background_hatch_pattern"] = cur_data["background_hatch_pattern"][
                : x.size
            ]
            cur_data["border_line_color"] = cur_data["border_line_color"][: x.size]
            cur_data["border_line_width"] = cur_data["border_line_width"][: x.size]
            cur_data["border_line_dash"] = cur_data["border_line_dash"][: x.size]
        self._data.data = cur_data

    def _plt_get_text_anchor(self) -> list[Alignment]:
        return [Alignment(anc) for anc in self._data.data["text_align"]]

    def _plt_set_text_anchor(self, anc: Alignment | list[Alignment]):
        if isinstance(anc, Alignment):
            anc = [anc] * len(self._data.data["text"])
        self._data.data["text_align"] = [a.value for a in anc]

    def _plt_get_text_rotation(self) -> NDArray[np.floating]:
        return np.array(self._data.data["angle"], dtype=np.float32)

    def _plt_set_text_rotation(self, rotation: float | NDArray[np.floating]):
        if is_real_number(rotation):
            rotation = np.full(len(self._data.data["text"]), rotation)
        self._data.data["angle"] = rotation

    def _plt_get_text_fontfamily(self) -> list[str]:
        return self._data.data["text_font"]

    def _plt_set_text_fontfamily(self, fontfamily: str | list[str]):
        if isinstance(fontfamily, str):
            fontfamily = [fontfamily] * len(self._data.data["text"])
        self._data.data["text_font"] = fontfamily

    ##### HasFaces #####

    def _plt_get_face_color(self):
        return np.stack(
            [arr_color(c) for c in self._data.data["background_fill_color"]]
        )

    def _plt_set_face_color(self, color):
        color = as_color_array(color, len(self._data.data["text"]))
        self._data.data["background_fill_color"] = [hex_color(c) for c in color]

    def _plt_get_face_hatch(self) -> list[Hatch]:
        return [
            from_bokeh_hatch(h) for h in self._data.data["background_hatch_pattern"]
        ]

    def _plt_set_face_hatch(self, pattern: Hatch | list[Hatch]):
        if isinstance(pattern, Hatch):
            pattern = [pattern] * len(self._data.data["text"])
        self._data.data["background_hatch_pattern"] = [
            to_bokeh_hatch(p) for p in pattern
        ]

    def _plt_get_edge_color(self):
        return np.stack([arr_color(c) for c in self._data.data["border_line_color"]])

    def _plt_set_edge_color(self, color):
        color = as_color_array(color, len(self._data.data["text"]))
        self._data.data["border_line_color"] = [hex_color(c) for c in color]

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.array(self._data.data["border_line_width"], dtype=np.float32)

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if is_real_number(width):
            width = np.full(len(self._data.data["text"]), width)
        self._data.data["border_line_width"] = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_bokeh_line_style(s) for s in self._data.data["border_line_dash"]]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * len(self._data.data["text"])
        self._data.data["border_line_dash"] = [to_bokeh_line_style(s) for s in style]
