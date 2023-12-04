from __future__ import annotations

from typing import Generic, TypeVar
import numpy as np
from numpy.typing import NDArray
import bokeh.models as bk_models
from whitecanvas.protocols import BaseProtocol
from whitecanvas.types import LineStyle, Symbol, FacePattern
from whitecanvas.utils.normalize import arr_color, hex_color

_M = TypeVar("_M", bound=bk_models.Model)


class BokehLayer(BaseProtocol, Generic[_M]):
    _model: _M
    _data: bk_models.ColumnDataSource


class HeteroLayer(BokehLayer[_M]):
    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._model.line_color = "edge_color"
            self._model.fill_color = "face_color"
        else:
            self._model.line_color = "#00000000"
            self._model.fill_color = "#00000000"
        self._visible = visible

    ##### HasFace protocol #####

    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return np.stack([arr_color(c) for c in self._data.data["face_color"]], axis=0)

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = [hex_color(color)] * self._plt_get_ndata()
        else:
            color = [hex_color(c) for c in color]
        self._data.data["face_color"] = color

    def _plt_get_face_pattern(self) -> list[FacePattern]:
        return [from_bokeh_hatch(p) for p in self._data.data["pattern"]]

    def _plt_set_face_pattern(self, pattern: FacePattern | list[FacePattern]):
        if isinstance(pattern, FacePattern):
            ptn = [to_bokeh_hatch(pattern)] * self._plt_get_ndata()
        else:
            ptn = [to_bokeh_hatch(p) for p in pattern]
        self._data.data["pattern"] = ptn

    ##### HasEdges protocol #####

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self._data.data["width"]

    def _plt_set_edge_width(self, width: float):
        if np.isscalar(width):
            width = np.full(self._plt_get_ndata(), width)
        self._data.data["width"] = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_bokeh_line_style(d) for d in self._data.data["style"]]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * self._plt_get_ndata()
        val = [to_bokeh_line_style(s) for s in style]
        self._data.data["style"] = val

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.stack([arr_color(c) for c in self._data.data["edge_color"]], axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = [hex_color(color)] * self._plt_get_ndata()
        else:
            color = [hex_color(c) for c in color]
        self._data.data["edge_color"] = color


def to_bokeh_line_style(style: LineStyle) -> str:
    if style is LineStyle.SOLID:
        return "solid"
    elif style is LineStyle.DASH:
        return "dashed"
    elif style is LineStyle.DOT:
        return "dotted"
    elif style is LineStyle.DASH_DOT:
        return "dashdot"


def from_bokeh_line_style(style: str) -> LineStyle:
    if style == "solid":
        return LineStyle.SOLID
    elif style == "dashed":
        return LineStyle.DASH
    elif style == "dotted":
        return LineStyle.DOT
    elif style in ("dashdot", "dotdash"):
        return LineStyle.DASH_DOT
    else:
        return LineStyle.SOLID


def to_bokeh_symbol(symbol: Symbol) -> tuple[str, float]:
    return _SYMBOL_MAP[symbol]


def from_bokeh_symbol(symbol: str, angle: float) -> Symbol:
    return _SYMBOL_MAP_INV[(symbol, angle)]


def to_bokeh_hatch(pattern: FacePattern) -> str:
    return _HATCH_MAP[pattern]


def from_bokeh_hatch(pattern: str) -> FacePattern:
    return _HATCH_MAP_INV[pattern]


# the second parameter is "angle" / pi * 2
_SYMBOL_MAP = {
    Symbol.CIRCLE: ("circle", 0),
    Symbol.SQUARE: ("square", 0),
    Symbol.TRIANGLE_UP: ("triangle", 0),
    Symbol.TRIANGLE_DOWN: ("inverted_triangle", 0),
    Symbol.TRIANGLE_LEFT: ("triangle", 1),
    Symbol.TRIANGLE_RIGHT: ("triangle", -1),
    Symbol.DIAMOND: ("diamond", 0),
    Symbol.CROSS: ("x", 0),
    Symbol.PLUS: ("cross", 0),
    Symbol.STAR: ("star", 0),
    Symbol.DOT: ("dot", 0),
    Symbol.VBAR: ("dash", 1),
    Symbol.HBAR: ("dash", 0),
}

_SYMBOL_MAP_INV = {v: k for k, v in _SYMBOL_MAP.items()}

_HATCH_MAP = {
    FacePattern.SOLID: " ",
    FacePattern.HORIZONTAL: "horizontal_line",
    FacePattern.VERTICAL: "vertical_line",
    FacePattern.CROSS: "cross",
    FacePattern.DIAGONAL_BACK: "right_diagonal_line",
    FacePattern.DIAGONAL_FORWARD: "left_diagonal_line",
    FacePattern.DIAGONAL_CROSS: "diagonal_cross",
    FacePattern.DOTS: "dot",
}
_HATCH_MAP_INV = {v: k for k, v in _HATCH_MAP.items()}
