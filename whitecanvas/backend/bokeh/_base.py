import bokeh.models as bk_models
from typing import Generic, TypeVar
from whitecanvas.protocols import BaseProtocol
from whitecanvas.types import LineStyle, Symbol, FacePattern

_M = TypeVar("_M", bound=bk_models.Model)


class BokehLayer(BaseProtocol, Generic[_M]):
    _model: _M
    _data: bk_models.ColumnDataSource


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
