from typing import Any
from whitecanvas.types import LineStyle, FacePattern, Symbol


class PlotlyLayer:
    _props: dict[str, Any]

    def _plt_get_visible(self) -> bool:
        return self._props["visible"]

    def _plt_set_visible(self, visible: bool) -> bool:
        self._props["visible"] = visible


_LINE_STYLES = {
    "solid": LineStyle.SOLID,
    "dash": LineStyle.DASH,
    "dashdot": LineStyle.DASH_DOT,
    "dot": LineStyle.DOT,
}


def from_plotly_linestyle(ls: str) -> LineStyle:
    return _LINE_STYLES.get(ls, LineStyle.SOLID)


def to_plotly_linestyle(ls: LineStyle) -> str:
    return ls.name.lower().replace("_", "")


def from_plotly_marker_symbol(s: str) -> Symbol:
    return _SYMBOLS.get(s, Symbol.CIRCLE)


def to_plotly_marker_symbol(s: Symbol) -> str:
    return _SYMBOLS_INV.get(s, "circle")


_SYMBOLS = {
    "circle": Symbol.CIRCLE,
    "square": Symbol.SQUARE,
    "diamond": Symbol.DIAMOND,
    "cross": Symbol.PLUS,
    "x": Symbol.CROSS,
    "triangle-up": Symbol.TRIANGLE_UP,
    "triangle-down": Symbol.TRIANGLE_DOWN,
    "triangle-left": Symbol.TRIANGLE_LEFT,
    "triangle-right": Symbol.TRIANGLE_RIGHT,
    "star": Symbol.STAR,
    "line-ew": Symbol.HBAR,
    "line-ns": Symbol.VBAR,
}

_SYMBOLS_INV = {v: k for k, v in _SYMBOLS.items()}
