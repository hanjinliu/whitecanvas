from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

from plotly.graph_objs import FigureWidget

from whitecanvas.types import LineStyle, Symbol


class PlotlyLayer:
    _props: dict[str, Any]
    _fig_ref: Callable[[], FigureWidget]

    def _plt_get_visible(self) -> bool:
        return self._props["visible"]

    def _plt_set_visible(self, visible: bool) -> bool:
        self._props["visible"] = visible


class PlotlyHoverableLayer(PlotlyLayer):
    def __init__(self):
        self._hover_texts: list[str] | None = None
        self._click_callbacks = []
        self._fig_ref = lambda: None

    def _plt_connect_pick_event(self, callback):
        fig = self._fig_ref()
        if fig is None:
            self._click_callbacks.append(callback)
            return
        else:
            raise NotImplementedError("post connection not implemented yet")

    def _plt_set_hover_text(self, text: list[str]):
        self._hover_texts = text
        fig = self._fig_ref()
        if fig is not None:
            self._update_hover_texts(fig)

    def _update_hover_texts(self, fig: FigureWidget):
        if self._hover_texts is None:
            return
        if len(self._hover_texts) != self._plt_get_ndata():
            warnings.warn(
                f"Length of hover text ({len(self._hover_texts)}) does not match the "
                f"number of data points ({self._plt_get_ndata()}). Ignore updating.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        fig.update_traces(
            customdata=self._hover_texts,
            selector={"uid": self._props["uid"]},
        )


@dataclass
class Location:
    row: int
    col: int
    secondary_y: bool = False

    def asdict(self):
        out = {}
        if self.row > 1 or self.col > 1:
            out["row"] = self.row
            out["col"] = self.col
        if self.secondary_y:
            out["secondary_y"] = True
        return out


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
