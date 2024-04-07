from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from plotly import graph_objs as go

from whitecanvas.types import LineStyle, Symbol

if TYPE_CHECKING:
    from plotly.basedatatypes import BaseTraceType

_O = TypeVar("_O", bound="BaseTraceType")


class PlotlyLayer(Generic[_O]):
    _props: dict[str, Any] | _O
    _fig_ref: Callable[[], go.Figure]

    def _plt_get_visible(self) -> bool:
        return self._props["visible"]

    def _plt_set_visible(self, visible: bool) -> bool:
        self._props["visible"] = visible


class PlotlyHoverableLayer(PlotlyLayer[_O]):
    def __init__(self):
        self._hover_texts: list[str] | None = None
        self._click_callbacks = []
        self._fig_ref = lambda: None

    def _plt_connect_pick_event(self, callback):
        fig = self._fig_ref()
        if fig is None:
            self._click_callbacks.append(callback)
        else:
            self._connect_mouse_events(fig, [callback])

    def _connect_mouse_events(
        self,
        fig: go.Figure,
        callbacks: list[Callable] | None = None,
    ):
        if callbacks is None:
            callbacks = self._click_callbacks
        gobj = self._props
        if not hasattr(gobj, "uid"):
            raise ValueError("Graph object is not created yet.")
        for cb in callbacks:
            gobj.on_click(_convert_cb(cb), append=True)
        self._fig_ref = weakref.ref(fig)
        self._update_hover_texts(fig)

    def _plt_set_hover_text(self, text: list[str]):
        self._hover_texts = text
        fig = self._fig_ref()
        if fig is not None:
            self._update_hover_texts(fig)

    def _update_hover_texts(self, fig: go.Figure):
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
class FigureLocation:
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

    def asdictn(self):
        out = {}
        if self.row > 1 or self.col > 1:
            out["rows"] = self.row
            out["cols"] = self.col
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


def _convert_cb(cb):
    def _out(trace, points, state):
        indices = points.point_inds
        if indices:
            cb(indices)

    return _out


def to_html(canvas) -> str:
    from plotly.io import to_html

    return to_html(canvas._figs)
