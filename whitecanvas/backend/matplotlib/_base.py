from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.markers as mmarkers

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent as mplMouseEvent

    from whitecanvas.backend.matplotlib.canvas import Canvas
    from whitecanvas.types import Symbol


class MplLayer:
    def _plt_get_visible(self):
        return self.get_visible()

    def _plt_set_visible(self, visible):
        self.set_visible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)


class FakeAxes:
    def __init__(self):
        self.transData = None
        self.transAxes = None


def symbol_to_path(symbol: Symbol):
    marker_obj = mmarkers.MarkerStyle(symbol.value)
    return marker_obj.get_path().transformed(marker_obj.get_transform())


OVERLAY_ZORDER = 10000


def as_overlay(layer: MplLayer, canvas):
    layer.set_transform(canvas._axes.transAxes)
    layer.set_zorder(OVERLAY_ZORDER)


class MplMouseEventsMixin(MplLayer):
    def __init__(self):
        self._pick_callbacks = []
        self._hover_texts: list[str] | None = None

    def _on_hover(self, event: mplMouseEvent = None):
        if self._hover_texts is None or not self._plt_get_visible():
            return
        contains, ind = self.contains(event)
        if not contains:
            return
        indices = ind["ind"]
        if len(indices) == 0:
            return
        index: int = indices[0]
        hover_text = self._hover_texts[index]
        return hover_text

    def _plt_set_hover_text(self, texts: list[str]):
        self._hover_texts = texts

    def post_add(self, canvas: Canvas):
        fig = canvas._axes.figure
        if fig is None:
            return
        for cb in self._pick_callbacks:
            fig.canvas.mpl_connect("pick_event", cb)

    def _plt_connect_pick_event(self, callback):
        def cb(event):
            if event.artist is not self:
                return
            callback(event.ind)

        self._pick_callbacks.append(cb)
