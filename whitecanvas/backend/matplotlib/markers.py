from __future__ import annotations
import weakref
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib.markers as mmarkers
import matplotlib.transforms as mtransforms

from psygnal import throttled
from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.protocols import MarkersProtocol, check_protocol
from whitecanvas.types import Symbol, FacePattern, LineStyle
from whitecanvas.utils.normalize import as_color_array

if TYPE_CHECKING:
    from whitecanvas.backend.matplotlib.canvas import Canvas
    from matplotlib.backend_bases import MouseEvent as mplMouseEvent


def _get_path(symbol: Symbol):
    marker_obj = mmarkers.MarkerStyle(symbol.value)
    return marker_obj.get_path().transformed(marker_obj.get_transform())


@check_protocol(MarkersProtocol)
class Markers(PathCollection, MplLayer):
    def __init__(self, xdata, ydata):
        offsets = np.stack([xdata, ydata], axis=1)
        self._symbol = Symbol.CIRCLE
        super().__init__(
            (_get_path(self._symbol),),
            sizes=[6] * len(offsets),
            offsets=offsets,
            offset_transform=plt.gca().transData,
        )
        self.set_transform(mtransforms.IdentityTransform())
        self._edge_styles = [LineStyle.SOLID] * len(offsets)
        self._pick_callbacks = []
        self._hover_texts: list[str] | None = None
        self._canvas = lambda: None

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        offsets = self.get_offsets()
        return offsets[:, 0], offsets[:, 1]

    def _plt_set_data(self, xdata, ydata):
        data = np.stack([xdata, ydata], axis=1)
        self.set_offsets(data)

    ##### HasSymbol protocol #####

    def _plt_get_symbol(self) -> Symbol:
        return self._symbol

    def _plt_set_symbol(self, symbol: Symbol):
        path = _get_path(symbol)
        self.set_paths([path] * len(self.get_offsets()))
        self._symbol = symbol

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return np.sqrt(self.get_sizes())

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if isinstance(size, (int, float, np.number)):
            size = np.full(len(self.get_offsets()), size)
        self.set_sizes(size**2)

    ##### HasFaces protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self.get_facecolor()

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.get_offsets()))
        self.set_facecolor(color)

    def _plt_get_face_pattern(self) -> FacePattern:
        return [FacePattern(self.get_hatch() or "")] * len(self.get_offsets())

    def _plt_set_face_pattern(self, pattern: FacePattern | list[FacePattern]):
        if not isinstance(pattern, FacePattern):
            pattern = pattern[0]
        if pattern is FacePattern.SOLID:
            ptn = None
        else:
            ptn = pattern.value
        self.set_hatch(ptn)

    ##### HasEdges protocol #####

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_edgecolor()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.get_offsets()))
        self.set_edgecolor(color)

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return self._edge_styles

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            styles = [style.value] * len(self.get_offsets())
        else:
            styles = [s.value for s in style]
        self.set_linestyle(styles)
        self._edge_styles = styles

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self.get_linewidth()

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if isinstance(width, (int, float, np.number)):
            width = np.full(len(self.get_offsets()), width)
        self.set_linewidth(width)

    def _plt_connect_pick_event(self, callback):
        def cb(event):
            if event.artist is not self:
                return
            callback(event.ind)

        self._pick_callbacks.append(cb)

    def _plt_set_hover_text(self, texts: list[str]):
        self._hover_texts = texts

    def post_add(self, canvas: Canvas):
        fig = self.get_figure()
        for cb in self._pick_callbacks:
            fig.canvas.mpl_connect("pick_event", cb)
        self._canvas = weakref.ref(canvas)
        fig.canvas.mpl_connect(
            "motion_notify_event", throttled(self._on_hover, timeout=500)
        )

    def _on_hover(self, event: mplMouseEvent = None):
        if self._hover_texts is None or not self._plt_get_visible():
            return
        canvas = self._canvas()
        if canvas is None:
            return
        if event.inaxes is not canvas._axes:
            return
        contains, ind = self.contains(event)
        if not contains:
            canvas._hide_tooltip()
            return
        indices = ind["ind"]
        if len(indices) == 0:
            canvas._hide_tooltip()
            return
        index: int = indices[0]
        hover_text = self._hover_texts[index]
        xy = self.get_offsets()[index]
        canvas._set_tooltip(xy, hover_text)
