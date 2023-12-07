from __future__ import annotations

from typing import Callable
from matplotlib.artist import Artist

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backend_bases import (
    MouseEvent as mplMouseEvent,
    MouseButton as mplMouseButton,
)
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from .bars import Bars
from .text import Texts as whitecanvasText
from .image import Image as whitecanvasImage
from ._labels import Title, XAxis, YAxis, XLabel, YLabel, XTicks, YTicks
from whitecanvas import protocols
from whitecanvas.types import MouseEvent, Modifier, MouseButton, MouseEventType
from whitecanvas.backend.matplotlib._base import MplLayer


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(self, ax: plt.Axes | None = None):
        if ax is None:
            ax = plt.gca()
        self._axes = ax
        self._xaxis = XAxis(self)
        self._yaxis = YAxis(self)
        self._title = Title(self)
        self._xlabel = XLabel(self)
        self._ylabel = YLabel(self)
        self._xticks = XTicks(self)
        self._yticks = YTicks(self)
        ax.set_axisbelow(True)  # grid lines below other layers
        self._annot = ax.annotate(
            text="", xy=(0, 0), xytext=(20, -20), textcoords="offset points",
            bbox=dict(fc="w"), fontproperties={"size": 14, "family": "Arial"},
        )  # fmt: skip
        self._annot.set_visible(False)

    def _set_tooltip(self, pos, text: str):
        self._annot.xy = pos
        self._annot.set_text(text)
        self._annot.set_visible(True)
        if fig := self._axes.get_figure():
            fig.canvas.draw_idle()

    def _hide_tooltip(self):
        if self._annot.get_visible():
            self._annot.set_visible(False)
            if fig := self._axes.get_figure():
                fig.canvas.draw_idle()

    def _plt_get_native(self):
        return self._axes

    def _plt_get_title(self):
        return self._title

    def _plt_get_xaxis(self):
        return self._xaxis

    def _plt_get_yaxis(self):
        return self._yaxis

    def _plt_get_xlabel(self):
        return self._xlabel

    def _plt_get_ylabel(self):
        return self._ylabel

    def _plt_get_xticks(self):
        return self._xticks

    def _plt_get_yticks(self):
        return self._yticks

    def _plt_reorder_layers(self, layers: list[MplLayer]):
        for i, layer in enumerate(layers):
            layer._plt_set_zorder(i)

    def _plt_get_aspect_ratio(self) -> float | None:
        out = self._axes.get_aspect()
        if out == "auto":
            return None
        return out

    def _plt_set_aspect_ratio(self, ratio: float | None):
        if ratio is None:
            self._axes.set_aspect("auto")
        else:
            self._axes.set_aspect(ratio)

    def _plt_add_layer(self, layer: Artist):
        if isinstance(layer, Line2D):
            self._axes.add_line(layer)
        elif isinstance(layer, Collection):
            self._axes.add_collection(layer, autolim=False)
        elif isinstance(layer, Bars):
            for child in layer.patches:
                self._axes.add_patch(child)
            self._axes.add_container(layer)
        elif isinstance(layer, whitecanvasText):
            layer.set_transform(self._axes.transData)
            for child in layer._children:
                self._axes._add_text(child)
        elif isinstance(layer, whitecanvasImage):
            self._axes.add_artist(layer)
        else:
            raise NotImplementedError(f"{layer}")
        if hasattr(layer, "post_add"):
            layer.post_add(self)

    def _plt_remove_layer(self, layer: Artist):
        """Remove layer from the canvas"""
        layer.remove()

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self._axes.get_visible()

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        self._axes.set_visible(visible)

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev: mplMouseEvent):
            if ev.inaxes is not self._axes or ev.dblclick:
                return
            callback(self._translate_mouse_event(ev, MouseEventType.CLICK))

        self._axes.figure.canvas.mpl_connect("button_press_event", _cb)

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev: mplMouseEvent):
            if ev.inaxes is not self._axes or ev.dblclick:
                return
            callback(self._translate_mouse_event(ev, MouseEventType.MOVE))

        self._axes.figure.canvas.mpl_connect("motion_notify_event", _cb)

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev: mplMouseEvent):
            if ev.inaxes is not self._axes or not ev.dblclick:
                return
            callback(self._translate_mouse_event(ev, MouseEventType.DOUBLE_CLICK))

        self._axes.figure.canvas.mpl_connect("button_press_event", _cb)

    def _plt_draw(self):
        if fig := self._axes.get_figure():
            fig.canvas.draw_idle()

    def _translate_mouse_event(
        self, ev: mplMouseEvent, typ: MouseEventType
    ) -> MouseEvent:
        if ev.key is None:
            modifiers = ()
        else:
            modifiers = []
            for k in ev.key.split("+"):
                if _MOUSE_MOD_MAP.get(k, None):
                    modifiers.append(_MOUSE_MOD_MAP.get(k, None))
            modifiers = tuple(modifiers)
        return MouseEvent(
            pos=(ev.xdata, ev.ydata),
            button=_MOUSE_BUTTON_MAP.get(ev.button, MouseButton.NONE),
            modifiers=modifiers,
            type=typ,
        )

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to x-limits changed event"""
        self._axes.callbacks.connect("xlim_changed", lambda ax: callback(ax.get_xlim()))

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to y-limits changed event"""
        self._axes.callbacks.connect("ylim_changed", lambda ax: callback(ax.get_ylim()))


_MOUSE_BUTTON_MAP = {
    mplMouseButton.LEFT: MouseButton.LEFT,
    mplMouseButton.MIDDLE: MouseButton.MIDDLE,
    mplMouseButton.RIGHT: MouseButton.RIGHT,
    mplMouseButton.BACK: MouseButton.BACK,
    mplMouseButton.FORWARD: MouseButton.FORWARD,
}
_MOUSE_MOD_MAP = {
    "control": Modifier.CTRL,
    "ctrl": Modifier.CTRL,
    "shift": Modifier.SHIFT,
    "alt": Modifier.ALT,
    "meta": Modifier.META,
}


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[int], widths: list[int], app: str = "default"):
        nr, nc = len(heights), len(widths)
        self._gridspec = plt.GridSpec(
            nr, nc, height_ratios=heights, width_ratios=widths
        )
        if app == "qt":
            app = "QtAgg"
        elif app == "wx":
            app = "WXAgg"
        elif app == "gtk":
            app = "GTK3Agg"
        elif app == "tk":
            app = "TkAgg"
        elif app == "notebook":
            app = "nbAgg"
        if app != "default":
            mpl.use(app)
        self._fig = plt.figure()

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int) -> Canvas:
        r1 = row + rowspan
        c1 = col + colspan
        axes = self._fig.add_subplot(self._gridspec[row:r1, col:c1])
        return Canvas(axes)

    def _plt_get_visible(self) -> bool:
        return self._fig.get_visible()

    def _plt_show(self):
        self._fig.show(warn=False)

    def _plt_get_background_color(self):
        self._fig.get_facecolor()

    def _plt_set_background_color(self, color):
        self._fig.set_facecolor(color)

    def _plt_screenshot(self):
        import io

        fig = self._fig
        with io.BytesIO() as buff:
            fig.savefig(buff, format="raw")
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = data.reshape((int(h), int(w), -1))
        return img

    def _plt_set_figsize(self, width: float, height: float):
        dpi = self._fig.get_dpi()
        self._fig.set_size_inches(width / dpi, height / dpi)
        self._fig.tight_layout()
