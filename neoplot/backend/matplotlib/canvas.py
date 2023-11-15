from __future__ import annotations

from typing import Callable
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent as mplMouseEvent, MouseButton as mplMouseButton
from neoplot import protocols
from .line import Line
from .scatter import Scatter
from ._labels import Title, XAxis, YAxis, XLabel, YLabel
from neoplot.types import MouseEvent, Modifier, MouseButton, MouseEventType


class Canvas:
    def __init__(self, *, ax: plt.Axes | None = None):
        if ax is None:
            ax = plt.gca()
        self._axes = ax
        self._xaxis = XAxis(self)
        self._yaxis = YAxis(self)
        self._title = Title(self)
        self._xlabel = XLabel(self)
        self._ylabel = YLabel(self)

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

    def _plt_insert_layer(self, idx: int, layer: protocols.BaseProtocol):
        if isinstance(layer, Line):
            self._axes.add_line(layer)
        elif isinstance(layer, Scatter):
            self._axes.add_collection(layer)
        else:
            raise NotImplementedError
        self._axes.autoscale_view()  # TODO: remove this line

    def _plt_remove_layer(self, layer):
        """Remove layer from the canvas"""
        raise NotImplementedError

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

    def _translate_mouse_event(self, ev: mplMouseEvent, typ: MouseEventType) -> MouseEvent:
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

    def _plt_connect_xlim_changed(self, callback: Callable[[tuple[float, float]], None]):
        """Connect callback to x-limits changed event"""
        self._axes.callbacks.connect("xlim_changed", lambda ax: callback(ax.get_xlim()))

    def _plt_connect_ylim_changed(self, callback: Callable[[tuple[float, float]], None]):
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


class MainCanvas:
    def __init__(self):
        self._canvas = Canvas()

    def _plt_get_canvas(self) -> protocols.CanvasProtocol:
        return self._canvas

    def _plt_get_visible(self) -> bool:
        return True

    def _plt_set_visible(self, visible: bool):
        pass
