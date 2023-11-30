from __future__ import annotations

from typing import Callable

import numpy as np

from whitecanvas.utils.normalize import arr_color, hex_color
from ._labels import Title, XAxis, YAxis, XLabel, YLabel, XTicks, YTicks
from whitecanvas import protocols
from whitecanvas.types import MouseEvent, Modifier, MouseButton, MouseEventType
from bokeh import (
    events as bk_events,
    models as bk_models,
    layouts as bk_layouts,
    plotting as bk_plotting,
)
from bokeh.io.state import curstate
from ._base import BokehLayer


def _prep_plot(width=400, height=300):
    plot = bk_plotting.figure(width=width, height=height)
    return plot


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(self, plot: bk_models.Plot | None = None):
        if plot is None:
            plot = _prep_plot()
        self._plot = plot
        self._xaxis = XAxis(self)
        self._yaxis = YAxis(self)
        self._title = Title(self)
        self._xlabel = XLabel(self)
        self._ylabel = YLabel(self)
        self._xticks = XTicks(self)
        self._yticks = YTicks(self)
        self._mouse_button: MouseButton = MouseButton.NONE

        # connect default mouse events
        plot.on_event(bk_events.Press, lambda event: self._set_mouse_down(event))
        plot.on_event(
            bk_events.PressUp, lambda event: self._set_mouse_down(MouseButton.NONE)
        )

    def _set_mouse_down(self, event):
        self._mouse_button = MouseButton.LEFT

    def _plt_get_native(self):
        return self._plot

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

    def _bokeh_renderers(self) -> list[bk_models.GlyphRenderer]:
        return self._plot.renderers

    def _plt_reorder_layers(self, layers: list[BokehLayer]):
        model_to_idx_map = {id(layer._model): i for i, layer in enumerate(layers)}
        renderes = self._bokeh_renderers()
        self._plot.renderers = [
            renderes[model_to_idx_map[id(r.glyph)]] for r in renderes
        ]

    def _plt_get_aspect_ratio(self) -> float | None:
        return self._plot.aspect_ratio

    def _plt_set_aspect_ratio(self, ratio: float | None):
        self._plot.aspect_ratio = ratio

    def _plt_add_layer(self, layer: BokehLayer):
        self._plot.add_glyph(layer._data, layer._model)

    def _plt_remove_layer(self, layer: BokehLayer):
        """Remove layer from the canvas"""
        idx = -1
        for i, renderer in enumerate(self._bokeh_renderers()):
            if renderer.glyph == layer._model:
                idx = i
                break
        del self._plot.renderers[idx]

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self._plot.visible

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        self._plot.visible = visible

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(event: bk_events.Tap):
            ev = MouseEvent(
                button=MouseButton.LEFT,
                modifiers=_translate_modifiers(event.modifiers),
                pos=(event.x, event.y),
                type=MouseEventType.CLICK,
            )
            callback(ev)

        self._plot.on_event(bk_events.Tap, _cb)

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(event: bk_events.MouseMove):
            ev = MouseEvent(
                button=self._mouse_button,
                modifiers=_translate_modifiers(event.modifiers),
                pos=(event.x, event.y),
                type=MouseEventType.CLICK,
            )
            callback(ev)

        self._plot.on_event(bk_events.MouseMove, _cb)

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(event: bk_events.DoubleTap):
            ev = MouseEvent(
                button=MouseButton.LEFT,
                modifiers=_translate_modifiers(event.modifiers),
                pos=(event.x, event.y),
                type=MouseEventType.CLICK,
            )
            callback(ev)

        self._plot.on_event(bk_events.DoubleTap, _cb)

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        self._plot.x_range.on_change(
            "start", lambda attr, old, new: callback((new, self._plot.x_range.end))
        )

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        self._plot.y_range.on_change(
            "start", lambda attr, old, new: callback((new, self._plot.y_range.end))
        )


def _translate_modifiers(mod: bk_events.KeyModifiers | None) -> tuple[Modifier, ...]:
    if mod is None:
        return ()
    modifiers = []
    if mod["alt"]:
        modifiers.append(Modifier.ALT)
    if mod["shift"]:
        modifiers.append(Modifier.SHIFT)
    if mod["ctrl"]:
        modifiers.append(Modifier.CTRL)
    return tuple(modifiers)


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[int], widths: list[int], app: str = "default"):
        nr, nc = len(heights), len(widths)
        children = []
        for r in range(nr):
            row = []
            for c in range(nc):
                row.append(_prep_plot())
            children.append(row)
        self._grid_plot: bk_layouts.GridPlot = bk_layouts.gridplot(children)
        self._shape = (nr, nc)
        self._app = app

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int) -> Canvas:
        r1 = row + rowspan
        c1 = col + colspan
        return Canvas(self._grid_plot.children[row][col])

    def _plt_show(self):
        if is_notebook() or self._app == "notebook":
            bk_plotting.show(lambda doc: doc.add_root(self._grid_plot))
        else:
            bk_plotting.show(self._grid_plot)
            if self._grid_plot.document is None:
                bk_plotting.curdoc().add_root(self._grid_plot)

    def _plt_get_background_color(self):
        return arr_color(self._grid_plot.background_fill_color)

    def _plt_set_background_color(self, color):
        color = hex_color(color)
        for r in range(self._shape[0]):
            for c in range(self._shape[1]):
                child = self._grid_plot.children[r][c]
                if not hasattr(child, "background_fill_color"):
                    continue
                child.background_fill_color = color

    def _plt_screenshot(self):
        import io
        from bokeh.io import export_png

        with io.BytesIO() as buff:
            export_png(self._grid_plot, filename=buff)
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = self._grid_plot.plot_width, self._grid_plot.plot_height
        img = data.reshape((int(h), int(w), -1))
        return img

    def _plt_set_figsize(self, width: int, height: int):
        self._grid_plot.width = width
        self._grid_plot.height = height


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
    except NameError:
        return False
    state = curstate()
    if state.notebook_type is None:
        return False
    return shell == 'ZMQInteractiveShell'
