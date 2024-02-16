from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
from bokeh import events as bk_events
from bokeh import layouts as bk_layouts
from bokeh import models as bk_models
from bokeh import plotting as bk_plotting
from bokeh.io import output_notebook

from whitecanvas import protocols
from whitecanvas.backend.bokeh._base import BokehLayer
from whitecanvas.backend.bokeh._labels import (
    Title,
    XAxis,
    XLabel,
    XTicks,
    YAxis,
    YLabel,
    YTicks,
)
from whitecanvas.types import Modifier, MouseButton, MouseEvent, MouseEventType
from whitecanvas.utils.normalize import arr_color, hex_color


def _prep_plot(width=400, height=300) -> bk_plotting.figure:
    plot = bk_plotting.figure(
        width=width,
        height=height,
        tooltips="@hovertexts",
    )
    plot.title.align = "center"
    return plot


SECOND_Y = "second-y"


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(
        self,
        plot: bk_models.Plot | None = None,
        second_y: bool = False,
    ):
        if plot is None:
            plot = _prep_plot()
        assert isinstance(plot, bk_models.Plot)
        self._plot = plot
        self._xaxis = XAxis(self)
        self._yaxis = YAxis(self)
        self._title = Title(self)
        self._xlabel = XLabel(self)
        self._ylabel = YLabel(self)
        self._xticks = XTicks(self)
        self._yticks = YTicks(self)
        self._mouse_button: MouseButton = MouseButton.NONE
        self._second_y = second_y

    def _set_mouse_down(self, event):
        self._mouse_button = event

    def _get_xaxis(self):
        return self._plot.xaxis[0]

    def _get_yaxis(self):
        if not self._second_y:
            return self._plot.yaxis[0]
        return self._plot.yaxis[1]

    def _get_xrange(self):
        return self._plot.x_range

    def _get_yrange(self):
        if not self._second_y:
            return self._plot.y_range
        return self._plot.extra_y_ranges[SECOND_Y]

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
        if self._second_y:
            self._plot.add_glyph(layer._data, layer._model, y_range_name=SECOND_Y)
        else:
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

        def _cb(event: bk_events.Press):
            self._set_mouse_down(MouseButton.LEFT)
            ev = MouseEvent(
                button=MouseButton.LEFT,
                modifiers=_translate_modifiers(event.modifiers),
                pos=(event.x, event.y),
                type=MouseEventType.CLICK,
            )
            callback(ev)

        self._plot.on_event(bk_events.Tap, _cb)
        self._plot.on_event(bk_events.Press, _cb)

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(event: bk_events.MouseMove):
            ev = MouseEvent(
                button=self._mouse_button,
                modifiers=_translate_modifiers(event.modifiers),
                pos=(event.x, event.y),
                type=MouseEventType.MOVE,
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
                type=MouseEventType.DOUBLE_CLICK,
            )
            callback(ev)

        self._plot.on_event(bk_events.DoubleTap, _cb)

    def _plt_connect_mouse_release(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(event: bk_events.PressUp):
            ev = MouseEvent(
                button=MouseButton.LEFT,
                modifiers=_translate_modifiers(event.modifiers),
                pos=(event.x, event.y),
                type=MouseEventType.RELEASE,
            )
            callback(ev)
            self._set_mouse_down(MouseButton.NONE)

        self._plot.on_event(bk_events.PressUp, _cb)
        self._plot.on_event(bk_events.Tap, _cb)
        self._plot.on_event(bk_events.PanEnd, _cb)

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        rng = self._plot.x_range
        rng.on_change(
            "start",
            lambda attr, old, new: callback((rng.start, rng.end)),  # noqa: ARG005
        )

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        rng = self._plot.y_range
        rng.on_change(
            "start",
            lambda attr, old, new: callback((rng.start, rng.end)),  # noqa: ARG005
        )

    def _plt_draw(self):
        pass

    def _plt_twinx(self):
        self._plot.add_layout(
            bk_models.LinearAxis(y_range_name=SECOND_Y),
            "right",
        )
        self._plot.extra_y_ranges = {SECOND_Y: bk_models.DataRange1d()}
        return Canvas(self._plot, second_y=True)


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
    def __init__(self, heights: list[float], widths: list[float], app: str = "default"):
        hsum = sum(heights)
        wsum = sum(widths)
        children = []
        for h in heights:
            row = []
            for w in widths:
                p = _prep_plot(width=int(w / wsum * 600), height=int(h / hsum * 600))
                p.visible = False
                row.append(p)
            children.append(row)
        self._grid_plot: bk_layouts.GridPlot = bk_layouts.gridplot(
            children, sizing_mode="fixed"
        )
        self._widths = widths
        self._heights = heights
        self._width_total = wsum
        self._height_total = hsum
        self._app = app

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int) -> Canvas:
        for r0, c0, plot in self._iter_bokeh_subplots():
            if r0 == row and c0 == col:
                plot.visible = True
                return Canvas(plot)
        raise ValueError(f"Canvas at ({row}, {col}) not found")

    def _plt_show(self):
        if self._app == "notebook":
            output_notebook()
            bk_plotting.show(lambda doc: doc.add_root(self._grid_plot))
        else:
            bk_plotting.show(self._grid_plot)
            if self._grid_plot.document is None:
                bk_plotting.curdoc().add_root(self._grid_plot)

    def _plt_get_background_color(self):
        return arr_color(self._grid_plot.background_fill_color)

    def _plt_set_background_color(self, color):
        color = hex_color(color)
        for _, _, child in self._iter_bokeh_subplots():
            child.background_fill_color = color

    def _plt_screenshot(self):
        import io

        from bokeh.io import export_png

        with io.BytesIO() as buff:
            export_png(self._grid_plot, filename=buff)
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = self._grid_plot.width, self._grid_plot.height
        img = data.reshape((int(h), int(w), -1))
        return img

    def _plt_set_figsize(self, width: int, height: int):
        for r, c, child in self._iter_bokeh_subplots():
            child.height = int(self._heights[r] / self._height_total * width)
            child.width = int(self._widths[c] / self._width_total * height)
        self._grid_plot.width = width
        self._grid_plot.height = height

    def _plt_set_spacings(self, wspace: float, hspace: float):
        self._grid_plot.spacing = (int(wspace), int(hspace))

    def _iter_bokeh_subplots(self) -> Iterator[tuple[int, int, bk_plotting.figure]]:
        for child, r, c in self._grid_plot.children:
            yield r, c, child
