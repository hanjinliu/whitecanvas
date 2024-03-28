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
from whitecanvas.backend.bokeh._legend import make_sample_item
from whitecanvas.layers._legend import LegendItem, LegendItemCollection
from whitecanvas.types import (
    Location,
    Modifier,
    MouseButton,
    MouseEvent,
    MouseEventType,
)
from whitecanvas.utils.normalize import arr_color, hex_color


def _prep_plot(width=400, height=300) -> bk_plotting.figure:
    plot = bk_plotting.figure(
        width=width,
        height=height,
        tooltips="@hovertexts",
    )
    plot.title.align = "center"
    return plot


SECOND_X = "second-x"
SECOND_Y = "second-y"


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(
        self,
        plot: bk_models.Plot,
        second_x: bool = False,
        second_y: bool = False,
    ):
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
        self._second_x = second_x
        self._second_y = second_y

    def _set_mouse_down(self, event):
        self._mouse_button = event

    def _get_xaxis(self):
        if not self._second_x:
            return self._plot.xaxis[0]
        return self._plot.xaxis[1]

    def _get_yaxis(self):
        if not self._second_y:
            return self._plot.yaxis[0]
        return self._plot.yaxis[1]

    def _get_xrange(self):
        if not self._second_x:
            return self._plot.x_range
        return self._plot.extra_x_ranges[SECOND_X]

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
        # NOTE: if plot has second_x or second_y, renderers will have more than layers
        model_to_idx_map = {id(layer._model): i for i, layer in enumerate(layers)}
        existing_glyphs = {id(layer._model) for layer in layers}
        renderers = self._bokeh_renderers()
        idx_converter = []
        for i, renderer in enumerate(renderers):
            if id(renderer.glyph) in existing_glyphs:
                idx_converter.append(i)
        renderers_sorted = []
        for r in renderers:
            idx = model_to_idx_map.get(id(r.glyph))
            if idx is not None:
                renderers_sorted.append(renderers[idx_converter[idx]])
            else:
                renderers_sorted.append(r)
        self._plot.renderers = renderers_sorted

    def _plt_get_aspect_ratio(self) -> float | None:
        return self._plot.aspect_ratio

    def _plt_set_aspect_ratio(self, ratio: float | None):
        self._plot.aspect_ratio = ratio

    def _plt_add_layer(self, layer: BokehLayer):
        if self._second_y:
            self._plot.add_glyph(layer._data, layer._model, y_range_name=SECOND_Y)
        elif self._second_x:
            self._plot.add_glyph(layer._data, layer._model, x_range_name=SECOND_X)
        else:
            self._plot.add_glyph(layer._data, layer._model)

    def _plt_remove_layer(self, layer: BokehLayer):
        """Remove layer from the canvas"""
        idx = -1
        for i, renderer in enumerate(self._bokeh_renderers()):
            if renderer.glyph == layer._model:
                idx = i
                del self._plot.renderers[idx]
                break
        else:
            raise ValueError(f"Layer {layer} not found")

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

    def _plt_get_mouse_enabled(self):
        return self._plot.toolbar.active_drag is None

    def _plt_set_mouse_enabled(self, enabled: bool):
        self._plot.toolbar.active_drag = "auto" if enabled else None

    def _plt_twinx(self):
        self._plot.extra_y_ranges[SECOND_Y] = bk_models.DataRange1d()
        self._plot.add_layout(
            bk_models.LinearAxis(y_range_name=SECOND_Y),
            "right",
        )
        return Canvas(self._plot, second_y=True)

    def _plt_twiny(self):
        self._plot.extra_x_ranges[SECOND_X] = bk_models.DataRange1d()
        self._plot.add_layout(
            bk_models.LinearAxis(x_range_name=SECOND_X),
            "above",
        )
        return Canvas(self._plot, second_x=True)

    def _plt_make_legend(
        self,
        items: list[tuple[str, LegendItem]],
        anchor: Location = Location.TOP_RIGHT,
    ):
        bk_items = []
        bk_samples = []
        for label, item in items:
            if item is None:
                continue
            if isinstance(item, LegendItemCollection):
                for sub_label, sub_item in item.items:
                    sample = make_sample_item(sub_item)
                    if sample is not None:
                        bk_items.append((sub_label, sample))
                        bk_samples.extend(sample)
            else:
                sample = make_sample_item(item)
                if sample is not None:
                    bk_items.append((label, sample))
                    bk_samples.extend(sample)
        location, side = _LEGEND_LOCATIONS[anchor]
        legend = bk_models.Legend(items=bk_items, location=location)
        self._plot.add_layout(legend, side)
        self._plot.renderers.extend(bk_samples)


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


# location and side
_LEGEND_LOCATIONS = {
    Location.TOP_LEFT: ("top_left", "center"),
    Location.TOP_CENTER: ("center_top", "center"),
    Location.TOP_RIGHT: ("top_right", "center"),
    Location.CENTER_LEFT: ("center_left", "center"),
    Location.CENTER: ("center_center", "center"),
    Location.CENTER_RIGHT: ("center_right", "center"),
    Location.BOTTOM_LEFT: ("bottom_left", "center"),
    Location.BOTTOM_CENTER: ("bottom_center", "center"),
    Location.BOTTOM_RIGHT: ("bottom_right", "center"),
    Location.LEFT_SIDE_TOP: ("top", "left"),
    Location.LEFT_SIDE_CENTER: ("center", "left"),
    Location.LEFT_SIDE_BOTTOM: ("bottom", "left"),
    Location.RIGHT_SIDE_TOP: ("top", "right"),
    Location.RIGHT_SIDE_CENTER: ("center", "right"),
    Location.RIGHT_SIDE_BOTTOM: ("bottom", "right"),
    Location.TOP_SIDE_LEFT: ("left", "above"),
    Location.TOP_SIDE_CENTER: ("center", "above"),
    Location.TOP_SIDE_RIGHT: ("right", "above"),
    Location.BOTTOM_SIDE_LEFT: ("left", "below"),
    Location.BOTTOM_SIDE_CENTER: ("center", "below"),
    Location.BOTTOM_SIDE_RIGHT: ("right", "below"),
}


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
                wheel_zoom = bk_models.WheelZoomTool()
                p.add_tools(wheel_zoom)
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
        for _, _, child in self._iter_bokeh_subplots():
            return child.background_fill_color
        return np.ones(4, dtype=np.float32)

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
            child.height = int(self._heights[r] / self._height_total * height)
            child.width = int(self._widths[c] / self._width_total * width)
        self._grid_plot.width = width
        self._grid_plot.height = height

    def _plt_set_spacings(self, wspace: float, hspace: float):
        self._grid_plot.spacing = (int(wspace), int(hspace))

    def _iter_bokeh_subplots(self) -> Iterator[tuple[int, int, bk_plotting.figure]]:
        for child, r, c in self._grid_plot.children:
            yield r, c, child
