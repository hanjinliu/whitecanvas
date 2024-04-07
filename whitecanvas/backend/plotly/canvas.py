from __future__ import annotations

import sys
import warnings
import weakref
from typing import TYPE_CHECKING, Callable

import numpy as np
from plotly import graph_objects as go

from whitecanvas import protocols
from whitecanvas.backend.plotly._base import (
    FigureLocation,
    PlotlyHoverableLayer,
    PlotlyLayer,
)
from whitecanvas.backend.plotly._labels import Axis, AxisLabel, Ticks, Title
from whitecanvas.backend.plotly._legend import make_sample_item
from whitecanvas.layers._legend import LegendItem, LegendItemCollection
from whitecanvas.types import Location, MouseEvent
from whitecanvas.utils.normalize import rgba_str_color

if TYPE_CHECKING:
    from plotly._subplots import SubplotXY


class Canvas:
    def __init__(
        self,
        fig: go.Figure,
        *,
        row: int = 0,
        col: int = 0,
        secondary_y: bool = False,
        app: str = "default",
    ):
        self._fig = fig
        self._loc = FigureLocation(row + 1, col + 1, secondary_y)
        self._xaxis = Axis(self, axis="xaxis")
        self._yaxis = Axis(self, axis="yaxis")
        self._xticks = Ticks(self, axis="xaxis")
        self._yticks = Ticks(self, axis="yaxis")
        self._title = Title(self)
        self._xlabel = AxisLabel(self, axis="xaxis")
        self._ylabel = AxisLabel(self, axis="yaxis")

    def _subplot_layout(self) -> SubplotXY:
        try:
            layout = self._fig.get_subplot(**self._loc.asdict())
        except Exception:  # manually wrapped backend are not created with subplots
            layout = self._fig.layout
        return layout

    def _plt_get_native(self):
        return self._fig

    def _plt_get_title(self):
        return self._title

    def _plt_get_xaxis(self):
        return self._xaxis

    def _plt_get_yaxis(self):
        return self._yaxis

    def _plt_get_xlabel(self):
        return self._xlabel

    def _plt_get_xticks(self):
        return self._xticks

    def _plt_get_yticks(self):
        return self._yticks

    def _plt_get_ylabel(self):
        return self._ylabel

    def _plt_reorder_layers(self, layers: list[PlotlyLayer]):
        model_to_idx_map = {id(layer._props): i for i, layer in enumerate(layers)}
        data = self._fig._data
        ordered_data = []
        data_in_other = []
        for _data in data:
            data_id = id(_data)
            if data_id in model_to_idx_map:
                ordered_data.append(data[model_to_idx_map[data_id]])
            else:
                data_in_other.append(_data)
        self._fig._data = [*ordered_data, *data_in_other]

    def _plt_get_aspect_ratio(self) -> float | None:
        """Get aspect ratio of canvas"""
        try:
            locked = self._fig["layout"][self._yaxis.name]["scaleanchor"] == "x"
        except KeyError:
            locked = False
        if locked:
            return 1
        return None

    def _plt_set_aspect_ratio(self, ratio: float | None):
        """Set aspect ratio of canvas"""
        if ratio is None:
            self._fig["layout"][self._yaxis.name]["scaleanchor"] = None
        elif ratio == 1:
            self._fig["layout"][self._yaxis.name]["scaleanchor"] = "x"
        else:
            raise NotImplementedError(
                f"Invalid aspect ratio for plotly backend: {ratio}"
            )

    def _plt_add_layer(self, layer: PlotlyLayer):
        self._fig.add_trace(layer._props, **self._loc.asdict())
        layer._props = self._fig.data[-1]
        layer._props["uid"] = layer._props.uid
        if isinstance(layer, PlotlyHoverableLayer):
            layer._connect_mouse_events(self._fig)

    def _plt_remove_layer(self, layer: PlotlyLayer):
        """Remove layer from the canvas"""
        # self._fig._data.remove(layer._props)
        for i, trace in enumerate(self._fig.data):
            if trace["uid"] == layer._props.uid:
                self._fig.data = [*self._fig.data[:i], *self._fig.data[i + 1 :]]
                break
        else:
            raise ValueError(f"Layer {layer._props} not found")

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return True

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        if not visible:
            warnings.warn(
                "Plotly backend does not support hiding canvas",
                UserWarning,
                stacklevel=2,
            )

    def _plt_connect_xlim_changed(self, callback):
        propname = f"{self._xaxis.name}.range"
        self._fig.layout.on_change(lambda _, lim: callback(lim), propname)

    def _plt_connect_ylim_changed(self, callback):
        propname = f"{self._yaxis.name}.range"
        self._fig.layout.on_change(lambda _, lim: callback(lim), propname)

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""
        # TODO

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to drag event"""
        # TODO

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to double-clicked event"""
        # TODO

    def _plt_connect_mouse_release(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

    def _plt_draw(self):
        pass

    def _plt_get_mouse_enabled(self):
        return self._fig.layout.xaxis.fixedrange

    def _plt_set_mouse_enabled(self, enabled: bool):
        self._fig.layout.xaxis.fixedrange = not enabled
        self._fig.layout.yaxis.fixedrange = not enabled

    def _plt_twinx(self):
        from plotly._subplots import SubplotRef

        # count axis numbers
        y_id = 1
        for _c in self._fig._grid_ref:
            for _r in _c:
                for _ in _r:
                    y_id += 1
        if y_id == 1:
            raise RuntimeError("did not find any existing axis.")
        cur_ref = self._fig._grid_ref[self._loc.row - 1][self._loc.col - 1]
        x_id: str = cur_ref[0].layout_keys[0].replace("xaxis", "")
        cur_ref = (
            *cur_ref,
            SubplotRef(
                subplot_type="xy",
                layout_keys=(f"xaxis{x_id}", f"yaxis{y_id}"),
                trace_kwargs={"xaxis": f"x{x_id}", "yaxis": f"y{y_id}"},
            ),
        )
        self._fig._grid_ref[self._loc.row - 1][self._loc.col - 1] = cur_ref
        self._fig.update_layout(
            yaxis2={
                "overlaying": "y",
                "side": "right",
            }
        )
        kwargs = self._loc.asdict()
        kwargs["secondary_y"] = True
        return Canvas(self._fig, **kwargs)

    def _plt_make_legend(
        self,
        items: list[tuple[str, LegendItem]],
        anchor: Location = Location.TOP_RIGHT,
    ):
        plotly_traces = []
        for label, item in items:
            if item is None:
                continue
            if isinstance(item, LegendItemCollection):
                for sub_label, sub_item in item.items:
                    sample = make_sample_item(sub_item)
                    if sample is not None:
                        sample.name = sub_label
                        plotly_traces.append(sample)
            else:
                sample = make_sample_item(item)
                if sample is not None:
                    sample.name = label
                    plotly_traces.append(sample)
        legend_kwargs = _LEGEND_KWARGS[anchor]
        self._fig.add_traces(plotly_traces, **self._loc.asdictn())
        self._fig.update_layout(showlegend=True, legend=legend_kwargs, overwrite=True)

    def _repr_mimebundle_(self, *args, **kwargs):
        return self._fig._repr_mimebundle_(*args, **kwargs)


_LEGEND_KWARGS = {
    Location.TOP_LEFT: {
        "x": 0.01,
        "y": 0.99,
        "xanchor": "left",
        "yanchor": "top",
    },
    Location.TOP_CENTER: {
        "x": 0.5,
        "y": 0.99,
        "xanchor": "center",
        "yanchor": "top",
    },
    Location.TOP_RIGHT: {
        "x": 0.99,
        "y": 0.99,
        "xanchor": "right",
        "yanchor": "top",
    },
    Location.BOTTOM_LEFT: {
        "x": 0.01,
        "y": 0.01,
        "xanchor": "left",
        "yanchor": "bottom",
    },
    Location.BOTTOM_CENTER: {
        "x": 0.5,
        "y": 0.01,
        "xanchor": "center",
        "yanchor": "bottom",
    },
    Location.BOTTOM_RIGHT: {
        "x": 0.99,
        "y": 0.01,
        "xanchor": "right",
        "yanchor": "bottom",
    },
    Location.CENTER_LEFT: {
        "x": 0.01,
        "y": 0.5,
        "xanchor": "left",
        "yanchor": "middle",
    },
    Location.CENTER_RIGHT: {
        "x": 0.99,
        "y": 0.5,
        "xanchor": "right",
        "yanchor": "middle",
    },
    Location.CENTER: {
        "x": 0.5,
        "y": 0.5,
        "xanchor": "center",
        "yanchor": "middle",
    },
    Location.TOP_SIDE_LEFT: {
        "x": 0.01,
        "y": 1.03,
        "xanchor": "left",
        "yanchor": "bottom",
    },
    Location.TOP_SIDE_CENTER: {
        "x": 0.5,
        "y": 1.03,
        "xanchor": "center",
        "yanchor": "bottom",
    },
    Location.TOP_SIDE_RIGHT: {
        "x": 0.99,
        "y": 1.03,
        "xanchor": "right",
        "yanchor": "bottom",
    },
    Location.BOTTOM_SIDE_LEFT: {
        "x": 0.01,
        "y": -0.03,
        "xanchor": "left",
        "yanchor": "top",
    },
    Location.BOTTOM_SIDE_CENTER: {
        "x": 0.5,
        "y": -0.03,
        "xanchor": "center",
        "yanchor": "top",
    },
    Location.BOTTOM_SIDE_RIGHT: {
        "x": 0.99,
        "y": -0.03,
        "xanchor": "right",
        "yanchor": "top",
    },
    Location.LEFT_SIDE_TOP: {
        "x": -0.03,
        "y": 0.99,
        "xanchor": "right",
        "yanchor": "top",
    },
    Location.LEFT_SIDE_CENTER: {
        "x": -0.03,
        "y": 0.5,
        "xanchor": "right",
        "yanchor": "middle",
    },
    Location.LEFT_SIDE_BOTTOM: {
        "x": -0.03,
        "y": 0.01,
        "xanchor": "right",
        "yanchor": "bottom",
    },
    Location.RIGHT_SIDE_TOP: {
        "x": 1.03,
        "y": 0.99,
        "xanchor": "left",
        "yanchor": "top",
    },
    Location.RIGHT_SIDE_CENTER: {
        "x": 1.03,
        "y": 0.5,
        "xanchor": "left",
        "yanchor": "middle",
    },
    Location.RIGHT_SIDE_BOTTOM: {
        "x": 1.03,
        "y": 0.01,
        "xanchor": "left",
        "yanchor": "bottom",
    },
}


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[float], widths: list[float], app: str = "default"):
        from plotly.subplots import make_subplots

        if app == "notebook":
            fig_class = go.FigureWidget
        else:
            fig_class = go.Figure
        self._figs = fig_class(
            make_subplots(
                rows=len(heights),
                cols=len(widths),
                row_heights=heights,
                column_widths=widths,
            ),
        )
        self._figs.update_layout(margin={"l": 6, "r": 6, "t": 30, "b": 6})
        self._app = app
        self._heights = heights
        self._widths = widths

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int) -> Canvas:
        if rowspan > 1 or colspan > 1:
            raise NotImplementedError("Plotly backend does not support rowspan/colspan")
        return Canvas(self._figs, row=row, col=col, app=self._app)

    def _plt_show(self):
        if self._app in ("qt", "wx", "tk"):
            return NotImplemented
        if self._app == "notebook" and "IPython" in sys.modules:
            from IPython import get_ipython
            from IPython.display import display

            if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
                display(self._figs)
                return
        self._figs.show(renderer="browser")

    def _plt_get_background_color(self):
        return self._figs.layout.paper_bgcolor

    def _plt_set_background_color(self, color):
        self._figs.layout.paper_bgcolor = rgba_str_color(color)

    def _plt_screenshot(self):
        import io

        from PIL import Image

        width, height = self._figs.layout.width, self._figs.layout.height
        img_bytes = self._figs.to_image(format="png", width=width, height=height)
        image = Image.open(io.BytesIO(img_bytes))
        return np.asarray(image, dtype=np.uint8)

    def _plt_set_figsize(self, width: int, height: int):
        self._figs.layout.width = width
        self._figs.layout.height = height

    def _plt_set_spacings(self, wspace: float, hspace: float):
        # plotly does not have a flexible way to set spacings
        pass
