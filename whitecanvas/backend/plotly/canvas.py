import sys
from typing import Callable
import weakref
import numpy as np
from plotly import graph_objects as go

from whitecanvas import protocols
from whitecanvas.types import MouseEventType, MouseEvent
from whitecanvas.utils.normalize import rgba_str_color
from .markers import Markers
from ._base import PlotlyLayer
from ._labels import Title, AxisLabel, Axis, Ticks


class Canvas:
    def __init__(self, fig: go.FigureWidget | None = None, row: int = 0, col: int = 0):
        # prepare widget
        if fig is None:
            fig = go.FigureWidget()
        self._fig = fig
        self._xaxis = Axis(self, axis="x")
        self._yaxis = Axis(self, axis="y")
        self._xticks = Ticks(self, axis="x")
        self._yticks = Ticks(self, axis="y")
        self._title = Title(self)
        self._xlabel = AxisLabel(self, axis="x")
        self._ylabel = AxisLabel(self, axis="y")
        # add empty scatter just for click events
        self._scatter = go.Scatter(
            x=[], y=[], mode="markers", marker_opacity=0, showlegend=False
        )
        self._fig.add_trace(self._scatter)
        self._row = row + 1
        self._col = col + 1

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
        first, *data = self._fig._data
        self._fig._data = [first] + [data[model_to_idx_map[id(r)]] for r in data]

    def _plt_get_aspect_ratio(self) -> float | None:
        """Get aspect ratio of canvas"""
        try:
            locked = self._fig['layout']['yaxis']['scaleanchor'] == 'x'
        except KeyError:
            locked = False
        if locked:
            return 1
        return None

    def _plt_set_aspect_ratio(self, ratio: float | None):
        """Set aspect ratio of canvas"""
        if ratio is None:
            self._fig['layout']['yaxis']['scaleanchor'] = None
        elif ratio == 1:
            self._fig['layout']['yaxis']['scaleanchor'] = 'x'
        else:
            raise NotImplementedError(
                f"Invalid aspect ratio for plotly backend: {ratio}"
            )

    def _plt_add_layer(self, layer: PlotlyLayer):
        self._fig.add_trace(layer._props, row=self._row, col=self._col)
        layer._props = self._fig._data[-1]
        if isinstance(layer, Markers):
            gobj: go.Scatter = self._fig.data[-1]
            for cb in layer._click_callbacks:
                gobj.on_click(_convert_cb(cb), append=True)
            layer._fig_ref = weakref.ref(self._fig)

    def _plt_remove_layer(self, layer: PlotlyLayer):
        """Remove layer from the canvas"""
        self._fig._data.remove(layer._props)

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self._fig.layout.visibility == "visible"

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        if visible:
            self._fig.layout.visibility = "visible"
        else:
            self._fig.layout.visibility = "hidden"

    def _plt_connect_xlim_changed(self, callback):
        self._fig.layout.on_change(lambda _, lim: callback(lim), 'xaxis.range')

    def _plt_connect_ylim_changed(self, callback):
        self._fig.layout.on_change(lambda _, lim: callback(lim), 'yaxis.range')

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""
        # TODO

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to drag event"""
        # TODO

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to double-clicked event"""
        # TODO

    def _plt_draw(self):
        pass


def _convert_cb(cb):
    return lambda _, points, state: cb(points.point_inds)


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[int], widths: list[int], app: str = "default"):
        from plotly.subplots import make_subplots

        self._figs = go.FigureWidget(
            make_subplots(
                rows=len(heights),
                cols=len(widths),
                row_heights=heights,
                column_widths=widths,
            )
        )
        self._figs.update_layout(margin=dict(l=6, r=6, t=6, b=6))
        self._app = app

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int) -> Canvas:
        return Canvas(self._figs, row=row, col=col)

    def _plt_show(self):
        if self._app in ("qt", "wx", "tk"):
            return NotImplemented
        if self._app == "notebook" or "IPython" in sys.modules:
            from IPython.display import display
            from IPython import get_ipython

            if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
                display(self._figs)
                return
        self._figs.show()

    def _plt_get_background_color(self):
        return self._figs.layout.paper_bgcolor

    def _plt_set_background_color(self, color):
        self._figs.layout.paper_bgcolor = rgba_str_color(color)

    def _plt_screenshot(self):
        from PIL import Image
        import io

        width, height = self._figs.layout.width, self._figs.layout.height
        img_bytes = self._figs.to_image(format="png", width=width, height=height)
        image = Image.open(io.BytesIO(img_bytes))
        return np.asarray(image, dtype=np.uint8)

    def _plt_set_figsize(self, width: float, height: float):
        self._figs.layout.width = width
        self._figs.layout.height = height
