from __future__ import annotations

import bokeh.models as bk_models
import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend.bokeh._base import (
    BokehLayer,
    SupportsMouseEvents,
    from_bokeh_line_style,
    to_bokeh_line_style,
)
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.utils.normalize import arr_color, hex_color


@check_protocol(LineProtocol)
class MonoLine(BokehLayer[bk_models.Line], SupportsMouseEvents):
    def __init__(self, xdata, ydata):
        self._data = bk_models.ColumnDataSource(
            data={"x": xdata, "y": ydata, "hovertexts": np.array([""] * len(xdata))}
        )
        self._model = bk_models.Line(x="x", y="y", line_join="round", line_cap="round")
        self._line_style = LineStyle.SOLID
        self._visible = True
        self._line_color = "#0000FF"

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._model.line_color = self._line_color
        else:
            self._model.line_color = "#00000000"
        self._visible = visible

    def _plt_get_data(self):
        return self._data.data["x"], self._data.data["y"]

    def _plt_set_data(self, xdata, ydata):
        self._data.data = {"x": xdata, "y": ydata}

    def _plt_get_edge_width(self) -> float:
        return self._model.line_width

    def _plt_set_edge_width(self, width: float):
        self._model.line_width = width

    def _plt_get_edge_style(self) -> LineStyle:
        return self._line_style

    def _plt_set_edge_style(self, style: LineStyle):
        self._model.line_dash = to_bokeh_line_style(style)
        self._line_style = style

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(arr_color(self._line_color))

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if self._visible:
            self._model.line_color = hex_color(color)
        self._line_color = hex_color(color)

    def _plt_get_antialias(self) -> bool:
        return self._model.line_join == "round"

    def _plt_set_antialias(self, antialias: bool):
        self._model.line_join = "round" if antialias else "miter"
        self._model.line_cap = "round" if antialias else "butt"


@check_protocol(MultiLineProtocol)
class MultiLine(BokehLayer[bk_models.MultiLine], SupportsMouseEvents):
    def __init__(self, data: list[NDArray[np.number]]):
        xdata: list[NDArray[np.number]] = []
        ydata: list[NDArray[np.number]] = []
        for seg in data:
            xdata.append(seg[:, 0])
            ydata.append(seg[:, 1])
        self._data = bk_models.ColumnDataSource(
            {
                "x": xdata,
                "y": ydata,
                "edge_color": ["blue"] * len(xdata),
                "width": [1.0] * len(xdata),
                "dash": ["solid"] * len(xdata),
                "hovertexts": [""] * len(xdata),
            }
        )
        self._model = bk_models.MultiLine(
            xs="x",
            ys="y",
            line_color="edge_color",
            line_width="width",
            line_dash="dash",
            line_join="round",
            line_cap="round",
        )
        self._visible = True
        self._line_color = "#0000FF"

    def _plt_get_ndata(self):
        return len(self._data.data["x"])

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._model.line_color = "edge_color"
        else:
            self._model.line_color = "#00000000"
        self._visible = visible

    def _plt_get_data(self):
        xs, ys = self._data.data["x"], self._data.data["y"]
        out = []
        for x, y in zip(xs, ys):
            out.append(np.stack([x, y], axis=1))
        return out

    def _plt_set_data(self, data):
        xdata = []
        ydata = []
        for seg in data:
            xdata.append(seg[:, 0])
            ydata.append(seg[:, 1])
        ndata = self._plt_get_ndata()
        edge_color = self._data.data["edge_color"]
        width = self._data.data["width"]
        dash = self._data.data["dash"]
        hovertexts = self._data.data["hovertexts"]
        if len(data) < ndata:
            loss = ndata - len(data)
            edge_color = edge_color[:-loss]
            width = width[:-loss]
            dash = dash[:-loss]
            hovertexts = hovertexts[:-loss]
        elif len(data) > ndata:
            if ndata == 0:
                edge_color = ["blue"] * len(data)
                width = [1.0] * len(data)
                dash = ["solid"] * len(data)
                hovertexts = [""] * len(data)
            else:
                gain = len(data) - ndata
                edge_color = edge_color + edge_color[-1] * gain
                width = width + width[-1] * gain
                dash = dash + dash[-1] * gain
                hovertexts = hovertexts + [""] * gain
        data = {
            "x": xdata,
            "y": ydata,
            "edge_color": edge_color,
            "width": width,
            "dash": dash,
            "hovertexts": hovertexts,
        }
        self._data.data.update(data)

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self._data.data["width"]

    def _plt_set_edge_width(self, width: float):
        if np.isscalar(width):
            width = np.full(self._plt_get_ndata(), width)
        self._data.data["width"] = width

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return [from_bokeh_line_style(d) for d in self._data.data["dash"]]

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            style = [style] * self._plt_get_ndata()
        val = [to_bokeh_line_style(s) for s in style]
        self._data.data["dash"] = val

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.stack([arr_color(c) for c in self._data.data["edge_color"]], axis=0)

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.size == 0:
            return
        if color.ndim == 1:
            color = [hex_color(color)] * self._plt_get_ndata()
        else:
            color = [hex_color(c) for c in color]
        self._data.data["edge_color"] = color

    def _plt_get_antialias(self) -> bool:
        return self._model.line_join == "round"

    def _plt_set_antialias(self, antialias: bool):
        self._model.line_join = "round" if antialias else "miter"
        self._model.line_cap = "round" if antialias else "butt"
