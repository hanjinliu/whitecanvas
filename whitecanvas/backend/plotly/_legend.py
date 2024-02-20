from functools import singledispatch
from typing import TYPE_CHECKING

from plotly import graph_objs as go

from whitecanvas.backend.plotly._base import (
    to_plotly_linestyle,
    to_plotly_marker_symbol,
)
from whitecanvas.layers import _legend as _leg
from whitecanvas.utils.normalize import hex_color

if TYPE_CHECKING:
    from plotly.basedatatypes import BaseTraceType


@singledispatch
def make_sample_item(item) -> "BaseTraceType | None":
    return None


@make_sample_item.register
def _(item: _leg.LineLegendItem):
    return go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line={
            "color": hex_color(item.color),
            "width": item.width,
            "dash": to_plotly_linestyle(item.style),
        },
        showlegend=True,
    )


@make_sample_item.register
def _(item: _leg.MarkersLegendItem):
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker={
            "color": hex_color(item.face.color),
            "size": item.size,
            "symbol": to_plotly_marker_symbol(item.symbol),
            "line": {"color": hex_color(item.edge.color), "width": item.edge.width},
        },
        showlegend=True,
    )


@make_sample_item.register
def _(item: _leg.ErrorLegendItem):
    # NOTE: plotly errorbars does not support dash
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        line={"color": hex_color(item.color), "width": item.width},
        marker={"color": "#000000", "size": 0},
        error_y={"type": "data", "array": None, "visible": True},
        showlegend=True,
    )


@make_sample_item.register
def _(item: _leg.LineErrorLegendItem):
    line = item.line
    if item.xerr is not None:
        error_x = {
            "array": [None],
            "visible": True,
            "color": hex_color(item.xerr.color),
            "width": item.xerr.width,
        }
    else:
        error_x = None
    if item.yerr is not None:
        error_y = {
            "array": [None],
            "visible": True,
            "color": hex_color(item.yerr.color),
            "width": item.yerr.width,
        }
    else:
        error_y = None
    return go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line={
            "color": hex_color(line.color),
            "width": line.width,
            "dash": to_plotly_linestyle(line.style),
        },
        error_y=error_y,
        error_x=error_x,
        showlegend=True,
    )


@make_sample_item.register
def _(item: _leg.MarkerErrorLegendItem):
    marker = item.markers
    if item.xerr is not None:
        error_x = {
            "array": [None],
            "visible": True,
            "color": hex_color(item.xerr.color),
            "width": item.xerr.width,
        }
    else:
        error_x = None
    if item.yerr is not None:
        error_y = {
            "array": [None],
            "visible": True,
            "color": hex_color(item.yerr.color),
            "width": item.yerr.width,
        }
    else:
        error_y = None
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker={
            "color": hex_color(marker.face.color),
            "size": marker.size,
            "symbol": to_plotly_marker_symbol(marker.symbol),
            "line": {"color": hex_color(marker.edge.color), "width": marker.edge.width},
        },
        error_y=error_y,
        error_x=error_x,
        showlegend=True,
    )


@make_sample_item.register
def _(item: _leg.PlotLegendItem):
    return _make_plot_item(item.line, item.markers)


@make_sample_item.register
def _(item: _leg.PlotErrorLegendItem):
    line = item.plot.line
    marker = item.plot.markers
    if item.xerr is not None:
        error_x = {
            "array": [None],
            "visible": True,
            "color": hex_color(item.xerr.color),
            "width": item.xerr.width,
        }
    else:
        error_x = None
    if item.yerr is not None:
        error_y = {
            "array": [None],
            "visible": True,
            "color": hex_color(item.yerr.color),
            "width": item.yerr.width,
        }
    else:
        error_y = None
    return go.Scatter(
        x=[None],
        y=[None],
        mode="lines+markers",
        line={
            "color": hex_color(line.color),
            "width": line.width,
            "dash": to_plotly_linestyle(line.style),
        },
        marker={
            "color": hex_color(marker.face.color),
            "size": marker.size,
            "symbol": to_plotly_marker_symbol(marker.symbol),
            "line": {"color": hex_color(marker.edge.color), "width": marker.edge.width},
        },
        error_y=error_y,
        error_x=error_x,
        showlegend=True,
    )


def _make_plot_item(line: _leg.LineLegendItem, marker: _leg.MarkersLegendItem):
    return go.Scatter(
        x=[None],
        y=[None],
        mode="lines+markers",
        line={
            "color": hex_color(line.color),
            "width": line.width,
            "dash": to_plotly_linestyle(line.style),
        },
        marker={
            "color": hex_color(marker.face.color),
            "size": marker.size,
            "symbol": to_plotly_marker_symbol(marker.symbol),
            "line": {"color": hex_color(marker.edge.color), "width": marker.edge.width},
        },
        showlegend=True,
    )


@make_sample_item.register
def _(item: _leg.StemLegendItem):
    line = item.line
    marker = item.markers
    return go.Scatter(
        x=[None],
        y=[None],
        mode="lines+markers",
        line={
            "color": hex_color(line.color),
            "width": line.width,
            "dash": to_plotly_linestyle(line.style),
        },
        marker={
            "color": hex_color(marker.face.color),
            "size": marker.size,
            "symbol": to_plotly_marker_symbol(marker.symbol),
            "line": {"color": hex_color(marker.edge.color), "width": marker.edge.width},
        },
        showlegend=True,
    )


@make_sample_item.register
def _(item: _leg.StemLegendItem):
    return _make_plot_item(item.line, item.markers)


@make_sample_item.register
def _(item: _leg.TitleItem):
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker={"color": "#000000", "size": 0},
        showlegend=True,
    )
