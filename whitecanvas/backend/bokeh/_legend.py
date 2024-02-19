from functools import singledispatch

import bokeh.models as bk_models

from whitecanvas.backend.bokeh._base import (
    to_bokeh_hatch,
    to_bokeh_line_style,
    to_bokeh_symbol,
)
from whitecanvas.layers import _legend as _leg
from whitecanvas.utils.normalize import hex_color

_DATA = bk_models.ColumnDataSource({"x": [0], "y": [0]})


@singledispatch
def make_sample_item(item) -> "list[bk_models.GlyphRenderer] | None":
    return None


@make_sample_item.register
def _(item: _leg.LineLegendItem):
    line = bk_models.Line(
        x="x", y="x", line_dash=to_bokeh_line_style(item.style),
        line_color=hex_color(item.color), line_width=item.width,
    )  # fmt: skip
    return [bk_models.GlyphRenderer(glyph=line, data_source=_DATA, visible=False)]


@make_sample_item.register
def _(item: _leg.MarkersLegendItem):
    marker, angle = to_bokeh_symbol(item.symbol)
    sc = bk_models.Scatter(
        x="x", y="x", fill_color=hex_color(item.face.color),
        marker=marker, angle=angle,
        hatch_pattern=to_bokeh_hatch(item.face.hatch),
        line_color=hex_color(item.edge.color),
        line_width=item.edge.width, line_dash=to_bokeh_line_style(item.edge.style),
        size=item.size,
    )  # fmt: skip
    return [bk_models.GlyphRenderer(glyph=sc, data_source=_DATA, visible=False)]


@make_sample_item.register
def _(item: _leg.BarLegendItem):
    quad = bk_models.Quad(
        left="x",
        right="x",
        bottom="x",
        top="x",
        fill_color=hex_color(item.face.color),
        hatch_pattern=to_bokeh_hatch(item.face.hatch),
        line_color=hex_color(item.edge.color),
        line_width=item.edge.width,
        line_dash=to_bokeh_line_style(item.edge.style),
    )
    return [bk_models.GlyphRenderer(glyph=quad, data_source=_DATA, visible=False)]


@make_sample_item.register
def _(item: _leg.PlotLegendItem):
    return make_sample_item(item.line) + make_sample_item(item.markers)


@make_sample_item.register
def _(item: _leg.ErrorLegendItem):
    line = bk_models.Line(
        x="x", y="x", line_dash=to_bokeh_line_style(item.style),
        line_color=hex_color(item.color), line_width=item.width,
    )  # fmt: skip
    return [bk_models.GlyphRenderer(glyph=line, data_source=_DATA, visible=False)]


@make_sample_item.register
def _(item: _leg.LineErrorLegendItem):
    items = make_sample_item(item.line)
    if item.xerr is not None:
        items.extend(make_sample_item(item.xerr))
    if item.yerr is not None:
        items.extend(make_sample_item(item.yerr))
    return items


@make_sample_item.register
def _(item: _leg.MarkerErrorLegendItem):
    items = make_sample_item(item.markers)
    if item.xerr is not None:
        items.extend(make_sample_item(item.xerr))
    if item.yerr is not None:
        items.extend(make_sample_item(item.yerr))
    return items


@make_sample_item.register
def _(item: _leg.PlotErrorLegendItem):
    items = make_sample_item(item.plot)
    if item.xerr is not None:
        items.extend(make_sample_item(item.xerr))
    if item.yerr is not None:
        items.extend(make_sample_item(item.yerr))
    return items


@make_sample_item.register
def _(item: _leg.StemLegendItem):
    return make_sample_item(item.markers)


@make_sample_item.register
def _(item: _leg.TitleItem):
    line = bk_models.Line(x="x", y="x", line_color="#00000000", line_width=0)
    return [bk_models.GlyphRenderer(glyph=line, data_source=_DATA, visible=False)]
