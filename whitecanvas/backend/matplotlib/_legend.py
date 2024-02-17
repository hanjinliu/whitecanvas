from functools import singledispatch

from matplotlib import patches
from matplotlib import transforms as mtransforms
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.lines import Line2D

from whitecanvas.backend.matplotlib._base import symbol_to_path
from whitecanvas.layers import _legend as _leg
from whitecanvas.types import Hatch


@singledispatch
def make_sample_item(item) -> "Artist | None":
    return None


@make_sample_item.register
def _(item: _leg.LineLegendItem):
    return Line2D(
        [], [], color=item.color, linewidth=item.width, linestyle=item.style.value
    )  # fmt: skip


@make_sample_item.register
def _(item: _leg.MarkersLegendItem):
    if item.face.hatch is Hatch.SOLID:
        hatch = None
    else:
        hatch = item.face.hatch.value
    col = PathCollection(
        (symbol_to_path(item.symbol),),
        sizes=[item.size**2],
        offsets=[(0, 0)],
        facecolors=item.face.color,
        edgecolors=item.edge.color,
        linewidths=[item.edge.width],
        linestyles=[item.edge.style.value],
        hatch=hatch,
    )
    col.set_transform(mtransforms.IdentityTransform())
    return col


@make_sample_item.register
def _(item: _leg.BarLegendItem):
    rect = patches.Rectangle(
        (0, 0), 1, 1, facecolor=item.face.color, edgecolor=item.edge.color,
        linewidth=item.edge.width, linestyle=item.edge.style.value,
        hatch=item.face.hatch.value
    )  # fmt: skip
    return BarContainer([rect])


@make_sample_item.register
def _(item: _leg.PlotLegendItem):
    return Line2D(
        [], [], color=item.line.color, linewidth=item.line.width,
        linestyle=item.line.style.value, marker=item.markers.symbol.value,
        markersize=item.markers.size, markerfacecolor=item.markers.face.color,
        markeredgecolor=item.markers.edge.color,
        markeredgewidth=item.markers.edge.width,
    )  # fmt: skip


@make_sample_item.register
def _(item: _leg.ErrorLegendItem):
    line_2d = Line2D([], [], color="#00000000")
    caps = Line2D([], [], color=item.color, linewidth=item.width)
    barlines = Line2D(
        [], [], color=item.color, linewidth=item.width, linestyle=item.style.value
    )  # fmt: skip
    return ErrorbarContainer((line_2d, (caps,), [barlines]), has_yerr=True)


@make_sample_item.register
def _(item: _leg.LineErrorLegendItem):
    line_2d = make_sample_item(item.line)
    caps, barlines, has_xerr, has_yerr = _norm_xyerr(item.xerr, item.yerr)
    return ErrorbarContainer(
        (line_2d, caps, barlines), has_xerr=has_xerr, has_yerr=has_yerr
    )


@make_sample_item.register
def _(item: _leg.MarkerErrorLegendItem):
    marker = _marker_as_line_2d(item.markers)
    caps, barlines, has_xerr, has_yerr = _norm_xyerr(item.xerr, item.yerr)
    return ErrorbarContainer(
        (marker, caps, barlines), has_xerr=has_xerr, has_yerr=has_yerr
    )


@make_sample_item.register
def _(item: _leg.PlotErrorLegendItem):
    plot = make_sample_item(item.plot)
    caps, barlines, has_xerr, has_yerr = _norm_xyerr(item.xerr, item.yerr)
    return ErrorbarContainer(
        (plot, caps, barlines), has_xerr=has_xerr, has_yerr=has_yerr
    )


@make_sample_item.register
def _(item: _leg.StemLegendItem):
    markers = _marker_as_line_2d(item.markers)
    markers.set_markersize(item.markers.size / 1.6)  # to make stem visible
    stemlines = make_sample_item(item.line)
    baseline = Line2D([], [], color="#00000000")
    return StemContainer((markers, [stemlines], baseline))


@make_sample_item.register
def _(item: _leg.TitleItem):
    return patches.Rectangle((0, 0), 0, 0, color="#00000000")


def _norm_xyerr(xerr: _leg.LineLegendItem | None, yerr: _leg.LineLegendItem | None):
    has_xerr = xerr is not None
    has_yerr = yerr is not None
    if has_xerr:
        er = _make_line_from_err(xerr)
        barline = _make_line_collection_from_err(xerr)
    elif has_yerr:
        er = _make_line_from_err(yerr)
        barline = _make_line_collection_from_err(yerr)
    else:
        er = None
        barline = []
    if er is None:
        caps = ()
    else:
        caps = (er,)
    return caps, [barline], has_xerr, has_yerr


def _make_line_from_err(item: _leg.LineLegendItem):
    return Line2D(
        [], [], color=item.color, linewidth=item.width, linestyle=item.style.value
    )  # fmt: skip


def _make_line_collection_from_err(item: _leg.LineLegendItem):
    return LineCollection(
        [], color=item.color, linewidth=item.width, linestyle=item.style.value
    )  # fmt: skip


def _marker_as_line_2d(item: _leg.MarkersLegendItem):
    return Line2D(
        [], [], color=None, marker=item.symbol.value, markersize=item.size,
        markerfacecolor=item.face.color, markeredgecolor=item.edge.color,
        markeredgewidth=item.edge.width
    )  # fmt: skip
