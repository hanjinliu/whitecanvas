from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cmap import Color

from whitecanvas.types import Hatch, LineStyle, Symbol


@dataclass(frozen=True)
class LegendItem:
    pass


@dataclass(frozen=True)
class FaceInfo:
    color: Color
    hatch: Hatch = Hatch.SOLID


@dataclass(frozen=True)
class EdgeInfo:
    color: Color
    width: float
    style: LineStyle = LineStyle.SOLID


class EmptyLegendItem(LegendItem):
    pass


class LineLegendItem(EdgeInfo):
    pass


@dataclass(frozen=True)
class ErrorLegendItem(EdgeInfo):
    color: Color
    width: float
    style: LineStyle = LineStyle.SOLID
    capsize: float = 1.0


@dataclass(frozen=True)
class MarkersLegendItem(LegendItem):
    symbol: Symbol
    size: float
    face: FaceInfo
    edge: EdgeInfo


@dataclass(frozen=True)
class BarLegendItem(LegendItem):
    face: FaceInfo
    edge: EdgeInfo

    @classmethod
    def simple(cls, color, hatch, edge_color):
        hatch = Hatch(hatch)
        return BarLegendItem(
            FaceInfo(np.array(Color(color).rgba), hatch),
            EdgeInfo(np.array(Color(edge_color).rgba), 1.0, LineStyle.SOLID),
        )


@dataclass(frozen=True)
class PlotLegendItem(LegendItem):
    line: LineLegendItem
    markers: MarkersLegendItem


@dataclass(frozen=True)
class LineErrorLegendItem(LegendItem):
    line: LineLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None


@dataclass(frozen=True)
class MarkerErrorLegendItem(LegendItem):
    markers: MarkersLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None


@dataclass(frozen=True)
class PlotErrorLegendItem(LegendItem):
    plot: PlotLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None


@dataclass(frozen=True)
class StemLegendItem(LegendItem):
    line: LineLegendItem
    markers: MarkersLegendItem


@dataclass(frozen=True)
class LegendItemCollection(LegendItem):
    items: list[tuple[str, LegendItem]]


@dataclass(frozen=True)
class TitleItem(LegendItem):
    pass
