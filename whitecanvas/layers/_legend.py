from __future__ import annotations

from dataclasses import dataclass

from cmap import Color

from whitecanvas.types import Hatch, LineStyle, Symbol


@dataclass(frozen=True)
class LegendItem:
    pass


@dataclass(frozen=True)
class FaceInfo:
    color: Color
    hatch: Hatch


@dataclass(frozen=True)
class EdgeInfo:
    color: Color
    width: float
    style: LineStyle


class EmptyLegendItem(LegendItem):
    pass


class LineLegendItem(EdgeInfo):
    pass


@dataclass(frozen=True)
class ErrorLegendItem(EdgeInfo):
    color: Color
    width: float
    style: LineStyle
    capsize: float


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
