from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cmap import Color

from whitecanvas.types import Hatch, LineStyle, Symbol


@dataclass
class LegendItem:
    pass


@dataclass
class FaceInfo:
    color: Color
    hatch: Hatch = Hatch.SOLID

    def __post_init__(self):
        self.color = np.array(Color(self.color).rgba)
        self.hatch = Hatch(self.hatch)


@dataclass
class EdgeInfo:
    color: Color
    width: float
    style: LineStyle = LineStyle.SOLID

    def __post_init__(self):
        self.color = np.array(Color(self.color).rgba)
        self.style = LineStyle(self.style)


class EmptyLegendItem(LegendItem):
    pass


class LineLegendItem(EdgeInfo):
    pass


@dataclass
class ErrorLegendItem(EdgeInfo):
    color: Color
    width: float
    style: LineStyle = LineStyle.SOLID
    capsize: float = 0.0


@dataclass
class MarkersLegendItem(LegendItem):
    symbol: Symbol
    size: float
    face: FaceInfo
    edge: EdgeInfo


@dataclass
class BarLegendItem(LegendItem):
    face: FaceInfo
    edge: EdgeInfo


@dataclass
class PlotLegendItem(LegendItem):
    line: LineLegendItem
    markers: MarkersLegendItem


@dataclass
class LineErrorLegendItem(LegendItem):
    line: LineLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None


@dataclass
class MarkerErrorLegendItem(LegendItem):
    markers: MarkersLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None


@dataclass
class PlotErrorLegendItem(LegendItem):
    plot: PlotLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None


@dataclass
class StemLegendItem(LegendItem):
    line: LineLegendItem
    markers: MarkersLegendItem


@dataclass
class LegendItemCollection(LegendItem):
    items: list[tuple[str, LegendItem]]


@dataclass
class TitleItem(LegendItem):
    pass
