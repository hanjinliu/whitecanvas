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


class LineLegendItem(EdgeInfo, LegendItem):
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

    def __post_init__(self):
        if not isinstance(self.line, LineLegendItem):
            raise ValueError(f"line got {type(self.line)}")
        if not isinstance(self.markers, MarkersLegendItem):
            raise ValueError(f"markers got {type(self.markers)}")


@dataclass
class LineErrorLegendItem(LegendItem):
    line: LineLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None

    def __post_init__(self):
        if not isinstance(self.line, LineLegendItem):
            raise ValueError(f"line got {type(self.line)}")
        if self.xerr is not None and not isinstance(self.xerr, ErrorLegendItem):
            raise ValueError(f"xerr got {type(self.xerr)}")
        if self.yerr is not None and not isinstance(self.yerr, ErrorLegendItem):
            raise ValueError(f"yerr got {type(self.yerr)}")


@dataclass
class MarkerErrorLegendItem(LegendItem):
    markers: MarkersLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None

    def __post_init__(self):
        if not isinstance(self.markers, MarkersLegendItem):
            raise ValueError(f"markers got {type(self.markers)}")
        if self.xerr is not None and not isinstance(self.xerr, ErrorLegendItem):
            raise ValueError(f"xerr got {type(self.xerr)}")
        if self.yerr is not None and not isinstance(self.yerr, ErrorLegendItem):
            raise ValueError(f"yerr got {type(self.yerr)}")


@dataclass
class PlotErrorLegendItem(LegendItem):
    plot: PlotLegendItem
    xerr: ErrorLegendItem | None = None
    yerr: ErrorLegendItem | None = None

    def __post_init__(self):
        if not isinstance(self.plot, PlotLegendItem):
            raise ValueError(f"plot got {type(self.plot)}")
        if self.xerr is not None and not isinstance(self.xerr, ErrorLegendItem):
            raise ValueError(f"xerr got {type(self.xerr)}")
        if self.yerr is not None and not isinstance(self.yerr, ErrorLegendItem):
            raise ValueError(f"yerr got {type(self.yerr)}")


@dataclass
class StemLegendItem(LegendItem):
    line: LineLegendItem
    markers: MarkersLegendItem

    def __post_init__(self):
        if not isinstance(self.line, LineLegendItem):
            raise ValueError(f"line got {type(self.line)}")
        if not isinstance(self.markers, MarkersLegendItem):
            raise ValueError(f"markers got {type(self.markers)}")


@dataclass
class LegendItemCollection(LegendItem):
    items: list[tuple[str, LegendItem]]


@dataclass
class TitleItem(LegendItem):
    pass
