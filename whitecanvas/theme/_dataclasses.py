from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Font:
    """Font of texts."""

    family: str = "Arial"
    size: int = 11
    color: str = "#FFFFFF"


@dataclass
class Line:
    """Line style."""

    width: float = 1.0
    style: str = "-"


@dataclass
class Markers:
    """Markers of points."""

    size: int = 12
    hatch: str = ""
    symbol: str = "o"


@dataclass
class Bars:
    """Bar style."""

    extent: float = 0.8
    hatch: str = ""


@dataclass
class ErrorBars:
    """Error bar style."""

    width: float = 1.0
    style: str = "-"


@dataclass
class Theme:
    """Plot theme."""

    font: Font = field(default_factory=Font)
    line: Line = field(default_factory=Line)
    markers: Markers = field(default_factory=Markers)
    bars: Bars = field(default_factory=Bars)
    errorbars: ErrorBars = field(default_factory=ErrorBars)
    foreground_color: str = "#000000"
    background_color: str = "#FFFFFF"
    canvas_size: tuple[float, float] = (800, 600)
    palette: str = "tab10"
