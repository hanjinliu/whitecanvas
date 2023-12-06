from qtpy import QtGui
from qtpy.QtCore import Qt
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
import numpy as np

from whitecanvas.types import LineStyle, Symbol, FacePattern


def array_to_qcolor(arr: np.ndarray) -> QtGui.QColor:
    """Convert an array to a QColor."""
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"Expected numeric array, got {arr.dtype}")
    rgba = [round(c) for c in (arr * 255)]
    if len(rgba) not in (3, 4):
        raise ValueError(f"Expected array of length 4, got {rgba}")
    return QtGui.QColor(*rgba)


_LINE_STYLE = {
    LineStyle.SOLID: Qt.PenStyle.SolidLine,
    LineStyle.DASH: Qt.PenStyle.DashLine,
    LineStyle.DOT: Qt.PenStyle.DotLine,
    LineStyle.DASH_DOT: Qt.PenStyle.DashDotLine,
}

_LINE_STYLE_INV = {v: k for k, v in _LINE_STYLE.items()}
_LINE_STYLE_INV[Qt.PenStyle.NoPen] = LineStyle.SOLID


def _create_symbol(coords: list[tuple[float, float]]):
    p = QtGui.QPainterPath()
    p.moveTo(*coords[0])
    for x, y in coords[1:]:
        p.lineTo(x, y)
    p.closeSubpath()
    return p


Symbols["-"] = _create_symbol([(0, -0.5), (0, 0.5)])
Symbols["|"] = _create_symbol([(-0.5, 0), (0.5, 0)])
Symbols["s"] = _create_symbol([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])
Symbols["."] = _create_symbol([(0, -0.1), (0.1, 0), (0, 0.1), (-0.1, 0)])

_SYMBOL = {
    Symbol.STAR: "star",
    Symbol.DIAMOND: "d",
    Symbol.TRIANGLE_UP: "t1",
    Symbol.TRIANGLE_LEFT: "t3",
    Symbol.TRIANGLE_DOWN: "t",
    Symbol.TRIANGLE_RIGHT: "t2",
    Symbol.VBAR: "-",
    Symbol.HBAR: "|",
    Symbol.SQUARE: "s",
}

_SYMBOL_INV = {v: k for k, v in _SYMBOL.items()}

_PAINT_STYLE = {
    FacePattern.SOLID: Qt.BrushStyle.SolidPattern,
    FacePattern.HORIZONTAL: Qt.BrushStyle.HorPattern,
    FacePattern.VERTICAL: Qt.BrushStyle.VerPattern,
    FacePattern.CROSS: Qt.BrushStyle.CrossPattern,
    FacePattern.DIAGONAL_BACK: Qt.BrushStyle.BDiagPattern,
    FacePattern.DIAGONAL_FORWARD: Qt.BrushStyle.FDiagPattern,
    FacePattern.DIAGONAL_CROSS: Qt.BrushStyle.DiagCrossPattern,
}

_PAINT_STYLE_INV = {v: k for k, v in _PAINT_STYLE.items()}
_PAINT_STYLE_INV[Qt.BrushStyle.NoBrush] = FacePattern.SOLID


def from_qt_line_style(style: Qt.PenStyle) -> LineStyle:
    return _LINE_STYLE_INV[style]


def to_qt_line_style(style: LineStyle) -> Qt.PenStyle:
    return _LINE_STYLE[style]


def from_qt_symbol(symbol: str) -> Symbol:
    out = _SYMBOL_INV.get(symbol, None)
    if out is None:
        return Symbol(symbol)
    return out


def to_qt_symbol(symbol: Symbol) -> str:
    return _SYMBOL.get(symbol, symbol.value)


def from_qt_brush_style(hatch: Qt.BrushStyle) -> FacePattern:
    return _PAINT_STYLE_INV[hatch]


def to_qt_brush_style(hatch: FacePattern) -> Qt.BrushStyle:
    # BUG: pyqtgraph does not support setting brush style correctly
    return _PAINT_STYLE[hatch]
