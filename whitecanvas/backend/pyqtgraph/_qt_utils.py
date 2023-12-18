from qtpy import QtGui
from qtpy.QtCore import Qt
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
import numpy as np

from whitecanvas.types import LineStyle, Symbol, Hatch, MouseButton, Modifier


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
    Hatch.SOLID: Qt.BrushStyle.SolidPattern,
    Hatch.HORIZONTAL: Qt.BrushStyle.HorPattern,
    Hatch.VERTICAL: Qt.BrushStyle.VerPattern,
    Hatch.CROSS: Qt.BrushStyle.CrossPattern,
    Hatch.DIAGONAL_BACK: Qt.BrushStyle.BDiagPattern,
    Hatch.DIAGONAL_FORWARD: Qt.BrushStyle.FDiagPattern,
    Hatch.DIAGONAL_CROSS: Qt.BrushStyle.DiagCrossPattern,
}

_PAINT_STYLE_INV = {v: k for k, v in _PAINT_STYLE.items()}
_PAINT_STYLE_INV[Qt.BrushStyle.NoBrush] = Hatch.SOLID

_QT_MODIFIERS_MAP = {
    Qt.KeyboardModifier.NoModifier: (),
    Qt.KeyboardModifier.ShiftModifier: (Modifier.SHIFT,),
    Qt.KeyboardModifier.ControlModifier: (Modifier.CTRL,),
    Qt.KeyboardModifier.AltModifier: (Modifier.ALT,),
    Qt.KeyboardModifier.MetaModifier: (Modifier.META,),
    Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.ControlModifier: (Modifier.SHIFT, Modifier.CTRL),
    Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.AltModifier: (Modifier.SHIFT, Modifier.ALT),
    Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.MetaModifier: (Modifier.SHIFT, Modifier.META),
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.AltModifier: (Modifier.CTRL, Modifier.ALT),
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.MetaModifier: (Modifier.CTRL, Modifier.META),
    Qt.KeyboardModifier.AltModifier
    | Qt.KeyboardModifier.MetaModifier: (Modifier.ALT, Modifier.META),
}
_QT_BUTTON_MAP = {
    Qt.MouseButton.NoButton: MouseButton.NONE,
    Qt.MouseButton.LeftButton: MouseButton.LEFT,
    Qt.MouseButton.RightButton: MouseButton.RIGHT,
    Qt.MouseButton.MiddleButton: MouseButton.MIDDLE,
    Qt.MouseButton.BackButton: MouseButton.BACK,
    Qt.MouseButton.ForwardButton: MouseButton.FORWARD,
}


def from_qt_modifiers(qt_modifiers: Qt.KeyboardModifier) -> tuple[Modifier, ...]:
    if (modifiers := _QT_MODIFIERS_MAP.get(qt_modifiers, None)) is None:
        # NOTE: some OS have default modifiers
        _lst = []
        if Qt.KeyboardModifier.ShiftModifier & qt_modifiers:
            _lst.append(Modifier.SHIFT)
        if Qt.KeyboardModifier.ControlModifier & qt_modifiers:
            _lst.append(Modifier.CTRL)
        if Qt.KeyboardModifier.AltModifier & qt_modifiers:
            _lst.append(Modifier.ALT)
        if Qt.KeyboardModifier.MetaModifier & qt_modifiers:
            _lst.append(Modifier.META)
        modifiers = tuple(_lst)
        _QT_MODIFIERS_MAP[qt_modifiers] = modifiers
    return modifiers


def from_qt_button(qt_button: Qt.MouseButton) -> MouseButton:
    return _QT_BUTTON_MAP.get(qt_button, MouseButton.NONE)


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


def from_qt_brush_style(hatch: Qt.BrushStyle) -> Hatch:
    return _PAINT_STYLE_INV[hatch]


def to_qt_brush_style(hatch: Hatch) -> Qt.BrushStyle:
    # BUG: pyqtgraph does not support setting brush style correctly
    return _PAINT_STYLE[hatch]
