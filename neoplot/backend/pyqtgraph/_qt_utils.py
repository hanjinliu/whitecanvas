from qtpy import QtGui
from qtpy.QtCore import Qt
import numpy as np

from neoplot.types import LineStyle


def array_to_qcolor(arr: np.ndarray) -> QtGui.QColor:
    """Convert an array to a QColor."""
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"Expected numeric array, got {arr.dtype}")
    rgba = [round(c) for c in (arr * 255)]
    if len(rgba) != 4:
        raise ValueError(f"Expected array of length 4, got {rgba}")
    return QtGui.QColor(*rgba)


_LINE_STYLE = {
    LineStyle.SOLID: Qt.PenStyle.SolidLine,
    LineStyle.DASH: Qt.PenStyle.DashLine,
    LineStyle.DOT: Qt.PenStyle.DotLine,
    LineStyle.DASH_DOT: Qt.PenStyle.DashDotLine,
}

_LINE_STYLE_INV = {v: k for k, v in _LINE_STYLE.items()}


def from_qt_line_style(style: Qt.PenStyle) -> LineStyle:
    return _LINE_STYLE_INV[style]


def to_qt_line_style(style: LineStyle) -> Qt.PenStyle:
    return _LINE_STYLE[style]
