from __future__ import annotations

from cmap import Color

from whitecanvas.types import Hatch, LineStyle, Symbol


def _sequence_of_column_name(keys: list[str], value) -> bool:
    if isinstance(value, str):
        return False
    if hasattr(value, "__iter__"):
        for each in value:
            if not isinstance(each, str):
                return False
            if each not in keys:
                return False
        return True
    return False


class PlotArg:
    def __init__(self, value, is_column: bool):
        self._value = value
        self._is_column = is_column

    @property
    def value(self):
        """The value of the argument."""
        return self._value

    @property
    def is_column(self) -> bool:
        """True if the value is a column name."""
        return self._is_column

    @classmethod
    def from_color(cls, keys: list[str], color) -> PlotArg:
        if color is None:
            return PlotArg(None, False)
        if isinstance(color, str):
            if color in keys:
                return PlotArg([color], True)
            else:
                return PlotArg.from_color(keys, Color(color))
        elif _sequence_of_column_name(keys, color):
            return PlotArg(list(color), True)
        else:
            try:
                col = Color(color)
            except Exception:
                raise ValueError(
                    f"'color' must be one of the column names {keys!r}, color-like "
                    "or sequence of them."
                )
            return PlotArg(col, False)

    @classmethod
    def from_symbol(cls, keys: list[str], symbol) -> PlotArg:
        if symbol is None:
            return PlotArg(None, False)
        if isinstance(symbol, str):
            if symbol in keys:
                return PlotArg([symbol], True)
            else:
                return PlotArg.from_symbol(keys, Symbol(symbol))
        elif _sequence_of_column_name(keys, symbol):
            return PlotArg(list(symbol), True)
        else:
            try:
                sym = Symbol(symbol)
            except Exception:
                raise ValueError(
                    f"'symbol' must be one of the column names {keys!r}, symbol-like "
                    "or sequence of them."
                )
            return PlotArg(sym, False)

    @classmethod
    def from_hatch(cls, keys: list[str], hatch) -> PlotArg:
        if hatch is None:
            return PlotArg(None, False)
        if isinstance(hatch, str):
            if hatch in keys:
                return PlotArg([hatch], True)
            else:
                return PlotArg.from_hatch(keys, Hatch(hatch))
        elif _sequence_of_column_name(keys, hatch):
            return PlotArg(list(hatch), True)
        else:
            try:
                htch = Hatch(hatch)
            except Exception:
                raise ValueError(
                    f"'hatch' must be one of the column names {keys!r}, hatch-like "
                    "or sequence of them."
                ) from None
            return PlotArg(htch, False)

    @classmethod
    def from_style(cls, keys: list[str], style) -> PlotArg:
        if style is None:
            return PlotArg(None, False)
        if isinstance(style, str):
            if style in keys:
                return PlotArg([style], True)
            else:
                return PlotArg.from_style(keys, LineStyle(style))
        elif _sequence_of_column_name(keys, style):
            return PlotArg(list(style), True)
        else:
            try:
                stl = LineStyle(style)
            except Exception:
                raise ValueError(
                    f"'style' must be one of the column names {keys!r}, style-like "
                    "or sequence of them."
                ) from None
            return PlotArg(stl, False)

    @classmethod
    def from_scalar(cls, keys: list[str], value) -> PlotArg:
        if value is None:
            return PlotArg(None, False)
        if isinstance(value, str):
            if value in keys:
                return PlotArg([value], True)
            else:
                raise ValueError(f"Not a valid column name: {value!r}")
        elif _sequence_of_column_name(keys, value):
            return PlotArg(list(value), True)
        else:
            return PlotArg(float(value), False)
