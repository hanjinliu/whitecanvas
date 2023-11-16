from __future__ import annotations

from whitecanvas.types import ColorType, _Void, Symbol, LineStyle
from whitecanvas.layers.primitive import Line, Markers, Bars, Errorbars
from whitecanvas.layers._base import LayerGroup, XYData


_void = _Void()


class _WithErrorbars(LayerGroup):
    def _set_data(self, xdata=None, ydata=None):
        self._children[0].set_data(xdata, ydata)

    @property
    def xerr(self) -> Errorbars:
        return self._children[1]

    @property
    def yerr(self) -> Errorbars:
        return self._children[2]

    @property
    def data(self) -> XYData:
        return self._children[0].data

    def set_data(self, xdata=None, ydata=None):
        data = self.data
        if xdata is None:
            dx = 0
        else:
            dx = xdata - data.x
        if ydata is None:
            dy = 0
        else:
            dy = ydata - data.y
        self._set_data(xdata, ydata)
        if self.xerr.ndata > 0:
            y, x0, x1 = self.xerr.data
            self.xerr.set_data(y + dy, x0 + dx, x1 + dx)
        if self.yerr.ndata > 0:
            x, y0, y1 = self.yerr.data
            self.yerr.set_data(x + dx, y0 + dy, y1 + dy)

    def setup_xerr(
        self,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ):
        self.xerr.setup(color=color, line_width=width, line_style=style, antialias=antialias, capsize=capsize)
        return self

    def setup_yerr(
        self,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ):
        self.yerr.setup(color=color, line_width=width, line_style=style, antialias=antialias, capsize=capsize)
        return self

    def with_xerr(
        self,
        len_lower: float,
        len_higher: float | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ):
        """
        Set the x error bar data.

        Parameters
        ----------
        len_lower : float
            Length of lower error.
        len_higher : float, optional
            Length of higher error. If not given, set to the same as `len_lower`.
        """
        if len_higher is None:
            len_higher = len_lower
        x, y = self.data
        self.xerr.set_data(y, x - len_lower, x + len_higher)
        return self.setup_xerr(color=color, width=width, style=style, antialias=antialias, capsize=capsize)

    def with_yerr(
        self,
        len_lower: float,
        len_higher: float | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ):
        """
        Set the y error bar data.

        Parameters
        ----------
        len_lower : float
            Length of lower error.
        len_higher : float, optional
            Length of higher error. If not given, set to the same as `len_lower`.
        """
        if len_higher is None:
            len_higher = len_lower
        x, y = self.data
        self.yerr.set_data(x, y - len_lower, y + len_higher)
        return self.setup_yerr(color=color, width=width, style=style, antialias=antialias, capsize=capsize)


class LineErrorbars(_WithErrorbars):
    def __init__(
        self,
        line: Line,
        xerr: Errorbars,
        yerr: Errorbars,
        name: str | None = None,
    ):
        super().__init__([line, xerr, yerr], name=name)

    @property
    def line(self) -> Line:
        """The line layer."""
        return self._children[0]

    def setup_line(
        self,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        self.line.setup(color=color, line_width=width, style=style, antialias=antialias)
        return self


class MarkerErrorbars(_WithErrorbars):
    def __init__(
        self,
        markers: Markers,
        xerr: Errorbars,
        yerr: Errorbars,
        name: str | None = None,
    ):
        super().__init__([markers, xerr, yerr], name=name)

    @property
    def markers(self) -> Markers:
        return self._children[0]

    def setup_markers(
        self,
        *,
        symbol: Symbol | str | _Void = _void,
        size: float | _Void = _void,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | str | _Void = _void,
    ):
        self.markers.setup(
            symbol=symbol,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=edge_width,
            edge_style=edge_style,
        )
        return self


class BarErrorbars(_WithErrorbars):
    def __init__(
        self,
        bars: Bars,
        xerr: Errorbars,
        yerr: Errorbars,
        name: str | None = None,
    ):
        super().__init__([bars, xerr, yerr], name=name)

    @property
    def bars(self) -> Bars:
        """The bars layer."""
        return self._children[0]

    @property
    def data(self) -> XYData:
        x, top, _ = self.bars.data
        return XYData(x, top)

    def _set_data(self, xdata=None, ydata=None):
        _, _, bottom = self.bars.data
        self.bars.set_data(xdata, ydata, bottom)

    def setup_bars(
        self,
        *,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | str | _Void = _void,
    ):
        self.bars.setup(face_color=face_color, edge_color=edge_color, edge_width=edge_width, edge_style=edge_style)
        return self
