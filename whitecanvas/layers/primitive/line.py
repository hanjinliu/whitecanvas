from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.protocols import LineProtocol, MultiLineProtocol
from whitecanvas.layers._base import XYData, PrimitiveLayer
from whitecanvas.layers._mixin import LineMixin
from whitecanvas.backend import Backend
from whitecanvas.types import (
    LineStyle,
    Symbol,
    ColorType,
    _Void,
    Alignment,
    FacePattern,
    Orientation,
)
from whitecanvas.utils.normalize import as_array_1d, as_color_array, normalize_xy

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg

_void = _Void()


class MonoLine(LineMixin[LineProtocol]):
    @property
    def antialias(self) -> bool:
        """Whether to use antialiasing."""
        return self._backend._plt_get_antialias()

    @antialias.setter
    def antialias(self, antialias: bool):
        self._backend._plt_set_antialias(antialias)


class Line(MonoLine):
    def __init__(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.name = name if name is not None else "Line"
        self.update(color=color, width=width, style=style, antialias=antialias)
        self._x_hint = xdata.min(), xdata.max()
        self._y_hint = ydata.min(), ydata.max()

    @property
    def data(self) -> XYData:
        """Current data of the layer."""
        return XYData(*self._backend._plt_get_data())

    def set_data(
        self,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
    ):
        x0, y0 = self.data
        if xdata is not None:
            x0 = as_array_1d(xdata)
        if ydata is not None:
            y0 = as_array_1d(ydata)
        if x0.size != y0.size:
            raise ValueError(
                "Expected xdata and ydata to have the same size, "
                f"got {x0.size} and {y0.size}"
            )
        self._backend._plt_set_data(x0, y0)
        self._x_hint = x0.min(), x0.max()
        self._y_hint = y0.min(), y0.max()

    def with_markers(
        self,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        color: ColorType | _Void = _void,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.Plot:
        from whitecanvas.layers.group import Plot
        from whitecanvas.layers.primitive import Markers

        if color is _void:
            color = self.color

        markers = Markers(
            *self.data, symbol=symbol, size=size, color=color, alpha=alpha,
            pattern=pattern, backend=self._backend_name,
        )  # fmt: skip
        return Plot(self, markers, name=self.name)

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> _lg.AnnotatedLine:
        from whitecanvas.layers.group import AnnotatedLine
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        if width is _void:
            width = self.width
        if style is _void:
            style = self.style
        if antialias is _void:
            antialias = self.antialias
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars([], [], [], orient="horizontal", backend=self._backend_name)
        return AnnotatedLine(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> _lg.AnnotatedLine:
        from whitecanvas.layers.group import AnnotatedLine
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        if width is _void:
            width = self.width
        if style is _void:
            style = self.style
        if antialias is _void:
            antialias = self.antialias
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        xerr = Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name)
        return AnnotatedLine(self, xerr, yerr, name=self.name)

    def with_xband(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        alpha: float = 0.5,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.LineBand:
        from whitecanvas.layers.group import LineBand
        from whitecanvas.layers.primitive import Band

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        data = self.data
        band = Band(
            data.y, data.x - err, data.x + err_high, orient="horizontal",
            color=color, alpha=alpha, pattern=pattern, backend=self._backend_name,
        )  # fmt: skip
        return LineBand(self, band, name=self.name)

    def with_yband(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        alpha: float = 0.5,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.LineBand:
        from whitecanvas.layers.group import LineBand
        from whitecanvas.layers.primitive import Band

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        data = self.data
        band = Band(
            data.x, data.y - err, data.y + err_high, orient=Orientation.VERTICAL,
            color=color, alpha=alpha, pattern=pattern, backend=self._backend_name,
        )  # fmt: skip
        return LineBand(self, band, name=self.name)

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
    ) -> _lg.AnnotatedLine:
        from whitecanvas.layers import Errorbars
        from whitecanvas.layers.group import TextGroup, AnnotatedLine

        if isinstance(strings, str):
            strings = [strings] * self.data.x.size
        else:
            strings = list(strings)
            if len(strings) != self.data.x.size:
                raise ValueError(
                    f"Number of strings ({len(strings)}) does not match the "
                    f"number of data ({self.data.x.size})."
                )
        texts = TextGroup.from_strings(
            *self.data,
            strings,
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            fontfamily=fontfamily,
            backend=self._backend_name,
        )
        return AnnotatedLine(
            self,
            Errorbars.empty(Orientation.HORIZONTAL, backend=self._backend_name),
            Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name),
            texts=texts,
            name=self.name,
        )


class MultiLine(PrimitiveLayer[MultiLineProtocol]):
    def __init__(
        self,
        data: list[ArrayLike],
        *,
        name: str | None = None,
        color: ColorType = "blue",
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        alpha: float = 1.0,
        backend: Backend | str | None = None,
    ):
        data_normed, self._x_hint, self._y_hint = _norm_data(data)
        self._backend = self._create_backend(Backend(backend), data_normed)
        self.name = name if name is not None else type(self).__name__
        self.update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )

    @property
    def data(self) -> list[XYData]:
        """Current data of the layer."""
        return [XYData(d) for d in self._backend._plt_get_data()]

    def set_data(self, data: list[ArrayLike]):
        data, x_hint, y_hint = _norm_data(data)
        self._backend._plt_set_data(data)
        self._x_hint, self._y_hint = x_hint, y_hint

    @property
    def nlines(self) -> int:
        """Number of lines."""
        return len(self._backend._plt_get_data())

    @property
    def color(self) -> NDArray[np.floating]:
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        self._backend._plt_set_edge_color(as_color_array(color, self.nlines))

    @property
    def width(self):
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @width.setter
    def width(self, width):
        self._backend._plt_set_edge_width(width)

    @property
    def style(self) -> LineStyle:
        """Style of the line."""
        return LineStyle(self._backend._plt_get_edge_style())

    @style.setter
    def style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))

    @property
    def alpha(self) -> float:
        """Alpha value of the line."""
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if width is not _void:
            self.width = width
        if style is not _void:
            self.style = style
        if alpha is not _void:
            self.alpha = alpha
        if antialias is not _void:
            self.antialias = antialias
        return self

    @property
    def antialias(self) -> bool:
        """Whether to use antialiasing."""
        return self._backend._plt_get_antialias()

    @antialias.setter
    def antialias(self, antialias: bool):
        self._backend._plt_set_antialias(antialias)


def _norm_data(data: list[ArrayLike]) -> NDArray[np.number]:
    data_normed: list[NDArray[np.number]] = []
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for each in data:
        arr = np.asarray(each)
        if arr.dtype.kind not in "uif":
            raise ValueError(f"Expected data to be numeric, got {arr.dtype}")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected data to be (N, 2), got {arr.shape}")
        data_normed.append(arr)
        xmins.append(arr[:, 0].min())
        xmaxs.append(arr[:, 0].max())
        ymins.append(arr[:, 1].min())
        ymaxs.append(arr[:, 1].max())
    if len(data) > 0:
        xhint = min(xmins), max(xmaxs)
        yhint = min(ymins), max(ymaxs)
    else:
        xhint = yhint = None
    return data_normed, xhint, yhint
