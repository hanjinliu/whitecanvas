from __future__ import annotations

from typing import TYPE_CHECKING
from numpy.typing import ArrayLike

from whitecanvas.protocols import LineProtocol
from whitecanvas.layers._base import XYDataLayer, XYData, PrimitiveLayer
from whitecanvas.layers._mixin import LineMixin
from whitecanvas.backend import Backend
from whitecanvas.types import (
    LineStyle,
    Symbol,
    ColorType,
    _Void,
    Alignment,
    FacePattern,
)
from whitecanvas.utils.normalize import as_array_1d, normalize_xy

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg

_void = _Void()


class MonoLine(LineMixin[LineProtocol], PrimitiveLayer[LineProtocol]):
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
        color: ColorType = "vlue",
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.name = name if name is not None else "Line"
        self.setup(color=color, width=width, style=style, antialias=antialias)

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
        xerr = Errorbars([], [], [], orient="vertical", backend=self._backend_name)
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
            data.x, data.y - err, data.y + err_high, orient="vertical",
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
            Errorbars([], [], [], orient="horizontal", backend=self._backend_name),
            Errorbars([], [], [], orient="vertical", backend=self._backend_name),
            texts=texts,
            name=self.name,
        )
