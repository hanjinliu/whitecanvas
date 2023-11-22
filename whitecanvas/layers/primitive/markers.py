from __future__ import annotations

from typing import TYPE_CHECKING
from numpy.typing import ArrayLike

from whitecanvas.protocols import MarkersProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYData
from whitecanvas.layers._mixin import FaceEdgeMixin
from whitecanvas.backend import Backend
from whitecanvas.types import (
    Symbol,
    LineStyle,
    ColorType,
    FacePattern,
    _Void,
    Alignment,
)
from whitecanvas.utils.normalize import as_array_1d, norm_color, normalize_xy

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg

_void = _Void()


class Markers(FaceEdgeMixin[MarkersProtocol]):
    def __init__(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        color: ColorType = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.name = name if name is not None else "Line"
        self.update(
            symbol=symbol, size=size, color=color, pattern=pattern, alpha=alpha
        )  # fmt: skip
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

    @property
    def symbol(self) -> Symbol:
        """Symbol used to mark the data points."""
        return self._backend._plt_get_symbol()

    @symbol.setter
    def symbol(self, symbol: str | Symbol):
        self._backend._plt_set_symbol(Symbol(symbol))

    @property
    def size(self) -> float:
        """Size of the symbol."""
        return self._backend._plt_get_symbol_size()

    @size.setter
    def size(self, size: float):
        self._backend._plt_set_symbol_size(size)

    def update(
        self,
        *,
        symbol: Symbol | str | _Void = _void,
        size: float | _Void = _void,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        pattern: str | FacePattern | _Void = _void,
    ):
        if symbol is not _void:
            self.symbol = symbol
        if size is not _void:
            self.size = size
        if color is not _void:
            self.face.color = color
        if pattern is not _void:
            self.face.pattern = pattern
        if alpha is not _void:
            self.face.alpha = alpha
        return self

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.AnnotatedMarkers:
        from whitecanvas.layers.group import AnnotatedMarkers
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge.color
        if width is _void:
            width = self.edge.width
        if style is _void:
            style = self.edge.style
        # if antialias is _void:
        #     antialias = self.antialias
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars([], [], [], orient="vertical", backend=self._backend_name)
        return AnnotatedMarkers(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
    ) -> _lg.AnnotatedMarkers:
        from whitecanvas.layers.group import AnnotatedMarkers
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge.color
        if width is _void:
            width = self.edge.width
        if style is _void:
            style = self.edge.style
        # if antialias is _void:
        #     antialias = self.antialias
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        xerr = Errorbars([], [], [], orient="horizontal", backend=self._backend_name)
        return AnnotatedMarkers(self, xerr, yerr, name=self.name)

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
    ) -> _lg.AnnotatedMarkers:
        from whitecanvas.layers import Errorbars
        from whitecanvas.layers.group import TextGroup, AnnotatedMarkers

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
            *self.data, strings, color=color, size=size, rotation=rotation,
            anchor=anchor, fontfamily=fontfamily, backend=self._backend_name,
        )  # fmt: skip
        return AnnotatedMarkers(
            self,
            Errorbars([], [], [], orient="horizontal", backend=self._backend_name),
            Errorbars([], [], [], orient="vertical", backend=self._backend_name),
            texts=texts,
            name=self.name,
        )
