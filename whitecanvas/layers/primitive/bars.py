from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import BarProtocol
from whitecanvas.layers._base import XYYData
from whitecanvas.layers._mixin import AggFaceEdgeMixin
from whitecanvas.layers._sizehint import xyy_size_hint
from whitecanvas.backend import Backend
from whitecanvas.types import FacePattern, ColorType, _Void, Alignment, Orientation
from whitecanvas.utils.normalize import as_array_1d

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg


_void = _Void()


def _norm_bar_inputs(t0, top, bot, orient: Orientation, bar_width: float):
    t0 = as_array_1d(t0)
    top = as_array_1d(top)
    if bot is None:
        bot = np.zeros_like(t0)
    bot = as_array_1d(bot)
    if not (t0.size == top.size == bot.size):
        raise ValueError(
            "Expected all arrays to have the same size, "
            f"got {t0.size}, {top.size}, {bot.size}"
        )
    x_hint, y_hint = xyy_size_hint(t0, top, bot, orient, bar_width / 2)

    if orient is Orientation.VERTICAL:
        dx = bar_width / 2
        x0, x1 = t0 - dx, t0 + dx
        y0, y1 = top, bot
    else:
        dy = bar_width / 2
        x0, x1 = top, bot
        y0, y1 = t0 - dy, t0 + dy
    return (x0, x1, y0, y1), x_hint, y_hint


class Bars(AggFaceEdgeMixin[BarProtocol]):
    def __init__(
        self,
        x: ArrayLike,
        top: ArrayLike,
        bottom: ArrayLike | None = None,
        *,
        orient: str | Orientation = Orientation.VERTICAL,
        bar_width: float = 0.8,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: Backend | str | None = None,
    ):
        ori = Orientation.parse(orient)
        xxyy, xhint, yhint = _norm_bar_inputs(x, top, bottom, ori, bar_width)
        self._backend = self._create_backend(Backend(backend), *xxyy)
        self._bar_width = bar_width
        self.name = name if name is not None else "Bars"
        self._orient = ori
        self.face.update(color=color, alpha=alpha, pattern=pattern)
        self._x_hint, self._y_hint = xhint, yhint

    @classmethod
    def from_histogram(
        cls,
        data: ArrayLike,
        *,
        bins: int | ArrayLike = 10,
        density: bool = False,
        range: tuple[float, float] | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        bar_width: float | None = None,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: Backend | str | None = None,
    ):
        """Construct a bar plot from a histogram."""
        data = as_array_1d(data)
        counts, edges = np.histogram(data, bins, density=density, range=range)
        centers = (edges[:-1] + edges[1:]) / 2
        if bar_width is None:
            bar_width = edges[1] - edges[0]
        return Bars(
            centers, counts, bar_width=bar_width, name=name, color=color, alpha=alpha,
            orient=orient, pattern=pattern, backend=backend,
        )  # fmt: skip

    @property
    def data(self) -> XYYData:
        """Current data of the layer."""
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self._orient.is_vertical:
            return XYYData((x0 + x1) / 2, y1, y0)
        else:
            return XYYData((y0 + y1) / 2, x1, x0)

    def set_data(
        self,
        x: ArrayLike | None = None,
        top: ArrayLike | None = None,
        bottom: ArrayLike | None = None,
    ):
        data = self.data
        if x is None:
            x = data.x
        if top is None:
            top = data.y1
        if bottom is None:
            bottom = data.y0
        xxyy, xhint, yhint = _norm_bar_inputs(
            x, top, bottom, self._orient, self._bar_width
        )
        self._backend._plt_set_data(*xxyy)
        self._x_hint, self._y_hint = xhint, yhint

    @property
    def bar_width(self) -> float:
        """Width of the bars."""
        return self._bar_width

    @bar_width.setter
    def bar_width(self, w: float):
        if w <= 0:
            raise ValueError(f"Expected width > 0, got {w}")
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self._orient is Orientation.VERTICAL:
            dx = (w - self._bar_width) / 2
            x0 = x0 - dx
            x1 = x1 + dx
        else:
            dy = (w - self._bar_width) / 2
            y0 = y0 - dy
            y1 = y1 + dy
        self._backend._plt_set_data(x0, x1, y0, y1)
        self._bar_width = w

    @property
    def orient(self) -> Orientation:
        """Orientation of the bars."""
        return self._orient

    def with_err(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.AnnotatedLine:
        if self.orient is Orientation.VERTICAL:
            return self.with_yerr(
                err, err_high, color=color, width=width,
                style=style, antialias=antialias, capsize=capsize
            )  # fmt: skip
        else:
            return self.with_xerr(
                err, err_high, color=color, width=width,
                style=style, antialias=antialias, capsize=capsize
            )  # fmt: skip

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.AnnotatedBars:
        from whitecanvas.layers.group import AnnotatedBars
        from whitecanvas.layers.primitive import Errorbars

        xerr = self._create_errorbars(
            err, err_high, color=color, width=width, style=style,
            antialias=antialias, capsize=capsize, orient=Orientation.HORIZONTAL,
        )  # fmt: skip
        yerr = Errorbars(
            [], [], [], orient=Orientation.HORIZONTAL, backend=self._backend_name
        )
        return AnnotatedBars(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
    ) -> _lg.AnnotatedBars:
        from whitecanvas.layers.group import AnnotatedBars
        from whitecanvas.layers.primitive import Errorbars

        yerr = self._create_errorbars(
            err, err_high, color=color, width=width, style=style,
            antialias=antialias, capsize=capsize, orient=Orientation.VERTICAL,
        )  # fmt: skip
        xerr = Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name)
        return AnnotatedBars(self, xerr, yerr, name=self.name)

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
    ) -> _lg.AnnotatedBars:
        from whitecanvas.layers import Errorbars
        from whitecanvas.layers.group import TextGroup, AnnotatedBars

        if isinstance(strings, str):
            strings = [strings] * self.data.x.size
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
        return AnnotatedBars(
            self,
            Errorbars.empty(Orientation.HORIZONTAL, backend=self._backend_name),
            Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name),
            texts=texts,
            name=self.name,
        )

    def _create_errorbars(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
        orient: str | Orientation = Orientation.VERTICAL,
    ):
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
        if self.orient is Orientation.VERTICAL:
            x = self.data.x
            y = self.data.y1
        else:
            x = self.data.y1
            y = self.data.x
        return Errorbars(
            x, y - err, y + err_high, color=color, width=width,
            style=style, antialias=antialias, capsize=capsize,
            orient=orient, backend=self._backend_name
        )  # fmt: skip
