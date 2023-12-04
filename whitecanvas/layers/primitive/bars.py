from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar
import numpy as np
from numpy.typing import NDArray

from whitecanvas.protocols import BarProtocol
from whitecanvas.layers.primitive.text import Texts
from whitecanvas.layers._mixin import (
    MultiFaceEdgeMixin,
    FaceNamespace,
    EdgeNamespace,
    ConstFace,
    ConstEdge,
    MultiFace,
    MultiEdge,
)
from whitecanvas.layers._sizehint import xyy_size_hint
from whitecanvas.backend import Backend
from whitecanvas.types import (
    FacePattern,
    ColorType,
    _Void,
    Alignment,
    LineStyle,
    Orientation,
    XYData,
    ArrayLike1D,
)
from whitecanvas.utils.normalize import as_array_1d

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg


_void = _Void()
_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)


def _norm_bar_inputs(t0, height, bot, orient: Orientation, bar_width: float):
    t0 = as_array_1d(t0)
    height = as_array_1d(height)
    if bot is None:
        bot = np.zeros_like(t0)
    bot = as_array_1d(bot)
    if not (t0.size == height.size == bot.size):
        raise ValueError(
            "Expected all arrays to have the same size, "
            f"got {t0.size}, {height.size}, {bot.size}"
        )
    y0 = height + bot
    x_hint, y_hint = xyy_size_hint(t0, y0, bot, orient, bar_width / 2)

    if orient.is_vertical:
        dx = bar_width / 2
        x0, x1 = t0 - dx, t0 + dx
        y0, y1 = bot, y0
    else:
        dy = bar_width / 2
        x0, x1 = bot, y0
        y0, y1 = t0 - dy, t0 + dy
    return (x0, x1, y0, y1), x_hint, y_hint


class Bars(MultiFaceEdgeMixin[BarProtocol, _Face, _Edge]):
    def __init__(
        self,
        x: ArrayLike1D,
        height: ArrayLike1D,
        bottom: ArrayLike1D | None = None,
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
        xxyy, xhint, yhint = _norm_bar_inputs(x, height, bottom, ori, bar_width)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), *xxyy)
        self._bar_width = bar_width
        self._orient = ori
        self.face.update(color=color, alpha=alpha, pattern=pattern)
        self._x_hint, self._y_hint = xhint, yhint

    @classmethod
    def from_histogram(
        cls,
        data: ArrayLike1D,
        *,
        bins: int | ArrayLike1D = 10,
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
    def data(self) -> XYData:
        """Current data of the layer."""
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self._orient.is_vertical:
            return XYData((x0 + x1) / 2, y1 - y0)
        else:
            return XYData((y0 + y1) / 2, x1 - x0)

    def set_data(
        self,
        xdata: ArrayLike1D | None = None,
        ydata: ArrayLike1D | None = None,
        bottom: ArrayLike1D | None = None,
    ):
        if xdata is None or ydata is None:
            data = self.data
            if xdata is None:
                xdata = data.x
            if ydata is None:
                ydata = data.y
        if bottom is None:
            bottom = self.bottom
        xxyy, xhint, yhint = _norm_bar_inputs(
            xdata, ydata, bottom, self._orient, self._bar_width
        )
        self._backend._plt_set_data(*xxyy)
        self._x_hint, self._y_hint = xhint, yhint

    @property
    def bottom(self) -> NDArray[np.floating]:
        x0, _, y0, _ = self._backend._plt_get_data()
        if self._orient.is_vertical:
            return y0
        else:
            return x0

    @bottom.setter
    def bottom(self, bot: ArrayLike1D):
        self.set_data(bottom=bot)

    @property
    def top(self) -> NDArray[np.floating]:
        _, x1, _, y1 = self._backend._plt_get_data()
        if self._orient.is_vertical:
            return y1
        else:
            return x1

    @top.setter
    def top(self, top: ArrayLike1D):
        top = as_array_1d(top)
        self.set_data(ydata=top - self.bottom)

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
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.LabeledBars:
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
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.LabeledBars:
        from whitecanvas.layers.group import LabeledBars
        from whitecanvas.layers.primitive import Errorbars

        xerr = self._create_errorbars(
            err, err_high, color=color, width=width, style=style,
            antialias=antialias, capsize=capsize, orient=Orientation.HORIZONTAL,
        )  # fmt: skip
        yerr = Errorbars(
            [], [], [], orient=Orientation.HORIZONTAL, backend=self._backend_name
        )
        return LabeledBars(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
    ) -> _lg.LabeledBars:
        from whitecanvas.layers.group import LabeledBars
        from whitecanvas.layers.primitive import Errorbars

        yerr = self._create_errorbars(
            err, err_high, color=color, width=width, style=style,
            antialias=antialias, capsize=capsize, orient=Orientation.VERTICAL,
        )  # fmt: skip
        xerr = Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name)
        return LabeledBars(self, xerr, yerr, name=self.name)

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
    ) -> _lg.LabeledBars:
        from whitecanvas.layers import Errorbars
        from whitecanvas.layers.group import LabeledBars

        if isinstance(strings, str):
            strings = [strings] * self.data.x.size
        texts = Texts(
            *self.data,
            strings,
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            fontfamily=fontfamily,
            backend=self._backend_name,
        )
        return LabeledBars(
            self,
            Errorbars.empty(Orientation.HORIZONTAL, backend=self._backend_name),
            Errorbars.empty(Orientation.VERTICAL, backend=self._backend_name),
            texts=texts,
            name=self.name,
        )

    def _create_errorbars(
        self,
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
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
        x, y = self.data
        y = y + self.bottom
        return Errorbars(
            x, y - err, y + err_high, color=color, width=width,
            style=style, antialias=antialias, capsize=capsize,
            orient=orient, backend=self._backend_name
        )  # fmt: skip

    @property
    def ndata(self) -> int:
        """The number of data points"""
        return self._backend._plt_get_data()[0].size

    def with_face(
        self,
        color: ColorType | None = None,
        pattern: FacePattern | str = FacePattern.SOLID,
        alpha: float = 1,
    ) -> Bars[ConstFace, _Edge]:
        return super().with_face(color, pattern, alpha)

    def with_face_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        pattern: str | FacePattern | Sequence[str | FacePattern] = FacePattern.SOLID,
        alpha: float = 1,
    ) -> Bars[MultiFace, _Edge]:
        return super().with_face_multi(color, pattern, alpha)

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Bars[_Face, ConstEdge]:
        return super().with_edge(color, width, style, alpha)

    def with_edge_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Bars[_Face, MultiEdge]:
        return super().with_edge_multi(color, width, style, alpha)
