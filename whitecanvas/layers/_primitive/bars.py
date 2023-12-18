from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar
import numpy as np
from numpy.typing import NDArray

from psygnal import Signal
from whitecanvas.protocols import BarProtocol
from whitecanvas.layers._primitive.text import Texts
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.layers._mixin import (
    MultiFaceEdgeMixin,
    FaceEdgeMixinEvents,
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
    Hatch,
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


class BarEvents(FaceEdgeMixinEvents):
    bar_width = Signal(float)
    bottom = Signal(np.ndarray)


class Bars(
    DataBoundLayer[BarProtocol, XYData],
    MultiFaceEdgeMixin[_Face, _Edge],
):
    """
    Layer that represents vertical/hosizontal bars.

    Attributes
    ----------
    face : :class:`~whitecanvas.layers._mixin.FaceNamespace`
        Face properties of the bars.
    edge : :class:`~whitecanvas.layers._mixin.EdgeNamespace`
        Edge properties of the bars.
    """

    events: BarEvents
    _events_class = BarEvents

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
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
        MultiFaceEdgeMixin.__init__(self)
        ori = Orientation.parse(orient)
        xxyy, xhint, yhint = _norm_bar_inputs(x, height, bottom, ori, bar_width)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), *xxyy)
        self._bar_width = bar_width
        self._orient = ori
        self.face.update(color=color, alpha=alpha, hatch=hatch)
        self._x_hint, self._y_hint = xhint, yhint
        self._bar_type = "bars"

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
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
        """Construct a bar plot from a histogram."""
        data = as_array_1d(data)
        counts, edges = np.histogram(data, bins, density=density, range=range)
        centers = (edges[:-1] + edges[1:]) / 2
        if bar_width is None:
            bar_width = edges[1] - edges[0]
        self = Bars(
            centers, counts, bar_width=bar_width, name=name, color=color, alpha=alpha,
            orient=orient, hatch=hatch, backend=backend,
        )  # fmt: skip
        if density:
            self._bar_type = "histogram-density"
        else:
            self._bar_type = "histogram-count"
        return self

    def _get_layer_data(self) -> XYData:
        """Current data of the layer."""
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self._orient.is_vertical:
            return XYData((x0 + x1) / 2, y1 - y0)
        else:
            return XYData((y0 + y1) / 2, x1 - x0)

    def _norm_layer_data(self, data: Any) -> XYData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(
                    "Expected data to be a 2D array of shape (N, 2), "
                    f"got {data.shape}"
                )
            x, height = data[:, 0], data[:, 1]
        else:
            x, height = data
        if x is not None:
            x = as_array_1d(x)
        else:
            x = self.data.x
        if height is not None:
            height = as_array_1d(height)
        else:
            height = self.data.y
        return XYData(x, height)

    def _set_layer_data(self, data: XYData):
        x, height = data
        if self._orient.is_vertical:
            self._backend._plt_set_data(x, height / 2, -height / 2)
        else:
            self._backend._plt_set_data(-height / 2, height / 2, x)
        self._x_hint, self._y_hint = xyy_size_hint(x, height, height, self.orient)

    def set_data(
        self,
        xdata: ArrayLike1D | None = None,
        ydata: ArrayLike1D | None = None,
        bottom: ArrayLike1D | None = None,
    ):
        emit_data = xdata is not None or ydata is not None
        emit_bottom = bottom is not None
        if xdata is None or ydata is None:
            data = self.data
            if xdata is None:
                xdata = data.x
            if ydata is None:
                ydata = data.y
        if not emit_bottom:
            bottom = self.bottom
        xxyy, xhint, yhint = _norm_bar_inputs(
            xdata, ydata, bottom, self._orient, self._bar_width
        )
        self._backend._plt_set_data(*xxyy)
        self._x_hint, self._y_hint = xhint, yhint
        if emit_data:
            self.events.data.emit(xdata, ydata)
        if emit_bottom:
            self.events.bottom.emit(bottom)

    @property
    def bottom(self) -> NDArray[np.floating]:
        """The bottom values of the bars."""
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
        """The top values of the bars."""
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
        self.events.bar_width.emit(w)

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
        from whitecanvas.layers._primitive import Errorbars

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
        from whitecanvas.layers._primitive import Errorbars

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
            family=fontfamily,
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
        from whitecanvas.layers._primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge.color
        if width is _void:
            width = self.edge.width
        if style is _void:
            style = self.edge.style
        x, y = self.data
        y = y + self.bottom
        if orient is not self.orient:
            x, y = y, x
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
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1,
    ) -> Bars[ConstFace, _Edge]:
        return super().with_face(color, hatch, alpha)

    def with_face_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        hatch: str | Hatch | Sequence[str | Hatch] = Hatch.SOLID,
        alpha: float = 1,
    ) -> Bars[MultiFace, _Edge]:
        return super().with_face_multi(color, hatch, alpha)

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
