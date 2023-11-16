from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.protocols import BarProtocol
from whitecanvas.layers._base import PrimitiveLayer, XYYData
from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, FacePattern, ColorType, _Void
from whitecanvas.utils.normalize import as_array_1d, norm_color

if TYPE_CHECKING:
    from whitecanvas.layers import group as _lg


_void = _Void()


class BarBase(PrimitiveLayer[BarProtocol]):
    @property
    def face_color(self):
        """Face color of the bar."""
        return self._backend._plt_get_face_color()

    @face_color.setter
    def face_color(self, color):
        self._backend._plt_set_face_color(norm_color(color))

    @property
    def face_pattern(self) -> FacePattern:
        """Face fill pattern of the bars."""
        return self._backend._plt_get_face_pattern()

    @face_pattern.setter
    def face_pattern(self, style: str | FacePattern):
        self._backend._plt_set_face_pattern(FacePattern(style))

    @property
    def edge_color(self):
        """Edge color of the bar."""
        return self._backend._plt_get_edge_color()

    @edge_color.setter
    def edge_color(self, color):
        self._backend._plt_set_edge_color(norm_color(color))

    @property
    def edge_width(self) -> float:
        return self._backend._plt_get_edge_width()

    @edge_width.setter
    def edge_width(self, width: float):
        self._backend._plt_set_edge_width(width)

    @property
    def edge_style(self) -> LineStyle:
        return self._backend._plt_get_edge_style()

    @edge_style.setter
    def edge_style(self, style: str | LineStyle):
        self._backend._plt_set_edge_style(LineStyle(style))


def _norm_bar_inputs(t0, e0, e1, orient: str, bar_width: float):
    t0 = as_array_1d(t0)
    e0 = as_array_1d(e0)
    if e1 is None:
        e1 = np.zeros_like(t0)
    e1 = as_array_1d(e1)
    if not (t0.size == e0.size == e1.size):
        raise ValueError("Expected all arrays to have the same size, " f"got {t0.size}, {e0.size}, {e1.size}")
    if orient == "vertical":
        dx = bar_width / 2
        x0, x1 = t0 - dx, t0 + dx
        y0, y1 = e0, e1
    elif orient == "horizontal":
        dy = bar_width / 2
        x0, x1 = e0, e1
        y0, y1 = t0 - dy, t0 + dy
    else:
        raise ValueError(f"orient must be 'vertical' or 'horizontal'")
    return x0, x1, y0, y1


class Bars(BarBase):
    def __init__(
        self,
        x: ArrayLike,
        top: ArrayLike,
        bottom: ArrayLike | None = None,
        *,
        orient: Literal["vertical", "horizontal"] = "vertical",
        bar_width: float = 0.8,
        name: str | None = None,
        face_color: ColorType = "blue",
        edge_color: ColorType = "black",
        edge_width: float = 0.0,
        edge_style: LineStyle | str = LineStyle.SOLID,
        backend: Backend | str | None = None,
    ):
        xxyy = _norm_bar_inputs(x, top, bottom, orient, bar_width)
        self._backend = self._create_backend(Backend(backend), *xxyy)
        self._bar_width = bar_width
        self.name = name if name is not None else "Bars"
        self._orient = orient
        self.setup(face_color=face_color, edge_color=edge_color, edge_width=edge_width, edge_style=edge_style)

    @property
    def data(self) -> XYYData:
        """Current data of the layer."""
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self._orient == "vertical":
            return XYYData((x0 + x1) / 2, y1, y0)
        elif self._orient == "horizontal":
            return XYYData((y0 + y1) / 2, x1, x0)
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")

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
        xxyy = _norm_bar_inputs(x, top, bottom, self._orient, self._bar_width)
        self._backend._plt_set_data(*xxyy)

    @property
    def bar_width(self) -> float:
        """Width of the bars."""
        return self._bar_width

    @bar_width.setter
    def bar_width(self, w: float):
        if w <= 0:
            raise ValueError(f"Expected width > 0, got {w}")
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self._orient == "vertical":
            dx = (w - self._bar_width) / 2
            x0 = x0 - dx
            x1 = x1 + dx
        else:
            dy = (w - self._bar_width) / 2
            y0 = y0 - dy
            y1 = y1 + dy
        self._backend._plt_set_data(x0, x1, y0, y1)

    @property
    def orient(self) -> Literal["vertical", "horizontal"]:
        """Orientation of the bars."""
        return self._orient

    def setup(
        self,
        *,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | str | _Void = _void,
    ):
        if face_color is not _void:
            self.face_color = face_color
        if edge_color is not _void:
            self.edge_color = edge_color
        if edge_width is not _void:
            self.edge_width = edge_width
        if edge_style is not _void:
            self.edge_style = edge_style
        return self

    def with_err(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.LineErrorbars:
        if self.orient == "vertical":
            return self.with_yerr(
                err, err_high, color=color, line_width=line_width,
                line_style=line_style, antialias=antialias, capsize=capsize
            )  # fmt: skip
        elif self.orient == "horizontal":
            return self.with_xerr(
                err, err_high, color=color, line_width=line_width,
                line_style=line_style, antialias=antialias, capsize=capsize
            )  # fmt: skip
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool | _Void = True,
        capsize: float = 0,
    ) -> _lg.BarErrorbars:
        from whitecanvas.layers.group import BarErrorbars
        from whitecanvas.layers.primitive import Errorbars

        xerr = self._create_errorbars(
            err, err_high, color=color, line_width=line_width, line_style=line_style,
            antialias=antialias, capsize=capsize, orient="horizontal",
        )  # fmt: skip
        yerr = Errorbars([], [], [], orient="horizontal", backend=self._backend_name)
        return BarErrorbars(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
    ) -> _lg.BarErrorbars:
        from whitecanvas.layers.group import BarErrorbars
        from whitecanvas.layers.primitive import Errorbars

        yerr = self._create_errorbars(
            err, err_high, color=color, line_width=line_width, line_style=line_style,
            antialias=antialias, capsize=capsize, orient="vertical",
        )  # fmt: skip
        xerr = Errorbars([], [], [], orient="vertical", backend=self._backend_name)
        return BarErrorbars(self, xerr, yerr, name=self.name)

    def _create_errorbars(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool = True,
        capsize: float = 0,
        orient: Literal["vertical", "horizontal"] = "vertical",
    ):
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.edge_color
        if line_width is _void:
            line_width = self.edge_width
        if line_style is _void:
            line_style = self.edge_style
        # if antialias is _void:
        #     antialias = self.antialias
        if self.orient == "vertical":
            x = self.data.x
            y = self.data.y1
        elif self.orient == "horizontal":
            x = self.data.y1
            y = self.data.x
        else:
            raise ValueError(f"orient must be 'vertical' or 'horizontal'")
        return Errorbars(
            x, y - err, y + err_high, color=color, line_width=line_width,
            line_style=line_style, antialias=antialias, capsize=capsize,
            orient=orient, backend=self._backend_name
        )  # fmt: skip


_void = _Void()


class Boxes(BarBase):
    def __init__(
        self,
        t: ArrayLike,
        edge_low: ArrayLike,
        edge_high: ArrayLike,
        *,
        orient: Literal["vertical", "horizontal"] = "vertical",
        bar_width: float = 0.8,
        name: str | None = None,
        face_color: ColorType = "blue",
        edge_color: ColorType = "black",
        edge_width: float = 0.0,
        edge_style: LineStyle | str = LineStyle.SOLID,
        backend: Backend | str | None = None,
    ):
        x0, x1, y0, y1 = _norm_bar_inputs(t, edge_low, edge_high, orient, bar_width)
        self._backend = self._create_backend(Backend(backend), x0, x1, y0, y1)
        self._orient = orient
        self.name = name if name is not None else "Boxes"
        self._bar_width = bar_width
        self.face_color = face_color
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.edge_style = edge_style

    @property
    def bar_width(self) -> float:
        """Width of the bars."""
        return self._bar_width

    @bar_width.setter
    def bar_width(self, w: float):
        if w <= 0:
            raise ValueError(f"Expected width > 0, got {w}")
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self._orient == "vertical":
            dx = (w - self._bar_width) / 2
            x0 = x0 - dx
            x1 = x1 + dx
        else:
            dy = (w - self._bar_width) / 2
            y0 = y0 - dy
            y1 = y1 + dy
        self._backend._plt_set_data(x0, x1, y0, y1)

    @property
    def orient(self) -> Literal["vertical", "horizontal"]:
        """Orientation of the bars."""
        return self._orient
