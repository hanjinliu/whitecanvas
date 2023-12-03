from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
from typing_extensions import Concatenate

import math
import numpy as np

from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle, ColorType, Rect
from whitecanvas.layers.primitive.line import MonoLine

if TYPE_CHECKING:
    from typing import NoReturn
    from whitecanvas.canvas import Canvas


class InfCurve(MonoLine):
    def __init__(
        self,
        model: Callable[[Concatenate[np.ndarray, ...]], np.ndarray],
        params: dict[str, Any] = {},
        *,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float = 1,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        _lower, _upper = bounds
        if not np.isfinite(_lower):
            _lower = 0
        if not np.isfinite(_upper):
            _upper = 1
        params = params.copy()
        xdata = np.array([_lower, _upper])
        ydata = model(xdata, **params)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.update(
            color=color, width=width, style=style, alpha=alpha,
            antialias=antialias
        )  # fmt: skip

        self._bounds = bounds
        self._model = model
        self._params = params
        self._linspace_num = 256

    @property
    def data(self) -> NoReturn:
        raise NotImplementedError("Cannot get data from an InfCurve layer.")

    def set_data(self, xdata=None, ydata=None) -> NoReturn:
        raise NotImplementedError("Cannot set data to an InfCurve layer.")

    def set_params(self, **params):
        xdata, _ = self._backend._plt_get_data()
        params = params.copy()
        ydata = self._model(xdata, **self._params)
        self._backend._plt_set_data(xdata, ydata)
        self._params = params

    @property
    def model(self) -> Callable[[np.ndarray], np.ndarray]:
        """The model function of the layer."""
        return self._model

    def _connect_canvas(self, canvas: Canvas):
        canvas.x.events.lim.connect(self._recalculate_line)
        self._recalculate_line(canvas.x.lim)
        super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        canvas.x.events.lim.disconnect(self._recalculate_line)
        super()._disconnect_canvas(canvas)

    def _recalculate_line(self, lim: tuple[float, float]) -> None:
        x0, x1 = lim
        b0, b1 = self._bounds
        x0 = max(x0, b0)
        x1 = min(x1, b1)
        if x0 >= x1:
            xdata, ydata = np.array([]), np.array([])
        else:
            xdata = np.linspace(x0, x1, self._linspace_num)
            ydata = self._model(xdata, **self._params)
        self._backend._plt_set_data(xdata, ydata)


class InfLine(MonoLine):
    def __init__(
        self,
        pos: tuple[float, float] = (0, 0),
        angle: float = 0.0,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float = 1,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        self._is_vline = angle % 180 == 90
        if self._is_vline:
            self._tan = 0  # not used
            self._intercept = pos[0]
        else:
            _radian = math.radians(angle) % (2 * math.pi)
            self._tan = math.tan(_radian)
            self._intercept = pos[1] - self._tan * pos[0]
        self._pos = pos
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), np.zeros(1), np.zeros(1))
        self.update(
            color=color, width=width, style=style, alpha=alpha,
            antialias=antialias,
        )  # fmt: skip
        self._last_rect = Rect(0, 0, 0, 0)

    @property
    def data(self) -> NoReturn:
        raise NotImplementedError("Cannot get data from an Line layer.")

    def set_data(self, xdata=None, ydata=None) -> NoReturn:
        raise NotImplementedError("Cannot set data to an Line layer.")

    @property
    def pos(self) -> tuple[float, float]:
        """One of the points on the line."""
        return self._pos

    @pos.setter
    def pos(self, pos: tuple[float, float]):
        self._pos = pos
        self._recalculate_line(self._last_rect)

    def _connect_canvas(self, canvas: Canvas):
        canvas.events.lims.connect(self._recalculate_line)
        self._recalculate_line(canvas.lims)
        self._last_rect = canvas.lims
        super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        canvas.events.lims.disconnect(self._recalculate_line)
        super()._disconnect_canvas(canvas)

    def _recalculate_line(self, rect: Rect) -> None:
        x0, x1, y0, y1 = rect
        self._last_rect = rect
        if self._is_vline:
            x = self._intercept
            xdata = np.array([x, x])
            ydata = np.array([y0, y1])
        else:
            xdata = np.array([x0, x1])
            ydata = self._tan * xdata + self._intercept
        self._backend._plt_set_data(xdata, ydata)
