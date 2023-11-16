from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
from typing_extensions import ParamSpec, Concatenate

import numpy as np

from whitecanvas.backend import Backend
from whitecanvas.types import LineStyle
from whitecanvas.layers.primitive.line import Line

if TYPE_CHECKING:
    from typing import NoReturn
    from whitecanvas.canvas import Canvas


class InfCurve(Line):
    def __init__(
        self,
        model: Callable[[Concatenate[np.ndarray, ...]], np.ndarray],
        params: dict[str, Any] = {},
        bounds: tuple[float, float] = (-np.inf, np.inf),
        name: str | None = None,
        color=None,
        line_width: float = 1,
        line_style: LineStyle | str = LineStyle.SOLID,
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
        super().__init__(
            xdata,
            ydata,
            name=name,
            color=color,
            line_width=line_width,
            line_style=line_style,
            antialias=antialias,
            backend=backend,
        )
        self._bounds = bounds
        self._model = model
        self._params = params
        self._linspace_num = 256

    @property
    def data(self) -> NoReturn:
        raise NotImplementedError("Cannot get data from an InfCurve layer.")

    def set_data(self, xdata=None, ydata=None) -> NoReturn:
        raise NotImplementedError("Cannot set data to an InfCurve layer.")

    def set_params(self, **params) -> NoReturn:
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
        canvas.x.lim_changed.connect(self._recalculate_line)
        self._recalculate_line(canvas.x.lim)
        super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        canvas.x.lim_changed.disconnect(self._recalculate_line)
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
