from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any, Callable, Generic

import numpy as np
from typing_extensions import Concatenate, ParamSpec

from whitecanvas.backend import Backend
from whitecanvas.layers._primitive.line import LineMixin
from whitecanvas.protocols import LineProtocol
from whitecanvas.types import ColorType, LineStyle, Rect, _Void

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas import Canvas

_P = ParamSpec("_P")
_void = _Void()


class InfCurve(LineMixin[LineProtocol], Generic[_P]):
    _backend_class_name = "MonoLine"
    _NO_PADDING_NEEDED = True

    def __init__(
        self,
        model: Callable[[Concatenate[np.ndarray, _P]], np.ndarray],
        *,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float | _Void = _void,
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
        super().__init__(name=name)

        xdata = np.array([_lower, _upper])
        ydata, self._y_hint, self._params_ready = _try_init_ydata(model, xdata)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.update(
            color=color, width=width, style=style, alpha=alpha,
            antialias=antialias
        )  # fmt: skip

        self._bounds = bounds
        self._model = model
        self._args = ()
        self._kwargs = {}
        self._linspace_num = 256

    def update_params(self, *args: _P.args, **kwargs: _P.kwargs) -> Self:
        """Set the parameters of the model function."""
        xdata, _ = self._backend._plt_get_data()
        ydata = self._model(xdata, *args, **kwargs)
        self._backend._plt_set_data(xdata, ydata)
        self._args, self._kwargs = args, kwargs
        if not self._params_ready:
            self._params_ready = True
            if canvas := self._canvas_ref():
                canvas.x.events.lim.emit(canvas.x.lim)
        return self

    @property
    def params(self) -> tuple[tuple, dict[str, Any]]:
        """The parameters of the model function."""
        return self._args, self._kwargs

    @property
    def model(self) -> Callable[[Concatenate[np.ndarray, _P]], np.ndarray]:
        """The model function of the layer."""
        return self._model

    def calculate(self, xdata: np.ndarray) -> np.ndarray:
        return self._model(xdata, *self._args, **self._kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a Line from a dictionary."""
        args, kwargs = d["params"]
        return cls(
            d["model"], bounds=d["bounds"], name=d["name"], color=d["color"],
            width=d["width"], style=d["style"],
            antialias=d["antialias"], backend=backend,
        ).update_params(*args, **kwargs)  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "model": self.model,
            "bounds": self._bounds,
            "params": self.params,
            "name": self.name,
            "visible": self.visible,
            "color": self.color,
            "width": self.width,
            "style": self.style,
            "antialias": self.antialias,
        }

    def with_hover_text(self, text: str) -> Self:
        if not isinstance(text, str):
            raise TypeError(f"Hover text must be str, got {type(text)}.")
        self._backend._plt_set_hover_text([text] * self._linspace_num)
        return self

    def _connect_canvas(self, canvas: Canvas):
        canvas.x.events.lim.connect(self._recalculate_line, max_args=1)
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
            if self._params_ready:
                ydata = self._model(xdata, *self._args, **self._kwargs)
            else:
                return
        self._backend._plt_set_data(xdata, ydata)


def _try_init_ydata(
    model: Callable[[Concatenate[np.ndarray, _P]], np.ndarray],
    xdata: np.ndarray,
):
    try:
        _sig = inspect.signature(model)
    except ValueError:
        return _try_init_ydata_for_ufunc(model, xdata)
    try:
        _sig.bind(xdata)
    except TypeError:
        ydata = np.zeros_like(xdata)
        yhint = None
        ready = False
    else:
        ydata = model(xdata)
        yhint = ydata.min(), ydata.max()
        ready = True
    return ydata, yhint, ready


def _try_init_ydata_for_ufunc(
    model: Callable[[Concatenate[np.ndarray, _P]], np.ndarray],
    xdata: np.ndarray,
):
    try:
        ydata = model(xdata)
    except TypeError:
        ydata = np.zeros_like(xdata)
        yhint = None
        ready = False
    else:
        yhint = ydata.min(), ydata.max()
        ready = True
    return ydata, yhint, ready


class InfLine(LineMixin[LineProtocol]):
    _backend_class_name = "MonoLine"
    _NO_PADDING_NEEDED = True

    def __init__(
        self,
        pos: tuple[float, float] = (0, 0),
        angle: float = 0.0,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float | _Void = _void,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        self._pos = pos
        self._is_vline = angle % 180 == 90
        if self._is_vline:
            self._tan = 0  # not used
            self._intercept = self._pos[0]
        else:
            _radian = math.radians(angle) % (2 * math.pi)
            self._tan = math.tan(_radian)
            self._intercept = self._pos[1] - self._tan * self._pos[0]
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), np.zeros(1), np.zeros(1))
        self.update(
            color=color, width=width, style=style, alpha=alpha,
            antialias=antialias,
        )  # fmt: skip
        self._last_rect = Rect(0, 0, 0, 0)

    @property
    def pos(self) -> tuple[float, float]:
        """One of the points on the line."""
        return self._pos

    @pos.setter
    def pos(self, pos: tuple[float, float]):
        self._pos = pos
        self._recalculate_line(self._last_rect)

    @property
    def angle(self) -> float:
        """The angle of the line in degrees."""
        if self._is_vline:
            return 90.0
        else:
            return math.degrees(math.atan(self._tan))

    @angle.setter
    def angle(self, angle: float):
        self._is_vline = angle % 180 == 90
        if self._is_vline:
            self._tan = 0  # not used
            self._intercept = self._pos[0]
        else:
            _radian = math.radians(angle) % (2 * math.pi)
            self._tan = math.tan(_radian)
            self._intercept = self._pos[1] - self._tan * self._pos[0]
        self._recalculate_line(self._last_rect)

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a Line from a dictionary."""
        return cls(
            d["pos"], d["angle"], name=d["name"], color=d["color"], width=d["width"],
            style=d["style"], antialias=d["antialias"], backend=backend,
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "pos": self.pos,
            "angle": self.angle,
            "name": self.name,
            "visible": self.visible,
            "color": self.color,
            "width": self.width,
            "style": self.style,
            "antialias": self.antialias,
        }

    def with_hover_text(self, text: str) -> Self:
        if not isinstance(text, str):
            raise TypeError(f"Hover text must be str, got {type(text)}.")
        self._backend._plt_set_hover_text([text] * 2)
        return self

    def _connect_canvas(self, canvas: Canvas):
        canvas.events.lims.connect(self._recalculate_line, max_args=1)
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
