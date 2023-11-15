from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import numpy as np

from neoplot.backend import Backend
from neoplot.types import LineStyle
from neoplot.layers.line import Line

if TYPE_CHECKING:
    from typing import NoReturn


class InfCurve(Line):
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[float, float] = (-np.inf, np.inf),
        *,
        name: str | None = None,
        color=None,
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
        xdata = np.array([_lower, _upper])
        ydata = model(xdata)
        super().__init__(
            xdata, ydata, name=name, color=color, width=width, style=style, antialias=antialias, backend=backend
        )
        self._bounds = bounds
        self._model = model

        # TODO: connect events

    @property
    def data(self) -> NoReturn:
        raise NotImplementedError("Cannot get data from an InfCurve layer.")

    @property
    def model(self) -> Callable[[np.ndarray], np.ndarray]:
        """The model function of the layer."""
        return self._model
