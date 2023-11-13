from __future__ import annotations

import numpy as np

from neoplot.protocols import ScatterProtocol
from neoplot.layers._base import Layer
from neoplot.layers import layer_namespace as _ns
from neoplot.backend import Backend

class Scatter(Layer[ScatterProtocol]):
    marker = _ns.MarkerNamespace()

    def __init__(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        name: str | None = None,
        marker={},
        backend: Backend | None = None
    ):
        if backend is None:
            backend = Backend()
        self._backend = self._create_backend(backend, xdata, ydata)
        self.name = name
        self.marker.update(marker)

    @property
    def data(self) -> np.ndarray:
        return self._backend._plt_get_data()
    
    @data.setter
    def data(self, data: np.ndarray):
        self._backend._plt_set_data(data)
