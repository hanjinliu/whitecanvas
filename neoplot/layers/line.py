from __future__ import annotations

import numpy as np

from neoplot.protocols import LineProtocol
from neoplot.layers._base import Layer
from neoplot.layers import layer_namespace as _ns
from neoplot.backend import Backend

class Line(Layer[LineProtocol]):
    marker = _ns.MarkerNamespace()
    line = _ns.LineNamespace()

    def __init__(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        name: str | None = None,
        line={},
        marker={},
        backend: Backend | None = None
    ):
        if backend is None:
            backend = Backend()
        self._backend = self._create_backend(backend, xdata, ydata)
        self.name = name
        self.line.update(line)
        self.marker.update(marker)

    @property
    def data(self) -> np.ndarray:
        return self._backend._plt_get_data()
    
    @data.setter
    def data(self, data: np.ndarray):
        self._backend._plt_set_data(data)

    @property
    def antialias(self) -> bool:
        return self._backend._plt_get_antialias()
    
    @antialias.setter
    def antialias(self, antialias: bool):
        self._backend._plt_set_antialias(antialias)
