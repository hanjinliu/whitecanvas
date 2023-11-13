from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from matplotlib.collections import PathCollection
import matplotlib.markers as mmarkers
from neoplot.protocols import ScatterProtocol
from neoplot.types import Symbol
from cmap import Color

def _get_path(symbol: Symbol):
    marker_obj = mmarkers.MarkerStyle(symbol.value)
    return marker_obj.get_path().transformed(marker_obj.get_transform())

class Scatter(PathCollection):
    def __init__(self, xdata, ydata):
        import matplotlib.pyplot as plt
        plt.scatter
        offsets = np.stack([xdata, ydata], axis=1)
        self._symbol = Symbol.CIRCLE
        super().__init__((_get_path(self._symbol),), offsets=offsets)

    ##### LayerProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.get_visible()
    
    def _plt_set_visible(self, visible: bool):
        self.set_visible(visible)
    
    def _plt_get_zorder(self) -> int:
        return self.get_zorder()
    
    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

    ##### XYDataProtocol #####
    def _plt_get_data(self) -> np.ndarray:
        return self.get_offsets()
    
    def _plt_set_data(self, data: np.ndarray):
        self.set_offsets(data)

    ##### HasSymbol protocol #####

    def _plt_get_symbol(self) -> Symbol:
        return self._symbol
    
    def _plt_set_symbol(self, symbol: Symbol):
        path = _get_path(symbol)
        self.set_paths([path] * len(self.get_offsets()))
    
    def _plt_get_symbol_size(self) -> float:
        return self.get_sizes()[0]
    
    def _plt_set_symbol_size(self, size: float):
        self.set_sizes([size] * len(self.get_offsets()))
    
    def _plt_get_symbol_face_color(self) -> NDArray[np.float32]:
        return self.get_facecolor()[0]
    
    def _plt_set_symbol_face_color(self, color: NDArray[np.float32]):
        self.set_facecolor([color] * len(self.get_offsets()))

    def _plt_get_symbol_edge_color(self) -> NDArray[np.float32]:
        return self.get_edgecolor()[0]
    
    def _plt_set_symbol_edge_color(self, color: NDArray[np.float32]):
        self.set_edgecolor([color] * len(self.get_offsets()))

assert isinstance(Scatter, ScatterProtocol)
