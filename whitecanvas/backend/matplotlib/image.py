from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from cmap import Colormap

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(AxesImage, MplLayer):
    def __init__(self, data: np.ndarray):
        super().__init__(plt.gca(), origin="lower")
        self.set_data(data)
        self.set_extent(self.get_extent())  # this is needed!
        self._cmap = Colormap("gray")

    def _plt_get_data(self) -> np.ndarray:
        return self.get_array()

    def _plt_set_data(self, data: np.ndarray):
        self.set_data(data)

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        self._cmap = cmap
        self.set_cmap(cmap.to_matplotlib())

    def _plt_get_clim(self) -> tuple[float, float]:
        return self.get_clim()

    def _plt_set_clim(self, clim: tuple[float, float]):
        self.set_clim(clim)
