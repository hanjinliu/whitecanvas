from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from cmap import Colormap
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(AxesImage):
    def __init__(self, data: np.ndarray):
        super().__init__(plt.gca(), origin="lower")
        self.set_data(data)
        self.set_extent(self.get_extent())  # this is needed!
        self._cmap = Colormap("gray")

    def _plt_get_visible(self) -> bool:
        return self.get_visible()

    def _plt_set_visible(self, visible: bool):
        self.set_visible(visible)

    def _plt_get_zorder(self) -> int:
        return self.get_zorder()

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)

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
