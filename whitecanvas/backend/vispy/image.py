from __future__ import annotations

import numpy as np
from vispy.scene import visuals
from cmap import Colormap
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(visuals.Image):
    def __init__(self, data: np.ndarray):
        super().__init__(data, cmap="gray")
        self._cmap = Colormap("gray")

    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    def _plt_get_data(self) -> np.ndarray:
        return self._data

    def _plt_set_data(self, data: np.ndarray):
        self.set_data(data)

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        self.cmap = cmap.to_vispy()

    def _plt_get_clim(self) -> tuple[float, float]:
        return self.clim

    def _plt_set_clim(self, clim: tuple[float, float]):
        self.clim = clim
