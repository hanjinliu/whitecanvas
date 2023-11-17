from __future__ import annotations

import pyqtgraph as pg
import numpy as np
from cmap import Colormap
from ._qt_utils import array_to_qcolor
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(pg.ImageItem):
    def __init__(self, data: np.ndarray):
        super().__init__(data)
        self._cmap = Colormap("gray")

    def _plt_get_visible(self) -> bool:
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        self.setVisible(visible)

    def _plt_get_zorder(self) -> int:
        return self.zValue()

    def _plt_set_zorder(self, zorder: int):
        self.setZValue(zorder)

    def _plt_get_data(self) -> np.ndarray:
        return self.image

    def _plt_set_data(self, data: np.ndarray):
        self.setImage(data)

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        self._cmap = cmap
        stops = cmap.color_stops
        colors = [array_to_qcolor(col) for col in stops.color_array]
        pg_cmap = pg.ColorMap(stops.stops, colors)
        self.setColorMap(pg_cmap)

    def _plt_get_clim(self) -> tuple[float, float]:
        return self.getLevels()

    def _plt_set_clim(self, clim: tuple[float, float]):
        self.setLevels(clim)
