from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from cmap import Colormap
from qtpy.QtGui import QTransform

from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._qt_utils import array_to_qcolor
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(pg.ImageItem, PyQtLayer):
    def __init__(self, data: np.ndarray):
        super().__init__(np.swapaxes(data, 0, 1))
        self._cmap = Colormap("gray")
        self.setTransform(QTransform())

    def _plt_get_data(self) -> np.ndarray:
        return np.swapaxes(self.image, 0, 1)

    def _plt_set_data(self, data: np.ndarray):
        self.setImage(np.swapaxes(data, 0, 1))

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        self._cmap = cmap
        stops = cmap.color_stops
        colors = [array_to_qcolor(col) for col in stops.color_array]
        pg_cmap = pg.ColorMap(stops.stops, colors)
        self.setColorMap(pg_cmap)

    def _plt_get_clim(self) -> tuple[float, float]:
        low, high = self.getLevels()
        return low, high

    def _plt_set_clim(self, clim: tuple[float, float]):
        self.setLevels(clim)

    def _get_qtransform(self) -> QTransform:
        return self.transform()

    def _plt_get_translation(self) -> tuple[float, float]:
        tr = self._get_qtransform()
        return tr.dx() + 0.5 * tr.m11(), tr.dy() + 0.5 * tr.m22()

    def _plt_set_translation(self, translation: tuple[float, float]):
        tr = self._get_qtransform()
        tr.setMatrix(
            tr.m11(),
            tr.m12(),
            tr.m13(),
            tr.m21(),
            tr.m22(),
            tr.m23(),
            translation[0] - 0.5 * tr.m11(),
            translation[1] - 0.5 * tr.m22(),
            tr.m33(),
        )
        self.setTransform(tr)

    def _plt_get_scale(self) -> tuple[float, float]:
        tr = self._get_qtransform()
        return tr.m11(), tr.m22()

    def _plt_set_scale(self, scale: tuple[float, float]):
        tr = self._get_qtransform()
        shift = self._plt_get_translation()
        tr.setMatrix(
            scale[0],
            tr.m12(),
            tr.m13(),
            tr.m21(),
            scale[1],
            tr.m23(),
            shift[0] - 0.5 * scale[0],
            shift[1] - 0.5 * scale[1],
            tr.m33(),
        )
        self.setTransform(tr)
