from __future__ import annotations

import pyqtgraph as pg
import numpy as np
from cmap import Colormap
from ._qt_utils import array_to_qcolor
from qtpy.QtGui import QTransform
from whitecanvas.protocols import ImageProtocol, check_protocol
from whitecanvas.backend.pyqtgraph._base import PyQtLayer


@check_protocol(ImageProtocol)
class Image(pg.ImageItem, PyQtLayer):
    def __init__(self, data: np.ndarray):
        super().__init__(data)
        self._cmap = Colormap("gray")
        self.setTransform(QTransform())

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
        low, high = self.getLevels()
        return low, high

    def _plt_set_clim(self, clim: tuple[float, float]):
        self.setLevels(clim)

    def _get_qtransform(self) -> QTransform:
        return self.transform()

    def _plt_get_translation(self) -> tuple[float, float]:
        tr = self._get_qtransform()
        return tr.dx(), tr.dy()

    def _plt_set_translation(self, translation: tuple[float, float]):
        tr = self._get_qtransform()
        tr.setMatrix(
            tr.m11(),
            tr.m12(),
            tr.m13(),
            tr.m21(),
            tr.m22(),
            tr.m23(),
            translation[0],
            translation[1],
            tr.m33(),
        )
        self.setTransform(tr)

    def _plt_get_scale(self) -> tuple[float, float]:
        tr = self._get_qtransform()
        return tr.m11(), tr.m22()

    def _plt_set_scale(self, scale: tuple[float, float]):
        tr = self._get_qtransform()
        tr.setMatrix(
            scale[0],
            tr.m12(),
            tr.m13(),
            tr.m21(),
            scale[1],
            tr.m23(),
            tr.m31(),
            tr.m32(),
            tr.m33(),
        )
        self.setTransform(tr)
