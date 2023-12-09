from __future__ import annotations

import numpy as np
from cmap import Colormap

import bokeh.models as bk_models
from ._base import BokehLayer
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(BokehLayer[bk_models.Image]):
    def __init__(self, data: np.ndarray):
        self._data = bk_models.ColumnDataSource(dict(image=[data], hovertexts=[""]))
        h, w = data.shape[:2]
        self._model = bk_models.Image(image="image", x=0, y=0, dw=w, dh=h)
        self._cmap = Colormap("gray")

    def _plt_get_data(self) -> np.ndarray:
        return self._data.data["image"][0]

    def _plt_set_data(self, data: np.ndarray):
        self._data.data = dict(image=[data])

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        mapper = cmap.to_bokeh()
        mapper.low = self._model.color_mapper.low
        mapper.high = self._model.color_mapper.high
        self._model.color_mapper = mapper
        self._cmap = cmap

    def _plt_get_clim(self) -> tuple[float, float]:
        mapper = self._model.color_mapper
        return mapper.low, mapper.high

    def _plt_set_clim(self, clim: tuple[float, float]):
        self._model.color_mapper.low, self._model.color_mapper.high = clim

    def _plt_get_translation(self) -> tuple[float, float]:
        sx, sy = self._plt_get_scale()
        return self._model.x + 0.5 * sx, self._model.y + 0.5 * sy

    def _plt_set_translation(self, translation: tuple[float, float]):
        dx, dy = translation
        sx, sy = self._plt_get_scale()
        self._model.x, self._model.y = dx - 0.5 * sx, dy - 0.5 * sy

    def _plt_get_scale(self) -> tuple[float, float]:
        h, w = self._data.data["image"][0].shape[:2]
        return self._model.dw / w, self._model.dh / h

    def _plt_set_scale(self, scale: tuple[float, float]):
        h, w = self._data.data["image"][0].shape[:2]
        dx, dy = self._plt_get_translation()
        sx, sy = scale
        self._model.dw, self._model.dh = sx * w, sy * h
        self._model.x, self._model.y = dx - 0.5 * sx, dy - 0.5 * sy
