from __future__ import annotations

import numpy as np
from cmap import Colormap

import bokeh.models as bk_models
from ._base import BokehLayer
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(BokehLayer[bk_models.Image]):
    def __init__(self, data: np.ndarray):
        self._data = bk_models.ColumnDataSource(dict(image=[data]))
        self._model = bk_models.Image(image="image", x=0, y=0, dw=1, dh=1)
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
        return self._model.x, self._model.y

    def _plt_set_translation(self, translation: tuple[float, float]):
        self._model.x, self._model.y = translation

    def _plt_get_scale(self) -> tuple[float, float]:
        return self._model.dw, self._model.dh

    def _plt_set_scale(self, scale: tuple[float, float]):
        self._model.dw, self._model.dh = scale