from __future__ import annotations

import numpy as np
from cmap import Colormap
from numpy.typing import NDArray
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(visuals.Image):
    def __init__(self, data: np.ndarray):
        self._cmap_obj = Colormap("gray")
        # GPU does not support f64
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        super().__init__(data, cmap="gray")
        tr = STTransform()
        self.transform = tr

    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    def _plt_get_data(self) -> np.ndarray:
        return self._data

    def _plt_set_data(self, data: np.ndarray):
        self.set_data(data)

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap_obj

    def _plt_set_colormap(self, cmap: Colormap):
        self.cmap = cmap.to_vispy()
        self._cmap_obj = cmap

    def _plt_get_clim(self) -> tuple[float, float]:
        return self.clim

    def _plt_set_clim(self, clim: tuple[float, float]):
        self.clim = clim

    def _plt_get_translation(self) -> NDArray[np.floating]:
        tr = self._plt_get_transform()
        dx, dy, *_ = tr.translate
        sx, sy = self._plt_get_scale()
        return (dx + 0.5 * sx, dy + 0.5 * sy)

    def _plt_set_translation(self, translation):
        tr = self._plt_get_transform()
        dx, dy = translation
        sx, sy = self._plt_get_scale()
        tr.translate = dx - 0.5 * sx, dy - 0.5 * sy, 0.0, 0.0
        self.transform = tr
        self.update()

    def _plt_get_scale(self) -> NDArray[np.floating]:
        tr = self._plt_get_transform()
        scale = tr.scale
        return scale[0], scale[1]

    def _plt_set_scale(self, scale):
        tr = self._plt_get_transform()
        dx, dy = self._plt_get_translation()
        sx, sy = scale[0], scale[1]
        tr.scale = (sx, sy, 1, 1)
        tr.translate = dx - 0.5 * sx, dy - 0.5 * sy, 0.0, 0.0
        self.transform = tr
        self.update()

    def _plt_get_transform(self) -> STTransform:
        tr = self.transform
        if not isinstance(tr, STTransform):
            raise TypeError(f"Unexpected transform type: {type(tr)}")
        return tr
