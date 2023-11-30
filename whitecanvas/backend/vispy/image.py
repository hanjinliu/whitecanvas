from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from cmap import Colormap
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(visuals.Image):
    def __init__(self, data: np.ndarray):
        self._cmap_obj = Colormap("gray")
        # GPU does not support f64
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        super().__init__(data, cmap="gray")

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
        tr = self.transform
        if tr is None:
            return (0.0, 0.0)
        elif isinstance(tr, STTransform):
            return tuple(tr.translate)
        else:
            raise TypeError(f"Unexpected transform type: {type(tr)}")

    def _plt_set_translation(self, translation):
        tr = self.transform
        if tr is None:
            tr = STTransform()
        tr.translate = np.asarray(translation, dtype=np.float32)
        self.transform = tr

    def _plt_get_scale(self) -> NDArray[np.floating]:
        tr = self.transform
        if tr is None:
            return (1.0, 1.0)
        elif isinstance(tr, STTransform):
            return tuple(tr.scale)
        else:
            raise TypeError(f"Unexpected transform type: {type(tr)}")

    def _plt_set_scale(self, scale):
        tr = self.transform
        if tr is None:
            tr = STTransform()
        tr.scale = np.asarray(scale, dtype=np.float32)
        self.transform = tr
