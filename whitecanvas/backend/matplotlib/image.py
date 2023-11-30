from __future__ import annotations
from matplotlib.backend_bases import RendererBase

import numpy as np
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D
from cmap import Colormap

from whitecanvas.backend.matplotlib._base import MplLayer
from whitecanvas.protocols import ImageProtocol, check_protocol


@check_protocol(ImageProtocol)
class Image(BboxImage, MplLayer):
    bbox: TransformedBbox

    def __init__(self, data: np.ndarray):
        if data.ndim == 2:
            shape = data.shape
        else:
            shape = data.shape[:2]
        self._image_shape_2d = shape
        super().__init__(self._make_bbox(), origin="lower")
        self.set_data(data)
        self._cmap = Colormap("gray")
        self._image_transform = Affine2D()

    def _make_bbox(self):
        h, w = self._image_shape_2d
        return Bbox.from_bounds(-0.5, -0.5, w, h)

    def _transform_bbox(self, ax):
        self.bbox = ax.transData

    def _plt_get_data(self) -> np.ndarray:
        return self.get_array()

    def _plt_set_data(self, data: np.ndarray):
        self.set_data(data)
        if data.ndim == 2:
            shape = data.shape
        else:
            shape = data.shape[:2]
        self._image_shape_2d = shape

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        self._cmap = cmap
        self.set_cmap(cmap.to_matplotlib())

    def _plt_get_clim(self) -> tuple[float, float]:
        return self.get_clim()

    def _plt_set_clim(self, clim: tuple[float, float]):
        self.set_clim(clim)

    def _plt_get_scale(self) -> tuple[float, float]:
        mtx = self._image_transform.get_matrix()
        return mtx[0, 0], mtx[1, 1]

    def _plt_set_scale(self, scale: tuple[float, float]):
        mtx = self._image_transform.get_matrix()
        mtx[0, 0], mtx[1, 1] = scale
        self._image_transform = Affine2D(mtx)

    def _plt_get_translation(self) -> tuple[float, float]:
        mtx = self._image_transform.get_matrix()
        return mtx[0, 2], mtx[1, 2]

    def _plt_set_translation(self, translation: tuple[float, float]):
        mtx = self._image_transform.get_matrix()
        mtx[0, 2], mtx[1, 2] = translation
        self._image_transform = Affine2D(mtx)

    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox:
        return (
            self._make_bbox().transformed(self._image_transform).transformed(self.bbox)
        )

    def make_image(
        self,
        renderer: RendererBase,
        magnification: float = 1,
        unsampled: bool = False,
    ) -> tuple[np.ndarray, float, float, Affine2D]:
        img, x, y, trans = super().make_image(renderer, magnification, unsampled)
        mtx = self.get_transform().get_matrix()
        if mtx[0, 0] < 0:
            img = img[:, ::-1]
        if mtx[1, 1] < 0:
            img = img[::-1, :]
        return img, x, y, trans
