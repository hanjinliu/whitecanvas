from __future__ import annotations

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
        # self.set_extent(self.get_extent())  # this is needed!
        self._cmap = Colormap("gray")
        self._image_transform = Affine2D()

    def _make_bbox(self):
        return Bbox.from_bounds(-0.5, -0.5, *self._image_shape_2d)

    def _transform_bbox(self, ax):
        self._ax_transform = ax.transData
        self.bbox = TransformedBbox(self.bbox, self._ax_transform)

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
        self.set_transform(Affine2D(mtx))
        self._image_transform = Affine2D(mtx)
        bbox = self._make_bbox().transformed(self._image_transform)
        self.bbox = TransformedBbox(bbox, self._ax_transform)

    def _plt_get_translation(self) -> tuple[float, float]:
        mtx = self._image_transform.get_matrix()
        return mtx[0, 2], mtx[1, 2]

    def _plt_set_translation(self, translation: tuple[float, float]):
        mtx = self._image_transform.get_matrix()
        mtx[0, 2], mtx[1, 2] = translation
        self.set_transform(Affine2D(mtx))
        self._image_transform = Affine2D(mtx)
        bbox = self._make_bbox().transformed(self._image_transform)
        self.bbox = TransformedBbox(bbox, self._ax_transform)
