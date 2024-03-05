from __future__ import annotations

import numpy as np
from cmap import Colormap

from whitecanvas.layers._primitive import Image
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.types import ColormapType, Orientation


class Colorbar(LayerContainer):
    _NO_PADDING_NEEDED = True

    def __init__(
        self,
        cmap: Colormap,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
    ):
        self._cmap = cmap
        arr = _cmap_to_image(cmap, orient)
        image = Image(arr, name="lut")
        super().__init__([image], name=name)
        self._orient = orient

    @property
    def lut(self) -> Image:
        """The LUT image layer."""
        return self._children[0]

    @property
    def orient(self) -> Orientation:
        """The orientation of the colorbar."""
        return self._orient

    @property
    def shift(self):
        """The top-left corner of the colorbar."""
        return self.lut.shift

    @shift.setter
    def shift(self, value):
        self.lut.shift = value
        self._move_children()

    @property
    def scale(self):
        """The scale of the colorbar."""
        return self.lut.scale

    @scale.setter
    def scale(self, value):
        self.lut.scale = value
        self._move_children()

    def fit_to(self, bbox):
        self.lut.fit_to(bbox)
        self._move_children()

    @property
    def cmap(self) -> ColormapType:
        """The colormap of the colorbar."""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: ColormapType):
        """Set the colormap of the colorbar."""
        cmap = Colormap(cmap)
        arr = _cmap_to_image(cmap, self.orient)
        self.lut.data = arr
        self._cmap = cmap

    # def with_text(self, ) -> Colorbar:
    #     return self

    def _move_children(self):
        # Move other children to fit the colorbar.
        pass


def _cmap_to_image(cmap: Colormap, orient: Orientation):
    lut = cmap.lut()  # (N, 4)
    width = 50
    if orient.is_vertical:
        arr = np.repeat(lut[:, np.newaxis, :], width, axis=1)
    else:
        arr = np.repeat(lut[np.newaxis, :, :], width, axis=0)
    return arr
