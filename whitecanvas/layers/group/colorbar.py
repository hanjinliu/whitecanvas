from __future__ import annotations

import numpy as np
from cmap import Colormap
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.layers._primitive import Image
from whitecanvas.types import ColormapType, Orientation


class Colorbar(LayerContainer):
    def __init__(
        self,
        cmap: Colormap,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
    ):
        lut = cmap.lut()  # (N, 4)
        width = 50
        if orient.is_vertical:
            arr = np.repeat(lut[:, np.newaxis, :], width, axis=1)
        else:
            arr = np.repeat(lut[np.newaxis, :, :], width, axis=0)
        image = Image(arr, name="lut")
        super().__init__([image], name=name)

    @property
    def lut(self) -> Image:
        """The LUT image layer."""
        return self._children[0]

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

    # def with_text(self, ) -> Colorbar:
    #     return self

    def _move_children(self):
        # Move other children to fit the colorbar.
        pass
