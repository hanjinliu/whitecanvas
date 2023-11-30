from __future__ import annotations

from typing import TypeVar, Generic, TYPE_CHECKING
import weakref
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.types import Orientation

if TYPE_CHECKING:
    from whitecanvas.layers.primitive import Image
    from whitecanvas.canvas._base import CanvasBase

_C = TypeVar("_C", bound="CanvasBase")


class ImageRef(Generic[_C]):
    def __init__(self, canvas: _C, image: Image):
        self._canvas_ref = weakref.ref(canvas)
        self._image_ref = weakref.ref(image)

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    def _image(self) -> Image:
        image = self._image_ref()
        if image is None:
            raise ReferenceDeletedError("Image has been deleted.")
        return image

    def add_colorbar(
        self,
        pos=None,
        orient: str | Orientation = Orientation.VERTICAL,
    ) -> _C:
        from whitecanvas.layers.group.colorbar import Colorbar

        canvas = self._canvas()
        image = self._image()
        orient = Orientation.parse(orient)
        cbar = Colorbar(image.cmap, name=f"colorbar<{image.name}>", orient=orient)
        cbar.shift = pos
        if orient.is_vertical:
            cbar.scale = image.data.shape[0] / 512
        else:
            cbar.scale = image.data.shape[1] / 512
        canvas.add_layer(cbar)
        return cbar

    def add_scalebar(self):
        ...
