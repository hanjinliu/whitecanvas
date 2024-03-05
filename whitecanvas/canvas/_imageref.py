# pragma: no cover
from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np

from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.layers import _mixin
from whitecanvas.types import ColorType, Orientation

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.layers._primitive import Image, Texts

_C = TypeVar("_C", bound="CanvasBase")


class ImageRef(Generic[_C]):
    def __init__(self, canvas: _C, image: Image):
        warnings.warn(
            "ImageRef is deprecated and will be removed in the future. "
            "Please use the Image methods `with_text` `with_colorbar` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        cbar = self._image()._make_colorbar(pos=pos, orient=orient)
        return self._canvas().add_layer(cbar)

    def add_text(
        self,
        *,
        size: int = 8,
        color_rule: ColorType | Callable[[np.ndarray], ColorType] | None = None,
        fmt: str = "",
        text_invalid: str | None = None,
    ) -> Texts[_mixin.MonoFace, _mixin.MonoEdge, _mixin.MultiFont]:
        text_layer = self._image()._make_text_layer(
            size=size, color_rule=color_rule, fmt=fmt, text_invalid=text_invalid
        )  # fmt: skip
        return self._canvas().add_layer(text_layer)
