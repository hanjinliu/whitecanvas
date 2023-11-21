from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from cmap import Colormap
from whitecanvas.protocols import ImageProtocol
from whitecanvas.types import ColormapType, _Void
from whitecanvas.backend import Backend
from whitecanvas.layers._base import PrimitiveLayer

_void = _Void()


class Image(PrimitiveLayer[ImageProtocol]):
    """
    Grayscale or RGBA image layer.

    Parameters
    ----------
    image : array_like
        2D or 3D array of image data.
    cmap : colormap type, default is "gray"
        Colormap to use.
    clim : tuple of float or None, optional
        Contrast limits. If ``None``, the limits are set to the min and max of the data.
        You can also pass ``None`` separately to either limit to only autoscale one of
        them.
    """

    def __init__(
        self,
        image: ArrayLike,
        *,
        name: str | None = None,
        cmap: ColormapType = "gray",
        clim: tuple[float | None, float | None] | None = None,
        backend: Backend | str | None = None,
    ) -> None:
        self._backend = self._create_backend(Backend(backend), _normalize_image(image))
        self.name = name if name is not None else "Image"
        self.update(cmap=cmap, clim=clim)

    @property
    def data(self) -> NDArray[np.number]:
        """Current data of the layer."""
        return self._backend._plt_get_data()

    @data.setter
    def data(self, data: ArrayLike):
        self._backend._plt_set_data(_normalize_image(data))

    @property
    def cmap(self) -> Colormap:
        """Current colormap."""
        return self._backend._plt_get_colormap()

    @cmap.setter
    def cmap(self, cmap: ColormapType):
        self._backend._plt_set_colormap(Colormap(cmap))

    @property
    def clim(self) -> tuple[float, float]:
        """Current contrast limits."""
        return self._backend._plt_get_clim()

    @clim.setter
    def clim(self, clim: tuple[float | None, float | None] | None):
        if clim is None:
            low, high = None, None
        else:
            low, high = clim
        if low is None:
            low = self.data.min()
        if high is None:
            high = self.data.max()
        self._backend._plt_set_clim((low, high))

    def update(
        self,
        *,
        cmap: ColormapType | _void = _void,
        clim: tuple[float | None, float | None] | None | _Void = _void,
    ) -> Image:
        if cmap is not _void:
            self.cmap = cmap
        if clim is not _void:
            self.clim = clim
        return self


def _normalize_image(image):
    img = np.asarray(image)
    if img.dtype.kind not in "uif":
        raise TypeError(f"Only numerical arrays are allowed, got {img.dtype}")
    # check shape
    if img.ndim == 2:
        pass
    elif img.ndim == 3:
        nchannels = img.shape[2]
        if nchannels not in (3, 4):
            raise ValueError(
                "If 3D array is given, the last dimension must be 3 or 4, "
                f"got shape {img.shape}."
            )
    else:
        raise ValueError(f"Only 2D or 3D arrays are allowed, got {img.ndim}")
    return img
