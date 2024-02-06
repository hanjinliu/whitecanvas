from __future__ import annotations

from typing import Any, overload

import numpy as np
from cmap import Colormap
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers._base import DataBoundLayer, LayerEvents
from whitecanvas.protocols import ImageProtocol
from whitecanvas.types import ArrayLike1D, ColormapType, HistBinType, Origin, _Void
from whitecanvas.utils.normalize import as_array_1d

_void = _Void()


class ImageEvents(LayerEvents):
    cmap = Signal(Colormap)
    clim = Signal(tuple)
    shift = Signal(tuple)
    scale = Signal(tuple)


class Image(DataBoundLayer[ImageProtocol, NDArray[np.number]]):
    """
    Grayscale or RGBA image layer.

    Parameters
    ----------
    image : array_like
        2D or 3D array of image data.
    cmap : colormap type, default "gray"
        Colormap to use.
    clim : tuple of float or None, optional
        Contrast limits. If `None`, the limits are set to the min and max of the data.
        You can also pass `None` separately to either limit to only autoscale one of
        them.
    origin : str or Origin, default "corner"
        Origin of the image. This is a redundant parameter which overlaps with `shift`,
        but it makes it easier to operate on the image.
    """

    events: ImageEvents
    _events_class = ImageEvents

    def __init__(
        self,
        image: ArrayLike,
        *,
        name: str | None = None,
        cmap: ColormapType = "gray",
        clim: tuple[float | None, float | None] | None = None,
        shift: tuple[float, float] = (0, 0),
        scale: tuple[float, float] = (1.0, 1.0),
        backend: Backend | str | None = None,
    ):
        img = _normalize_image(image)
        self._origin = Origin.CORNER
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), img)
        if img.ndim == 3:
            cmap = clim = _void
        self._x_hint = self._y_hint = None
        self.update(cmap=cmap, clim=clim, shift=shift, scale=scale)

    def _get_layer_data(self) -> NDArray[np.number]:
        """Current image data of the layer."""
        return self._backend._plt_get_data()

    def _norm_layer_data(self, data: Any) -> NDArray[np.number]:
        return _normalize_image(data)

    def _set_layer_data(self, data: NDArray[np.number]):
        """Set the data of the layer."""
        self._backend._plt_set_data(data)

    @property
    def cmap(self) -> Colormap:
        """Current colormap."""
        return self._backend._plt_get_colormap()

    @cmap.setter
    def cmap(self, cmap: ColormapType):
        _cmap = Colormap(cmap)
        self._backend._plt_set_colormap(_cmap)
        self.events.cmap.emit(_cmap)

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
        self.events.clim.emit((low, high))

    @property
    def shape(self) -> tuple[int, int]:
        """The visual shape of the image (shape without the color axis)."""
        if self.is_rgba:
            return self.data.shape[:2]
        else:
            return self.data.shape

    @property
    def data_mapped(self) -> NDArray[np.number]:
        """The colored image data (N, M, 4) mapped by the colormap."""
        if self.is_rgba:
            return self.data
        # normalize data to [0, 1]
        cmin, cmax = self.clim
        data_norm = (self.data - cmin) / (cmax - cmin)
        return self.cmap(data_norm)

    @property
    def shift(self) -> tuple[float, float]:
        """Current shift from the origin."""
        shift = self._backend._plt_get_translation()
        sx, sy = self.scale
        if self.origin is Origin.EDGE:
            shift = shift[0] - 0.5 * sx, shift[1] - 0.5 * sy
        elif self.origin is Origin.CORNER:
            pass
        elif self.origin is Origin.CENTER:
            sizex, sizey = self.data.shape[:2]
            shift = shift[0] + (sizex - 1) / 2 * sx, shift[1] + (sizey - 1) / 2 * sy
        else:
            raise RuntimeError("Unreachable")
        return shift

    @shift.setter
    def shift(self, shift: tuple[float, float]):
        img = self.data
        sx, sy = self.scale
        if self.origin is Origin.EDGE:
            shift = shift[0] + 0.5 * sx, shift[1] + 0.5 * sy
        elif self.origin is Origin.CORNER:
            pass
        elif self.origin is Origin.CENTER:
            sizex, sizey = img.shape[:2]
            shift = shift[0] - (sizex - 1) / 2 * sx, shift[1] - (sizey - 1) / 2 * sy
        else:
            raise RuntimeError("Unreachable")
        self._backend._plt_set_translation(shift)
        self._x_hint, self._y_hint = _hint_for(
            (img.shape[1], img.shape[0]), shift=shift, scale=self.scale
        )
        self.events.shift.emit(shift)

    @property
    def shift_raw(self) -> tuple[float, float]:
        """Current shift from the origin as a raw data."""
        return self._backend._plt_get_translation()

    @property
    def scale(self) -> tuple[float, float]:
        """Current scale."""
        return self._backend._plt_get_scale()

    @scale.setter
    def scale(self, scale: float | tuple[float, float]):
        if isinstance(scale, (int, float, np.number)):
            scale = float(scale), float(scale)
        dx, dy = scale
        if dx <= 0 or dy <= 0:
            raise ValueError("Scale must be positive.")
        self._backend._plt_set_scale(scale)
        img = self.data
        self._x_hint, self._y_hint = _hint_for(
            (img.shape[1], img.shape[0]), shift=self.shift, scale=scale
        )
        self.events.scale.emit(scale)

    @property
    def origin(self) -> Origin:
        """Current origin of the image."""
        return self._origin

    @origin.setter
    def origin(self, origin: Origin | str):
        self._origin = Origin(origin)
        self.shift = self.shift  # recalculate

    def update(
        self,
        *,
        cmap: ColormapType | _Void = _void,
        clim: tuple[float | None, float | None] | None | _Void = _void,
        shift: tuple[float, float] | _Void = _void,
        scale: tuple[float, float] | _Void = _void,
        origin: str | Origin | _Void = _void,
    ) -> Image:
        if cmap is not _void:
            if self.is_rgba:
                raise ValueError("Cannot set colormap for an RGBA image.")
            self.cmap = cmap
        if clim is not _void:
            if self.is_rgba:
                raise ValueError("Cannot set contrast limits for an RGBA image.")
            self.clim = clim
        if origin is not _void:
            self.origin = origin
        if shift is not _void:
            self.shift = shift
        if scale is not _void:
            self.scale = scale
        return self

    @overload
    def fit_to(self, bbox: tuple[float, float, float, float], /) -> Image:
        ...

    @overload
    def fit_to(self, x0, y0, x1, y1, /) -> Image:
        ...

    def fit_to(self, *args) -> Image:
        """Fit the image to the given bounding box."""
        if len(args) == 1:
            x0, y0, x1, y1 = args[0]
        elif len(args) == 4:
            x0, y0, x1, y1 = args
        else:
            raise TypeError("fit_to() takes 1 or 4 positional arguments.")
        dx, dy = x1 - x0, y1 - y0
        self.shift = (x0 + 0.5, y0 + 0.5)
        self.scale = (dx / self.data.shape[0], dy / self.data.shape[1])
        return self

    @property
    def is_rgba(self) -> bool:
        """Whether the image is RGBA."""
        return self.data.ndim == 3

    @classmethod
    def build_hist(
        cls,
        x: ArrayLike1D,
        y: ArrayLike1D,
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        range=None,
        name: str | None = None,
        density: bool = False,
        cmap: ColormapType = "inferno",
        backend: Backend | str | None = None,
    ):
        _x = as_array_1d(x)
        _y = as_array_1d(y)
        if _x.size != _y.size:
            raise ValueError("x and y must have the same size.")
        if isinstance(bins, (int, np.number, str)):
            xbins = ybins = bins
        else:
            xbins, ybins = bins
        if range is None:
            xrange = yrange = None
        else:
            xrange, yrange = range
        _bins = (
            np.histogram_bin_edges(_x, xbins, xrange),
            np.histogram_bin_edges(_y, ybins, yrange),
        )
        hist, xedges, yedges = np.histogram2d(
            _x, _y, bins=_bins, range=(xrange, yrange), density=density
        )
        hist_t = hist.T
        shift = (xedges[0], yedges[0])
        scale = (xedges[1] - xedges[0], yedges[1] - yedges[0])
        self = cls(
            hist_t, name=name, cmap=cmap, shift=shift, scale=scale, backend=backend
        )
        self.origin = Origin.EDGE
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


def _hint_for(
    shape: tuple[int, int],
    shift: tuple[float, float] = (0, 0),
    scale: tuple[float, float] = (1, 1),
) -> tuple[float, float]:
    xhint = np.array([-0.5, shape[0] - 0.5]) * scale[0] + shift[0]
    yhint = np.array([-0.5, shape[1] - 0.5]) * scale[1] + shift[1]
    return tuple(xhint), tuple(yhint)
