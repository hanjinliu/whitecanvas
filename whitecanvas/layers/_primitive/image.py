from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, overload

import numpy as np
from cmap import Color, Colormap
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers._base import DataBoundLayer, LayerEvents
from whitecanvas.protocols import ImageProtocol
from whitecanvas.types import (
    ArrayLike1D,
    ColormapType,
    ColorType,
    HistBinType,
    KdeBandWidthType,
    Orientation,
    OrientationLike,
    Origin,
    Rect,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d, decode_array, encode_array
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers import Texts, _mixin
    from whitecanvas.layers.group import Colorbar, LabeledImage

    MultiFontTexts = Texts[_mixin.MonoFace, _mixin.MonoEdge, _mixin.MultiFont]

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
    _NO_PADDING_NEEDED = True

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
        if is_real_number(scale):
            scale = float(scale), float(scale)
        dx, dy = scale
        if dx <= 0 or dy <= 0:
            raise ValueError("Scale must be positive.")
        self._backend._plt_set_scale(scale)
        shape = self.data.shape[:2]
        self._x_hint, self._y_hint = _hint_for(
            (shape[1], shape[0]), shift=self.shift, scale=scale
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

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create an Image from a dictionary."""
        return cls(
            d["data"], name=d["name"], cmap=d["cmap"], clim=d["clim"], shift=d["shift"],
            scale=d["scale"], backend=backend,
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "data": self._get_layer_data(),
            "name": self.name,
            "visible": self.visible,
            "cmap": self.cmap,
            "clim": self.clim,
            "shift": self.shift,
            "scale": self.scale,
        }

    @classmethod
    def _post_to_dict(cls, d: dict[str, Any]) -> dict[str, Any]:
        d["data"] = encode_array(d["data"], "image")
        return d

    @classmethod
    def _pre_from_dict(cls, d: dict[str, Any]) -> dict[str, Any]:
        d["data"] = decode_array(d["data"], "image")
        return d

    @overload
    def fit_to(self, bbox: Rect | tuple[float, float, float, float], /) -> Image: ...

    @overload
    def fit_to(
        self, left: float, right: float, bottom: float, top: float, /
    ) -> Image: ...

    def fit_to(self, *args) -> Image:
        """Fit the image to the given bounding box."""
        if len(args) == 1:
            rect = Rect(*args[0])
        elif len(args) == 4:
            rect = Rect(*args)
        else:
            raise TypeError("fit_to() takes 1 or 4 positional arguments.")
        shape = self.data.shape[:2]
        x0, y0 = rect.left, rect.bottom
        sx, sy = rect.width / shape[1], rect.height / shape[0]
        if self.origin is Origin.CORNER:
            self.shift = (x0 + 0.5 * sx, y0 + 0.5 * sy)
        elif self.origin is Origin.EDGE:
            self.shift = x0, y0
        else:
            ny, nx = shape
            self.shift = x0 + (nx + 1) / 2 * sx, y0 + (ny + 1) / 2 * sy
        self.scale = sx, sy
        return self

    def with_text(
        self,
        *,
        size: int = 8,
        color_rule: ColorType | Callable[[np.ndarray], ColorType] | None = None,
        fmt: str = "",
        text_invalid: str | None = None,
        mask: NDArray[np.bool_] | None = None,
    ) -> LabeledImage:
        """
        Add text layer that displays the pixel values of the image.

        Parameters
        ----------
        size : int, default 8
            Font size of the text.
        color_rule : color-like, callable, optional
            Rule to define the color for each text based on the color-mapped image
            intensity.
        fmt : str, optional
            Format string for the text.
        mask : array_like, optional
            Mask to specify which pixel to add text if specified.
        """
        from whitecanvas.layers.group import LabeledImage

        text_layer = self._make_text_layer(
            size=size, color_rule=color_rule, fmt=fmt, text_invalid=text_invalid,
            mask=mask,
        )  # fmt: skip
        return LabeledImage(self, texts=text_layer, name=self.name)

    def with_colorbar(
        self,
        bbox: Rect | None = None,
        *,
        orient: OrientationLike = "vertical",
    ) -> LabeledImage:
        """
        Add a colorbar to the image.

        Parameters
        ----------
        bbox : four float, optional
            Bounding box of the colorbar. If `None`, the colorbar will be placed at the
            bottom-right corner of the image.
        orient : str or Orientation, default "vertical"
            Orientation of the colorbar.
        """
        from whitecanvas.layers.group import LabeledImage

        ori = Orientation.parse(orient)
        cbar = self._make_colorbar(bbox=bbox, orient=ori)
        if canvas := self._canvas_ref():
            if ori.is_vertical and canvas.y.flipped:
                cbar.lut.data = cbar.lut.data[::-1]
            elif ori.is_horizontal and canvas.x.flipped:
                cbar.lut.data = cbar.lut.data[:, ::-1]
        return LabeledImage(self, colorbar=cbar, name=self.name)

    def _make_text_layer(
        self,
        *,
        size: int = 8,
        color_rule: ColorType | Callable[[np.ndarray], ColorType] | None = None,
        fmt: str = "",
        text_invalid: str | None = None,
        mask: NDArray[np.bool_] | None = None,
    ) -> MultiFontTexts:
        from whitecanvas.layers._primitive import Texts

        # normalize color_rule
        _color_rule: Callable[[np.ndarray], np.ndarray]

        def _norm_color(x) -> NDArray[np.float32]:
            if isinstance(x, np.ndarray):
                return x.astype(np.float32, copy=False)
            return np.array(Color(x).rgba, dtype=np.float32)

        if color_rule is None:
            _b = _norm_color("black")
            _w = _norm_color("white")
            _bg = _norm_color(theme.get_theme().background_color)[:3]

            def _color_rule(x: NDArray[np.number]) -> NDArray[np.float32]:
                alpha = x[3]
                _col = x[:3] * alpha + _bg * (1 - alpha)
                return _b if _col.sum() > 1.5 else _w

        elif callable(color_rule):
            _color_rule = lambda x: _norm_color(color_rule(x))  # noqa: E731
        else:
            _col = _norm_color(color_rule)
            _color_rule = lambda _: _col  # noqa: E731

        # normalize mask
        if mask is None:
            ij_iter = np.ndindex(self.data.shape[:2])
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != self.data.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask.shape} must be the same as the image "
                    f"shape {self.data.shape[:2]}."
                )
            ij_iter = np.argwhere(mask)

        img_data = self.data
        img_color = self.data_mapped
        ny, nx = self.shape
        dx, dy = self.shift_raw
        sx, sy = self.scale
        ys = np.arange(ny) * sy + dy
        xs = np.arange(nx) * sx + dx
        texts: list[str] = []
        xdata: list[float] = []
        ydata: list[float] = []
        colors: list[np.ndarray] = []
        # normalize fmt
        if fmt:
            if fmt.startswith(":"):
                fmt_style = "{" + fmt + "}"
            elif fmt.startswith("{") and fmt.endswith("}"):
                fmt_style = fmt
            else:
                fmt_style = "{:" + fmt + "}"
        else:
            fmt_style = "{}"

        for iy, ix in ij_iter:
            x = xs[ix]
            y = ys[iy]
            if np.isfinite(img_data[iy, ix]):
                text = fmt_style.format(img_data[iy, ix])
            else:
                if text_invalid is None:
                    text = repr(img_data[iy, ix])
                else:
                    text = text_invalid
            texts.append(text)
            xdata.append(x)
            ydata.append(y)
            colors.append(_color_rule(img_color[iy, ix]))
        return Texts(
            np.array(xdata), np.array(ydata), texts, size=size, anchor="center"
        ).with_font_multi(color=np.stack(colors, axis=0))

    def _make_colorbar(
        self,
        bbox: Rect | None = None,
        orient: OrientationLike = "vertical",
    ) -> Colorbar:
        from whitecanvas.layers.group import Colorbar

        orient = Orientation.parse(orient)
        cbar = Colorbar.from_cmap(
            self.cmap, name=f"{self.name}:colorbar", orient=orient
        )
        if bbox is None:
            img_bbox = self.bbox
            if orient.is_vertical:
                bbox = Rect(
                    img_bbox.left + img_bbox.width * 0.92,
                    img_bbox.left + img_bbox.width * 0.97,
                    img_bbox.bottom + img_bbox.height * 0.03,
                    img_bbox.bottom + img_bbox.height * 0.23,
                )
            else:
                bbox = Rect(
                    img_bbox.left + img_bbox.width * 0.77,
                    img_bbox.left + img_bbox.width * 0.97,
                    img_bbox.bottom + img_bbox.height * 0.03,
                    img_bbox.bottom + img_bbox.height * 0.08,
                )
        cbar.fit_to(bbox)
        return cbar

    @property
    def is_rgba(self) -> bool:
        """Whether the image is RGBA."""
        return self.data.ndim == 3

    @property
    def bbox(self) -> Rect:
        """Bounding box of the image."""
        ox, oy = self.shift_raw
        sizey, sizex = self.data.shape[:2]
        sx, sy = self.scale
        return Rect(
            left=ox - 0.5 * sx,
            right=ox + (sizex - 0.5) * sx,
            bottom=oy - 0.5 * sy,
            top=oy + (sizey - 0.5) * sy,
        )

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
        if isinstance(bins, tuple):
            xbins, ybins = bins
        else:
            xbins = ybins = bins
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

    @classmethod
    def build_kde(
        cls,
        x: ArrayLike1D,
        y: ArrayLike1D,
        shape: tuple[int, int] = (256, 256),
        range=None,
        band_width: KdeBandWidthType = "scott",
        name: str | None = None,
        cmap: ColormapType = "inferno",
        backend: Backend | str | None = None,
    ) -> Image:
        from whitecanvas.utils.kde import gaussian_kde

        _x = as_array_1d(x)
        _y = as_array_1d(y)
        kde = gaussian_kde([_x, _y], bw_method=band_width)
        if range is None:
            xrange = yrange = None
        else:
            xrange, yrange = range
        if xrange is None:
            xrange = _x.min(), _x.max()
        if yrange is None:
            yrange = _y.min(), _y.max()
        xedges = np.linspace(*xrange, shape[0])
        yedges = np.linspace(*yrange, shape[1])
        xx, yy = np.meshgrid(xedges, yedges)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        val = np.reshape(kde(positions).T, xx.shape)
        shift = (xedges[0], yedges[0])
        scale = (xedges[1] - xedges[0], yedges[1] - yedges[0])
        self = cls(val, name=name, cmap=cmap, shift=shift, scale=scale, backend=backend)
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
