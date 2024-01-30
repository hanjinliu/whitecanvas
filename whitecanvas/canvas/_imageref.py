from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
from cmap import Color
from numpy.typing import NDArray

from whitecanvas import theme
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.layers import _mixin
from whitecanvas.types import ColorType, Orientation

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase
    from whitecanvas.layers._primitive import Image, Texts

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

    def add_text(
        self,
        *,
        size: int = 8,
        color_rule: ColorType | Callable[[np.ndarray], ColorType] | None = None,
        fmt: str = "",
    ) -> Texts[_mixin.MonoFace, _mixin.MonoEdge, _mixin.MultiFont]:
        """
        Add text annotation to each pixel of the image.

        Parameters
        ----------
        size : int, default 8
            Font size of the text.
        color_rule : color-like, callable, optional
            Rule to define the color for each text based on the color-mapped image
            intensity.
        fmt : str, optional
            Format string for the text.

        Returns
        -------
        Texts
            Texts layer of the text annotation.
        """
        canvas = self._canvas()
        image = self._image()

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

        img_data = image.data
        img_color = image.data_mapped
        ny, nx = image.shape
        dx, dy = image.shift_raw
        sx, sy = image.scale
        ys = np.arange(ny) * sy + dy
        xs = np.arange(nx) * sx + dx
        texts: list[str] = []
        xdata: list[float] = []
        ydata: list[float] = []
        colors: list[np.ndarray] = []
        if fmt:
            fmt_style = "{:" + fmt + "}"
        else:
            fmt_style = "{}"
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                texts.append(fmt_style.format(img_data[iy, ix]))
                xdata.append(x)
                ydata.append(y)
                colors.append(_color_rule(img_color[iy, ix]))
        return canvas.add_text(
            xdata,
            ydata,
            texts,
            size=size,
            anchor="center",
        ).with_font_multi(color=np.stack(colors, axis=0))
