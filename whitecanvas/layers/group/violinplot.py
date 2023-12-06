from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Literal, Hashable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, Orientation, LineStyle
from whitecanvas.layers._primitive import Band
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.utils.normalize import as_array_1d, as_color_array


class ViolinPlot(ListLayerGroup):
    def __init__(
        self,
        bands: dict[Hashable, Band],
        *,
        name: str | None = None,
        shape: Literal["both", "left", "right"] = "both",
        extent: float = 0.5,
        orient: Orientation = Orientation.VERTICAL,
    ):
        super().__init__(bands, name=name)
        self._shape = shape
        self._extent = extent
        self._orient = Orientation(orient)

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike],
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        shape: Literal["both", "left", "right"] = "both",
        extent: float = 0.5,
        kde_band_width: float | str = "scott",
        color: ColorType | list[ColorType] = "blue",
        alpha: float = 1,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: str | Backend | None = None,
    ):
        from whitecanvas.utils.kde import gaussian_kde

        backend = Backend(backend)
        _violin_o = Orientation.parse(orient)
        _o = _violin_o.transpose()
        if extent <= 0:
            raise ValueError(f"extent must be positive, got {extent}")
        x, data = check_array_input(x, data)
        color = as_color_array(color, len(x))
        layers: list[Band] = []
        for ith, (offset, values, col) in enumerate(zip(x, data, color)):
            arr = as_array_1d(values)
            kde = gaussian_kde(arr, bw_method=kde_band_width)

            sigma = np.sqrt(kde.covariance[0, 0])
            pad = sigma * 2.5
            x_ = np.linspace(arr.min() - pad, arr.max() + pad, 100)
            y = kde(x_)
            if shape in ("both", "left"):
                y0 = -y + offset
            else:
                y0 = np.zeros_like(y) + offset
            if shape in ("both", "right"):
                y1 = y + offset
            else:
                y1 = np.zeros_like(y) + offset

            layer = Band(
                x_, y0, y1, name=f"violin_{ith}", orient=_o, color=col,
                alpha=alpha, pattern=pattern, backend=backend,
            )  # fmt: skip
            layers.append(layer)

        half_widths = []
        for band in layers:
            half_width = np.max(np.abs(band.data.ydiff))
            if shape == "both":
                half_width /= 2
            half_widths.append(half_width)
        factor = extent / np.max(half_widths) / 2
        for band, xoffset in zip(layers, x):
            bd = band.data
            y0 = (bd.y0 - xoffset) * factor + xoffset
            y1 = (bd.y1 - xoffset) * factor + xoffset
            band.set_data(bd.x, y0, y1)
        return cls(layers, name=name, shape=shape, extent=extent, orient=_violin_o)

    def __getitem__(self, key: Hashable) -> Band:
        return self._children[key]

    def iter_children(self) -> Iterator[Band]:
        # Just for typing
        return super().iter_children()

    @property
    def offsets(self) -> NDArray[np.floating]:
        _it = enumerate(self.iter_children())
        if self.orient.is_vertical:
            return np.array([band.data.x[0] - i for i, band in _it], dtype=float)
        else:
            return np.array([band.data.y[0] - i for i, band in _it], dtype=float)

    @property
    def orient(self) -> Orientation:
        return self._orient

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, width: float):
        if width <= 0:
            raise ValueError(f"extent must be positive, got {width}")
        factor = width / self.extent
        for band in self.iter_children():
            bd = band.data
            ycenter = bd.ycenter
            y0 = (bd.y0 - ycenter) * factor + ycenter
            y1 = (bd.y1 - ycenter) * factor + ycenter
            band.set_data(bd.x, y0, y1)
        self._extent = width

    def with_edge(
        self,
        *,
        color: ColorType = "black",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> ViolinPlot:
        """Add edges to the strip plot."""
        for layer in self.iter_children():
            layer.with_edge(color=color, alpha=alpha, width=width, style=style)
        return self

    if TYPE_CHECKING:

        def iter_children(self) -> Iterator[Band]:
            ...
