from __future__ import annotations
from typing import Iterator, Literal, Hashable

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, LineStyle, _Void
from whitecanvas.layers.primitive import Band
from whitecanvas.layers.group._collections import DictLayerGroup
from whitecanvas.utils.normalize import as_array_1d


class ViolinPlot(DictLayerGroup):
    def __init__(
        self,
        bands: dict[Hashable, Band],
        *,
        name: str | None = None,
        shape: Literal["both", "left", "right"] = "both",
        violin_width: float = 0.3,
    ):
        if violin_width <= 0:
            raise ValueError(f"violin_width must be positive, got {violin_width}")
        self._violin_width = violin_width
        self._shape = shape
        half_widths = []
        for band in bands.values():
            half_width = np.max(np.abs(band.data.ydiff))
            if shape == "both":
                half_width /= 2
            half_widths.append(half_width)
        factor = violin_width / np.max(half_widths) / 2
        for band in bands.values():
            data = band.data
            ycenter = data.ycenter
            y0 = (data.y0 - ycenter) * factor + ycenter
            y1 = (data.y1 - ycenter) * factor + ycenter
            band.set_data(data.x, y0, y1)
        super().__init__(bands, name=name)

    @classmethod
    def from_dict(
        cls,
        data: dict[Hashable, ArrayLike],
        *,
        name: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        shape: Literal["both", "left", "right"] = "both",
        violin_width: float = 0.5,
        band_width: float | str = "scott",
        color: ColorType | None = None,
        alpha: float = 1,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: str | Backend | None = None,
    ):
        from whitecanvas.utils.kde import gaussian_kde

        backend = Backend(backend)
        layers = {}
        if orient == "vertical":
            _o = "horizontal"
        else:
            _o = "vertical"
        for offset, (key, values) in enumerate(data.items()):
            arr = as_array_1d(values)
            kde = gaussian_kde(arr, bw_method=band_width)

            sigma = np.sqrt(kde.covariance[0, 0])
            pad = sigma * 4
            x = np.linspace(arr.min() - pad, arr.max() + pad, 100)
            y = kde(x)
            if shape in ("both", "left"):
                y0 = -y + offset
            else:
                y0 = np.zeros_like(y) + offset
            if shape in ("both", "right"):
                y1 = y + offset
            else:
                y1 = np.zeros_like(y) + offset

            layer = Band(
                x, y0, y1, name=key, orient=_o, color=color, alpha=alpha,
                pattern=pattern, backend=backend,
            )  # fmt: skip
            layers[key] = layer
        return cls(layers, name=name, violin_width=violin_width, shape=shape)

    def __getitem__(self, key: Hashable) -> Band:
        return self._children[key]

    def iter_children(self) -> Iterator[Band]:
        # Just for typing
        return super().iter_children()

    @property
    def offsets(self):
        if self.orient == "vertical":
            return [band.data.x[0] for band in self.iter_children()]
        else:
            return [band.data.y[0] for band in self.iter_children()]

    @property
    def orient(self):
        try:
            band = next(self.iter_children())
        except StopIteration:
            raise ValueError("No children in this group") from None
        return band.orient

    @property
    def violin_width(self):
        return self._violin_width
