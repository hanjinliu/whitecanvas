from __future__ import annotations
from typing import Iterator, Literal, Hashable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, Orientation
from whitecanvas.layers.primitive import Band
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.utils.normalize import as_array_1d


class ViolinPlot(ListLayerGroup):
    def __init__(
        self,
        bands: dict[Hashable, Band],
        *,
        name: str | None = None,
        shape: Literal["both", "left", "right"] = "both",
        violin_width: float = 0.5,
        orient: Orientation = Orientation.VERTICAL,
    ):
        super().__init__(bands, name=name)
        self._shape = shape
        self._violin_width = violin_width
        self._orient = Orientation(orient)

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike],
        labels: list[str] | None = None,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        shape: Literal["both", "left", "right"] = "both",
        violin_width: float = 0.5,
        kde_band_width: float | str = "scott",
        color: ColorType | None = None,
        alpha: float = 1,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: str | Backend | None = None,
    ):
        from whitecanvas.utils.kde import gaussian_kde

        backend = Backend(backend)
        _violin_o = Orientation.parse(orient)
        _o = _violin_o.transpose()
        if violin_width <= 0:
            raise ValueError(f"violin_width must be positive, got {violin_width}")
        x, data, labels = check_array_input(x, data, labels)
        layers: list[Band] = []
        for offset, values, label in zip(x, data, labels):
            arr = as_array_1d(values)
            kde = gaussian_kde(arr, bw_method=kde_band_width)

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
                x, y0, y1, name=label, orient=_o, color=color, alpha=alpha,
                pattern=pattern, backend=backend,
            )  # fmt: skip
            layers.append(layer)

        half_widths = []
        for band in layers:
            half_width = np.max(np.abs(band.data.ydiff))
            if shape == "both":
                half_width /= 2
            half_widths.append(half_width)
        factor = violin_width / np.max(half_widths) / 2
        for band in layers:
            bd = band.data
            ycenter = bd.ycenter
            y0 = (bd.y0 - ycenter) * factor + ycenter
            y1 = (bd.y1 - ycenter) * factor + ycenter
            band.set_data(bd.x, y0, y1)
        return cls(
            layers, name=name, shape=shape, violin_width=violin_width, orient=_violin_o
        )

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
    def violin_width(self):
        return self._violin_width

    @violin_width.setter
    def violin_width(self, width: float):
        if width <= 0:
            raise ValueError(f"violin_width must be positive, got {width}")
        factor = width / self.violin_width
        for band in self.iter_children():
            bd = band.data
            ycenter = bd.ycenter
            y0 = (bd.y0 - ycenter) * factor + ycenter
            y1 = (bd.y1 - ycenter) * factor + ycenter
            band.set_data(bd.x, y0, y1)
        self._violin_width = width
