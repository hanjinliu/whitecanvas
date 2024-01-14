from __future__ import annotations

from typing import Any

import numpy as np

from whitecanvas.backend import Backend
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.layers._mixin import FaceEdgeMixin
from whitecanvas.layers._sizehint import xyy_size_hint
from whitecanvas.protocols import BandProtocol
from whitecanvas.types import ArrayLike1D, ColorType, Hatch, Orientation, XYYData
from whitecanvas.utils.normalize import as_array_1d


class Band(DataBoundLayer[BandProtocol, XYYData], FaceEdgeMixin):
    def __init__(
        self,
        t: ArrayLike1D,
        edge_low: ArrayLike1D,
        edge_high: ArrayLike1D,
        orient: str | Orientation = Orientation.VERTICAL,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
        FaceEdgeMixin.__init__(self)
        ori = Orientation.parse(orient)
        x = as_array_1d(t)
        y0 = as_array_1d(edge_low)
        y1 = as_array_1d(edge_high)
        if x.size != y0.size or x.size != y1.size:
            raise ValueError(
                "Expected xdata, ydata0, ydata1 to have the same size, "
                f"got {x.size}, {y0.size}, {y1.size}"
            )
        super().__init__(name=name if name is not None else "Band")
        self._backend = self._create_backend(Backend(backend), x, y0, y1, ori)
        self._orient = ori
        self.face.update(color=color, alpha=alpha, hatch=hatch)
        self._x_hint, self._y_hint = xyy_size_hint(x, y0, y1, ori)
        self._band_type = "band"

    @property
    def orient(self) -> Orientation:
        """Orientation of the band."""
        return self._orient

    def _get_layer_data(self) -> XYYData:
        """Current data of the layer."""
        if self._orient.is_vertical:
            x, y0, y1 = self._backend._plt_get_vertical_data()
        else:
            x, y0, y1 = self._backend._plt_get_horizontal_data()
        return XYYData(x, y0, y1)

    def _norm_layer_data(self, data: Any) -> XYYData:
        t0, y0, y1 = self.data
        t, edge_low, edge_high = data
        if t is not None:
            t0 = t
        if edge_low is not None:
            y0 = as_array_1d(edge_low)
        if edge_high is not None:
            y1 = as_array_1d(edge_high)
        if t0.size != y0.size or t0.size != y1.size:
            raise ValueError(
                "Expected data to have the same size,"
                f"got {t0.size}, {y0.size}, {y1.size}"
            )
        return XYYData(t0, y0, y1)

    def _set_layer_data(self, data: XYYData):
        t0, y0, y1 = data
        if self._orient.is_vertical:
            self._backend._plt_set_vertical_data(t0, y0, y1)
        else:
            self._backend._plt_set_horizontal_data(t0, y0, y1)
        self._x_hint, self._y_hint = xyy_size_hint(t0, y0, y1, self.orient)

    def set_data(
        self,
        t: ArrayLike1D | None = None,
        edge_low: ArrayLike1D | None = None,
        edge_high: ArrayLike1D | None = None,
    ):
        self.data = t, edge_low, edge_high

    @classmethod
    def from_kde(
        cls,
        data: ArrayLike1D,
        bottom: float = 0.0,
        *,
        name: str | None = None,
        band_width: float | None = None,
        color: ColorType = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: Backend | str | None = None,
    ):
        from whitecanvas.utils.kde import gaussian_kde

        data = as_array_1d(data)
        kde = gaussian_kde(data, bw_method=band_width)

        sigma = np.sqrt(kde.covariance[0, 0])
        pad = sigma * 2.5
        x = np.linspace(data.min() - pad, data.max() + pad, 100)
        y1 = kde(x)
        y0 = np.full_like(y1, bottom)
        self = cls(
            x, y0, y1, name=name, orient=orient, color=color, alpha=alpha,
            hatch=hatch, backend=backend,
        )  # fmt: skip
        self._band_type = "kde"
        return self
