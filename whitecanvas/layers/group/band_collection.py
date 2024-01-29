from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers._mixin import CollectionFaceEdgeMixin
from whitecanvas.layers._primitive import Band
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.layers.group._collections import (
    LayerCollectionBase,
    RichContainerEvents,
)
from whitecanvas.types import Orientation, XYYData
from whitecanvas.utils.normalize import as_array_1d


class BandCollection(
    LayerCollectionBase[Band],
    CollectionFaceEdgeMixin,
):
    events: RichContainerEvents
    _events_class = RichContainerEvents

    def __init__(
        self,
        data: list[XYYData],
        *,
        orient: Orientation = Orientation.VERTICAL,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        bands = [Band(*each, orient=orient, backend=backend) for each in data]
        LayerCollectionBase.__init__(self, bands, name=name)
        CollectionFaceEdgeMixin.__init__(self)
        self._orient = orient
        self._init_events()

    @property
    def data(self) -> list[XYYData]:
        return [line.data for line in self]

    @data.setter
    def data(self, data: list[XYYData]):
        ndata_in = len(data)
        ndata_now = len(self)
        if ndata_in > ndata_now:
            for _ in range(ndata_now, ndata_in):
                self.append(Band([], [], []))
        elif ndata_in < ndata_now:
            for _ in range(ndata_in, ndata_now):
                del self[-1]
        for line, d in zip(self, data):
            line.data = d

    @property
    def orient(self) -> Orientation:
        """Orientation of the bands."""
        return self._orient

    @classmethod
    def from_arrays(
        cls,
        y: list[float],
        data: list[XYYData],
        *,
        band_width: float | None = None,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ):
        from whitecanvas.utils.kde import gaussian_kde

        input_ = []
        for bottom, each in zip(y, data):
            _each = as_array_1d(each)
            kde = gaussian_kde(_each, bw_method=band_width)
            sigma = np.sqrt(kde.covariance[0, 0])
            pad = sigma * 2.5
            x = np.linspace(_each.min() - pad, _each.max() + pad, 100)
            y1 = kde(x)
            y0 = np.full_like(y1, bottom)
            input_.append(XYYData(x, y0, y1))
        return cls(input_, name=name, orient=orient.transpose(), backend=backend)


class ViolinPlot(BandCollection):
    def __init__(
        self,
        data: list[XYYData],
        *,
        name: str | None = None,
        shape: Literal["both", "left", "right"] = "both",
        extent: float = 0.5,
        orient: Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ):
        super().__init__(data, name=name, orient=orient, backend=backend)
        self._shape = shape
        self._extent = extent

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
        backend: str | Backend | None = None,
    ):
        from whitecanvas.utils.kde import gaussian_kde

        ori = Orientation.parse(orient)
        if extent <= 0:
            raise ValueError(f"extent must be positive, got {extent}")
        x, data = check_array_input(x, data)
        xyy_values: list[XYYData] = []
        for offset, values in zip(x, data):
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

            data = XYYData(x_, y0, y1)
            xyy_values.append(data)

        half_widths: list[float] = []
        for xyy in xyy_values:
            half_width = np.max(np.abs(xyy.ydiff))
            if shape == "both":
                half_width /= 2
            half_widths.append(half_width)
        factor = extent / np.max(half_widths) / 2
        new_vals: list[XYYData] = []
        for xyy, xoffset in zip(xyy_values, x):
            y0 = (xyy.y0 - xoffset) * factor + xoffset
            y1 = (xyy.y1 - xoffset) * factor + xoffset
            new_vals.append(XYYData(xyy.x, y0, y1))
        return cls(
            new_vals,
            name=name,
            shape=shape,
            extent=extent,
            orient=ori.transpose(),
            backend=backend,
        )

    @property
    def offsets(self) -> NDArray[np.floating]:
        if self._shape == "both":

            def _getter(x: XYYData):
                return (x.y0[0] + x.y0[1]) / 2

        elif self._shape == "left":

            def _getter(x: XYYData):
                return x.y1[0]

        elif self._shape == "right":

            def _getter(x: XYYData):
                return x.y0[0]

        else:
            raise ValueError(self._shape)
        return np.array([_getter(band.data) for band in self])

    @property
    def orient(self) -> Orientation:
        return self._orient

    @property
    def ndata(self) -> int:
        return len(self)

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, width: float):
        if width <= 0:
            raise ValueError(f"extent must be positive, got {width}")
        factor = width / self.extent
        for band in self:
            bd = band.data
            ycenter = bd.ycenter
            y0 = (bd.y0 - ycenter) * factor + ycenter
            y1 = (bd.y1 - ycenter) * factor + ycenter
            band.set_data(bd.x, y0, y1)
        self._extent = width

    def set_datasets(
        self,
        offsets: list[float] | None = None,
        dataset: list[np.ndarray] | None = None,
        kde_band_width: float | str = "scott",
    ):
        from whitecanvas.utils.kde import gaussian_kde

        if offsets is None:
            _offsets = self.offsets
        else:
            _offsets = offsets
        if dataset is None:
            raise NotImplementedError
        if len(offsets) != len(dataset):
            raise ValueError("Length mismatch.")
        xyy_values: list[XYYData] = []
        for offset, values in zip(_offsets, dataset):
            arr = as_array_1d(values)
            kde = gaussian_kde(arr, bw_method=kde_band_width)

            sigma = np.sqrt(kde.covariance[0, 0])
            pad = sigma * 2.5
            x_ = np.linspace(arr.min() - pad, arr.max() + pad, 100)
            y = kde(x_)
            if self._shape in ("both", "left"):
                y0 = -y + offset
            else:
                y0 = np.zeros_like(y) + offset
            if self._shape in ("both", "right"):
                y1 = y + offset
            else:
                y1 = np.zeros_like(y) + offset

            data = XYYData(x_, y0, y1)
            xyy_values.append(data)

        half_widths: list[float] = []
        for xyy in xyy_values:
            half_width = np.max(np.abs(xyy.ydiff))
            if self._shape == "both":
                half_width /= 2
            half_widths.append(half_width)
        factor = self.extent / np.max(half_widths) / 2
        for xyy, xoffset, band in zip(xyy_values, _offsets, self):
            y0 = (xyy.y0 - xoffset) * factor + xoffset
            y1 = (xyy.y1 - xoffset) * factor + xoffset
            band.data = XYYData(xyy.x, y0, y1)
