from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.backend import Backend
from whitecanvas.layers._mixin import CollectionFaceEdgeMixin
from whitecanvas.layers._primitive import Band
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.layers.group._collections import (
    LayerCollectionBase,
    RichContainerEvents,
)
from whitecanvas.types import Orientation, XYYData
from whitecanvas.utils.normalize import as_array_1d, parse_texts


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

    def with_hover_text(self, text: str | Iterable[Any]):
        """Set hover text for each band."""
        if isinstance(text, str):
            texts = [text] * len(self)
        else:
            texts = [str(t) for t in text]
        for band, t in zip(self, texts):
            band.with_hover_text(t)
        return self

    def with_hover_template(self, template: str, extra: dict[str, Any] | None = None):
        if self._backend_name in ("plotly", "bokeh"):  # conversion for HTML
            template = template.replace("\n", "<br>")
        params = parse_texts(template, len(self), extra)
        # set default format keys
        if "i" not in params:
            params["i"] = np.arange(len(self))
        texts = [
            template.format(**{k: v[i] for k, v in params.items()})
            for i in range(len(self))
        ]
        return self.with_hover_text(texts)


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
        ori = Orientation.parse(orient)
        new_vals = cls._convert_data(
            x, data, shape=shape, extent=extent, kde_band_width=kde_band_width
        )
        return cls(
            new_vals,
            name=name,
            shape=shape,
            extent=extent,
            orient=ori.transpose(),
            backend=backend,
        )

    @property
    def orient(self) -> Orientation:
        """Orientation of the violin plot (perpendicular to the fill orientation)."""
        return self._orient.transpose()

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

    @staticmethod
    def _convert_data(
        x: list[float],
        data: list[ArrayLike],
        shape: Literal["both", "left", "right"] = "both",
        extent: float = 0.5,
        kde_band_width: float | str = "scott",
    ):
        from whitecanvas.utils.kde import gaussian_kde

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
        return new_vals
