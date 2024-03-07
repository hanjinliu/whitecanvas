from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from cmap import Colormap
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers._base import HoverableDataBoundLayer
from whitecanvas.layers._primitive.line import (
    MultiLine,
    MultiLineEvents,
    MultiLineProtocol,
)
from whitecanvas.types import (
    ArrayLike1D,
    ColormapType,
    ColorType,
    KdeBandWidthType,
    LineStyle,
    Orientation,
    OrientationLike,
    XYYData,
)
from whitecanvas.utils.normalize import as_array_1d
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self


class Rug(MultiLine, HoverableDataBoundLayer[MultiLineProtocol, NDArray[np.number]]):
    """
    Rug plot (event plot) layer.

      │ ││  │   │
    ──┴─┴┴──┴───┴──>
    """

    _ATTACH_TO_AXIS = True
    events: MultiLineEvents
    _events_class = MultiLineEvents

    def __init__(
        self,
        events: ArrayLike1D,
        *,
        low: ArrayLike1D | float = 0.0,
        high: ArrayLike1D | float = 1.0,
        name: str | None = None,
        color: ColorType = "black",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
        antialias: bool = True,
        orient: OrientationLike = "vertical",
        backend: str | Backend | None = None,
    ):
        events, segs, orient = _norm_input(events, low, high, orient)
        super().__init__(
            segs, name=name, color=color, alpha=alpha, width=width, style=style,
            antialias=antialias, backend=backend
        )  # fmt: skip
        self._orient = Orientation.parse(orient)

    def _get_layer_data(self) -> NDArray[np.number]:
        segs = super()._get_layer_data()
        idx = 0 if self.orient.is_vertical else 1
        return np.array([seg[0, idx] for seg in segs])

    def _norm_layer_data(self, data: Any) -> NDArray[np.number]:
        return as_array_1d(data)

    def _set_layer_data(self, data: NDArray[np.number]):
        _, segs, _ = _norm_input(data, self.low, self.high, self.orient)
        return super()._set_layer_data(segs)

    @property
    def low(self) -> NDArray[np.number]:
        """Coordinate of the lower bound."""
        return self.data_full.y0

    @low.setter
    def low(self, val):
        _, segs, _ = _norm_input(self.data, val, self.high, self.orient)
        super()._set_layer_data(segs)
        self.events.data.emit(segs)

    @property
    def high(self) -> NDArray[np.number]:
        """Coordinate of the higher bound."""
        return self.data_full.y1

    @high.setter
    def high(self, val):
        _, segs, _ = _norm_input(self.data, self.low, val, self.orient)
        super()._set_layer_data(segs)
        self.events.data.emit(segs)

    @property
    def data_full(self) -> XYYData:
        segs = MultiLine._get_layer_data(self)
        seg_stack = np.stack(segs, axis=2)
        idx = 1 if self.orient.is_vertical else 0
        low = seg_stack[0, idx, :]
        high = seg_stack[1, idx, :]
        _t = seg_stack[0, 1 - idx, :]
        return XYYData(x=_t, y0=low, y1=high)

    @data_full.setter
    def data_full(self, val: XYYData):
        val = XYYData(*val)
        _, segs, _ = _norm_input(val.x, val.y0, val.y1, self.orient)
        super()._set_layer_data(segs)
        self.events.data.emit(segs)

    def set_data(self, events: ArrayLike1D):
        self.data = events

    @property
    def orient(self) -> Orientation:
        """Orientation of the rug plot."""
        return self._orient

    def update_length(
        self,
        lengths: float | NDArray[np.number],
        *,
        offset: float | None = None,
        align: str = "low",
    ) -> Self:
        """
        Update the length of the rug lines.

        Parameters
        ----------
        lengths : float or array-like
            Length of the rug lines. If a scalar, all the lines have the same length.
            If an array, each line has a different length.
        offset : float, optional
            Offset of the lines. If not given, the mean of the lower and upper bounds is
            used.
        align : {'low', 'high', 'center'}, optional
            How to align the rug lines around the offset. This parameter is defined as
            follows.

            ```
               "low"     "high"    "center"
              ──┴─┴──   ──┬─┬──    ──┼─┼──
            ```
        """
        data_full = self.data_full
        if offset is None:
            offset = data_full.y0.mean()
        if not is_real_number(lengths):
            lengths = as_array_1d(lengths)
            if lengths.size != data_full.x.size:
                raise ValueError(
                    "`lengths` must be a scalar or an array with the same size as "
                    f"events, but got {lengths.size} and {data_full.x.size} events."
                )

        if align == "low":
            y0 = np.full((data_full.x.size,), offset)
            y1 = offset + lengths
        elif align == "high":
            y0 = offset - lengths
            y1 = np.full((data_full.x.size,), offset)
        elif align == "center":
            y0 = offset - lengths / 2
            y1 = offset + lengths / 2
        else:
            raise ValueError(
                f"`align` must be 'low', 'high', or 'center', got {align!r}."
            )
        self.data_full = XYYData(data_full.x, y0, y1)
        return self

    def color_by_density(
        self,
        cmap: ColormapType = "jet",
        *,
        band_width: KdeBandWidthType = "scott",
    ) -> Self:
        """
        Set the color of the markers by density.

        Parameters
        ----------
        cmap : ColormapType, optional
            Colormap used to map the density to colors.
        band_width : float, "scott" or "silverman", optional
            Method to calculate the estimator bandwidth.
        """
        from whitecanvas.utils.kde import gaussian_kde

        events = self.data
        density = gaussian_kde(events, band_width)(events)
        normed = density / density.max()
        self.color = Colormap(cmap)(normed)
        return self

    def scale_by_density(
        self,
        max_length: float = 1.0,
        *,
        offset: float | None = None,
        align: str = "low",
        band_width: KdeBandWidthType = "scott",
    ) -> Self:
        """
        Set the height of the lines by density.

        Parameters
        ----------
        max_length : float, optional
            Maximum length of the lines.
        offset : float, optional
            Offset of the lines. If not given, the mean of the lower and upper bounds is
            used.
        align : {'low', 'high', 'center'}, optional
            How to align the rug lines around the offset. This parameter is defined as
            follows.

            ```
               "low"     "high"    "center"
              ──┴─┴──   ──┬─┬──    ──┼─┼──
            ```
        band_width : float, "scott" or "silverman", optional
            Method to calculate the estimator bandwidth.
        """
        from whitecanvas.utils.kde import gaussian_kde

        events = self.data
        density = gaussian_kde(events, band_width)(events)
        normed = density / density.max() * max_length
        return self.update_length(normed, offset=offset, align=align)


def _norm_input(
    events: ArrayLike1D,
    low: ArrayLike1D | float,
    high: ArrayLike1D | float,
    orient: OrientationLike,
) -> tuple[NDArray[np.number], NDArray[np.number], Orientation]:
    _t = as_array_1d(events)
    if is_real_number(low):
        y0 = np.full(_t.shape, low)
    else:
        y0 = as_array_1d(low)
        if y0.size != _t.size:
            raise ValueError(
                "`low` must be a scalar or an array with the same size as events, but "
                f"got {y0.size} and {events.size} events."
            )
    if is_real_number(high):
        y1 = np.full(_t.shape, high)
    else:
        y1 = as_array_1d(high)
        if y1.size != _t.size:
            raise ValueError(
                "`high` must be a scalar or an array with the same size as events, but "
                f"got {y1.size} and {events.size} events."
            )
    ori = Orientation.parse(orient)
    if ori.is_vertical:
        start = np.stack([_t, y0], axis=1)
        stop = np.stack([_t, y1], axis=1)
        segs = np.stack([start, stop], axis=0)
    else:
        start = np.stack([y0, _t], axis=1)
        stop = np.stack([y1, _t], axis=1)
        segs = np.stack([start, stop], axis=0)
    return events, np.moveaxis(segs, 1, 0), orient
