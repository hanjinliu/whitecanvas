from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers._mixin import EdgeNamespace, FaceNamespace, MonoEdge, MonoFace
from whitecanvas.layers._primitive import Markers, MultiLine
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.types import ArrayLike1D, Orientation, XYData
from whitecanvas.utils.normalize import normalize_xy

if TYPE_CHECKING:
    from typing_extensions import Self

_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)
_Size = TypeVar("_Size", float, NDArray[np.floating])


class StemPlot(LayerContainer, Generic[_Face, _Edge, _Size]):
    _ATTACH_TO_AXIS = True

    def __init__(
        self,
        markers: Markers[_Face, _Edge, _Size],
        lines: MultiLine,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
    ):
        self._orient = Orientation.parse(orient)
        super().__init__([markers, lines], name=name)

    def _default_ordering(self, n: int) -> list[int]:
        assert n == 2
        return [1, 0]

    @property
    def data(self) -> XYData:
        """XYData as (x, height)."""
        mdata = self.markers.data
        if self.orient.is_vertical:
            xdata = mdata.x
            ydata = mdata.y - self.bottom
        else:
            xdata = mdata.x - self.bottom
            ydata = mdata.y
        return XYData(xdata, ydata)

    @property
    def bottom(self) -> NDArray[np.floating]:
        """Bottom of the stem."""
        if self.orient.is_vertical:
            return np.array([d[0, 1] for d in self.lines.data])
        else:
            return np.array([d[0, 0] for d in self.lines.data])

    @property
    def top(self) -> NDArray[np.floating]:
        """Top of the stem."""
        if self.orient.is_vertical:
            return self.markers.data.y
        else:
            return self.markers.data.x

    @property
    def markers(self) -> Markers[_Face, _Edge, _Size]:
        """Markers layer."""
        return self._children[0]

    @property
    def lines(self) -> MultiLine:
        """Lines for the stems."""
        return self._children[1]

    @property
    def orient(self) -> Orientation:
        """Orientation of the stem plot."""
        return self._orient

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        self = super().from_dict(d, backend=backend)
        self._orient = Orientation.parse(d["orient"])
        return self

    def bbox_hint(self) -> NDArray[np.float64]:
        xmin, xmax, ymin, ymax = self.markers.bbox_hint()
        # NOTE: min/max is nan-safe (returns nan)
        if self.orient.is_vertical:
            ymin = min(ymin, 0)
            ymax = max(ymax, 0)
        else:
            xmin = min(xmin, 0)
            xmax = max(xmax, 0)
        return xmin, xmax, ymin, ymax

    @classmethod
    def from_arrays(
        cls,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        bottom: ArrayLike1D | float = 0,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> StemPlot[MonoFace, MonoEdge, float]:
        xdata, ydata = normalize_xy(xdata, ydata)
        top = ydata + bottom
        return Markers(xdata, top, name=name, backend=backend).with_stem(
            orient, bottom=bottom
        )

    def _as_legend_item(self) -> _legend.StemLegendItem:
        return _legend.StemLegendItem(
            self.lines._as_legend_item(), self.markers._as_legend_item()
        )
