from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.layers._mixin import FaceEdgeMixin, FaceEdgeMixinEvents
from whitecanvas.layers._sizehint import xyy_size_hint
from whitecanvas.protocols import BandProtocol
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    Hatch,
    Orientation,
    OrientationLike,
    XYYData,
    _Void,
)
from whitecanvas.utils.normalize import as_array_1d

if TYPE_CHECKING:
    from typing_extensions import Self


class BandEvents(FaceEdgeMixinEvents):
    clicked = Signal()


_void = _Void()


class Band(DataBoundLayer[BandProtocol, XYYData], FaceEdgeMixin):
    events: BandEvents
    _events_class = BandEvents

    def __init__(
        self,
        t: ArrayLike1D,
        edge_low: ArrayLike1D,
        edge_high: ArrayLike1D,
        orient: OrientationLike = "vertical",
        *,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float | _Void = _void,
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
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
        FaceEdgeMixin.__init__(self)
        self._backend = self._create_backend(Backend(backend), x, y0, y1, ori)
        self._orient = ori
        self.face.update(color=color, alpha=alpha, hatch=hatch)
        self._x_hint, self._y_hint = xyy_size_hint(x, y0, y1, ori)
        self._band_type = "band"
        self.edge.width = 0.0
        self._init_events()
        self._backend._plt_connect_pick_event(self.events.clicked.emit)

    @property
    def orient(self) -> Orientation:
        """Orientation of the band."""
        return self._orient

    def set_data(
        self,
        t: ArrayLike1D | None = None,
        edge_low: ArrayLike1D | None = None,
        edge_high: ArrayLike1D | None = None,
    ):
        self.data = t, edge_low, edge_high

    def with_hover_text(self, text: str) -> Self:
        """Add hover text to the data points."""
        self._backend._plt_set_hover_text(str(text))
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a Band from a dictionary."""
        return cls(
            d["data"]["x"], d["data"]["y0"], d["data"]["y1"], orient=d["orient"],
            name=d["name"], color=d["face"]["color"],
            hatch=d["face"]["hatch"], backend=backend,
        ).with_edge(
            color=d["edge"]["color"], width=d["edge"]["width"], style=d["edge"]["style"]
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "data": self._get_layer_data().to_dict(),
            "orient": self.orient.value,
            "name": self.name,
            "visible": self.visible,
            "face": self.face.to_dict(),
            "edge": self.edge.to_dict(),
        }

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

    def _as_legend_item(self) -> _legend.BarLegendItem:
        face = _legend.FaceInfo(self.face.color, self.face.hatch)
        edge = _legend.EdgeInfo(self.edge.color, self.edge.width, self.edge.style)
        return _legend.BarLegendItem(face, edge)
