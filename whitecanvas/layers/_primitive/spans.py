from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, Sequence, Generic
import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.protocols import BarProtocol
from whitecanvas.layers._base import DataBoundLayer
from whitecanvas.layers._mixin import (
    MultiFaceEdgeMixin,
    FaceNamespace,
    EdgeNamespace,
    ConstFace,
    ConstEdge,
    MultiFace,
    MultiEdge,
)
from whitecanvas.types import ColorType, Hatch, Orientation, LineStyle, Rect
from whitecanvas.backend import Backend

if TYPE_CHECKING:
    from whitecanvas.canvas import Canvas

_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)


class Spans(
    DataBoundLayer[BarProtocol, NDArray[np.number]],
    MultiFaceEdgeMixin[_Face, _Edge],
    Generic[_Face, _Edge],
):
    """
    Layer that represents vertical/hosizontal spans.

    Attributes
    ----------
    face : :class:`~whitecanvas.layers._mixin.FaceNamespace`
        Face properties of the spans.
    edge : :class:`~whitecanvas.layers._mixin.EdgeNamespace`
        Edge properties of the spans.
    """

    _backend_class_name = "Bars"

    def __init__(
        self,
        spans: ArrayLike,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
        MultiFaceEdgeMixin.__init__(self)
        _spans = self._norm_layer_data(spans)
        ori = Orientation.parse(orient)
        nspans = _spans.shape[0]
        if ori.is_vertical:
            xxyy = _spans[:, 0], _spans[:, 1], np.zeros(nspans), np.ones(nspans)
            xhint = _spans.min(), _spans.max()
            yhint = None
        else:
            xxyy = np.zeros(nspans), np.ones(nspans), _spans[:, 0], _spans[:, 1]
            xhint = None
            yhint = _spans.min(), _spans.max()
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), *xxyy)
        self._orient = ori
        self.face.update(color=color, alpha=alpha, hatch=hatch)
        self._x_hint, self._y_hint = xhint, yhint
        self._low_lim = -1e10
        self._high_lim = 1e10

    @property
    def orient(self) -> Orientation:
        """Orientation of the spans."""
        return self._orient

    def _get_layer_data(self) -> NDArray[np.number]:
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self.orient.is_vertical:
            return np.column_stack([x0, x1])
        else:
            return np.column_stack([y0, y1])

    def _norm_layer_data(self, data: Any) -> NDArray[np.number]:
        _spans = np.asarray(data)
        if _spans.ndim != 2:
            raise ValueError(f"spans must be 2-dimensional, got {_spans.ndim}")
        if _spans.shape[1] != 2:
            raise ValueError(f"spans must be (N, 2), got {_spans.shape}")
        if _spans.dtype.kind not in "uif":
            raise ValueError(f"spans must be numeric, got {_spans.dtype}")
        return _spans

    def _set_layer_data(self, data: NDArray[np.number]):
        _low = np.full_like(data[:, 0], self._low_lim)
        _high = np.full_like(data[:, 0], self._high_lim)
        if self.orient.is_vertical:
            xxyy = data[:, 0], data[:, 1], _low, _high
            self._x_hint = data.min(), data.max()
        else:
            xxyy = _low, _high, data[:, 0], data[:, 1]
            self._y_hint = data.min(), data.max()
        self._backend._plt_set_data(*xxyy)

    def set_data(self, spans: ArrayLike):
        self.data = spans

    @property
    def ndata(self) -> int:
        """The number of data points"""
        return self._backend._plt_get_data()[0].size

    def _connect_canvas(self, canvas: Canvas):
        canvas.events.lims.connect(self._recalculate_spans)
        return super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        canvas.events.lims.connect(self._recalculate_spans)
        return super()._disconnect_canvas(canvas)

    def _recalculate_spans(self, rect: Rect):
        # update the rectangles so that their limits are not visible
        if self.orient.is_vertical:
            _min, _max = rect.bottom, rect.top
        else:
            _min, _max = rect.left, rect.right
        if _min > _max:
            _min, _max = _max, _min
        if _min >= self._low_lim and _max <= self._high_lim:
            return
        self._low_lim = _min - 1e10
        self._high_lim = _max + 1e10
        x0, x1, y0, y1 = self._backend._plt_get_data()
        _min_arr = np.full_like(x0, self._low_lim)
        _max_arr = np.full_like(x0, self._high_lim)
        if self.orient.is_vertical:
            spans = x0, x1, _min_arr, _max_arr
        else:
            spans = _min_arr, _max_arr, y0, y1
        self._backend._plt_set_data(*spans)

    def with_face(
        self,
        color: ColorType | None = None,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1,
    ) -> Spans[ConstFace, _Edge]:
        return super().with_face(color, hatch, alpha)

    def with_face_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        hatch: str | Hatch | Sequence[str | Hatch] = Hatch.SOLID,
        alpha: float = 1,
    ) -> Spans[MultiFace, _Edge]:
        return super().with_face_multi(color, hatch, alpha)

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Spans[_Face, ConstEdge]:
        return super().with_edge(color, width, style, alpha)

    def with_edge_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Spans[_Face, MultiEdge]:
        return super().with_edge_multi(color, width, style, alpha)
