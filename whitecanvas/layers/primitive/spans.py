from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.protocols import BarProtocol
from whitecanvas.layers._mixin import HeteroFaceEdgeMixin
from whitecanvas.types import ColorType, FacePattern, Orientation
from whitecanvas.backend import Backend

if TYPE_CHECKING:
    from whitecanvas.canvas import Canvas


class Spans(HeteroFaceEdgeMixin[BarProtocol]):
    """
    Layer that represents vertical/hosizontal spans.

       |///|      |///////////|
       |///|      |///////////|
    ──────────────────────────────>
       |///|      |///////////|
       |///|      |///////////|
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
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: Backend | str | None = None,
    ):
        _spans = _norm_data(spans)
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
        self._backend = self._create_backend(Backend(backend), *xxyy)
        self.name = name if name is not None else self.__class__.__name__
        self._orient = ori
        self.face.update(color=color, alpha=alpha, pattern=pattern)
        self._x_hint, self._y_hint = xhint, yhint

    @property
    def orient(self) -> Orientation:
        return self._orient

    @property
    def data(self) -> NDArray[np.float_]:
        x0, x1, y0, y1 = self._backend._plt_get_data()
        if self.orient.is_vertical:
            return np.column_stack([x0, x1])
        else:
            return np.column_stack([y0, y1])

    def set_data(self, spans: ArrayLike):
        _old_spans = self.data
        _spans = _norm_data(spans)
        if self.orient.is_vertical:
            xxyy = _spans[:, 0], _spans[:, 1], _old_spans[:, 2], _old_spans[:, 3]
            self._x_hint = _spans.min(), _spans.max()
        else:
            xxyy = _old_spans[:, 0], _old_spans[:, 1], _spans[:, 0], _spans[:, 1]
            self._y_hint = _spans.min(), _spans.max()
        self._backend._plt_set_data(*xxyy)

    @property
    def ndata(self) -> int:
        """The number of data points"""
        return self._backend._plt_get_data()[0].size

    def _connect_canvas(self, canvas: Canvas):
        canvas.y.lim_changed.connect(self._recalculate_spans)
        return super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        canvas.x.lim_changed.connect(self._recalculate_spans)
        return super()._disconnect_canvas(canvas)

    def _recalculate_spans(self, lim: tuple[float, float]):
        _min, _max = lim
        x0, x1, y0, y1 = self._backend._plt_get_data()
        _min_arr = np.full_like(x0, _min)
        _max_arr = np.full_like(x0, _max)
        if self.orient.is_vertical:
            spans = x0, x1, _min_arr, _max_arr
        else:
            spans = _min_arr, _max_arr, x0, x1
        self._backend._plt_set_data(*spans)


def _norm_data(spans):
    _spans = np.asarray(spans)
    if _spans.ndim != 2:
        raise ValueError(f"spans must be 2-dimensional, got {_spans.ndim}")
    if _spans.shape[1] != 2:
        raise ValueError(f"spans must be (N, 2), got {_spans.shape}")
    if _spans.dtype.kind not in "uif":
        raise ValueError(f"spans must be numeric, got {_spans.dtype}")
    return _spans
