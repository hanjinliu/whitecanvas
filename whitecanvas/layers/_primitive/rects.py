from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend
from whitecanvas.layers._base import HoverableDataBoundLayer
from whitecanvas.layers._mixin import (
    ConstEdge,
    ConstFace,
    EdgeNamespace,
    FaceEdgeMixinEvents,
    FaceNamespace,
    MultiEdge,
    MultiFace,
    MultiFaceEdgeMixin,
)
from whitecanvas.protocols import BarProtocol
from whitecanvas.types import ColorType, Hatch, LineStyle, Rect, _Void
from whitecanvas.utils.normalize import parse_texts

if TYPE_CHECKING:
    from typing_extensions import Self

_void = _Void()
_Face = TypeVar("_Face", bound=FaceNamespace)
_Edge = TypeVar("_Edge", bound=EdgeNamespace)


class RectsEvents(FaceEdgeMixinEvents):
    clicked = Signal(int)


class Rects(
    HoverableDataBoundLayer[BarProtocol, NDArray[np.float32]],
    MultiFaceEdgeMixin[_Face, _Edge],
):
    """
    Layer that represents Rectangles.

    Attributes
    ----------
    face : `whitecanvas.layers._mixin.FaceNamespace`
        Face properties of the bars.
    edge : `whitecanvas.layers._mixin.EdgeNamespace`
        Edge properties of the bars.
    """

    _backend_class_name = "Bars"
    events: RectsEvents
    _events_class = RectsEvents

    def __init__(
        self,
        coords: NDArray[np.number],
        *,
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float | _Void = _void,
        hatch: str | Hatch = Hatch.SOLID,
        backend: Backend | str | None = None,
    ):
        MultiFaceEdgeMixin.__init__(self)
        xxyy, xhint, yhint = _norm_inputs(coords)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), *xxyy)
        self.face.update(color=color, alpha=alpha, hatch=hatch)
        self._x_hint, self._y_hint = xhint, yhint
        self._init_events()
        self._backend._plt_connect_pick_event(self.events.clicked.emit)

    def _get_layer_data(self) -> NDArray[np.float32]:
        """Current data of the layer."""
        x0, x1, y0, y1 = self._backend._plt_get_data()
        return np.column_stack([x0, x1, y0, y1])

    def _norm_layer_data(self, data: ArrayLike) -> NDArray[np.float32]:
        arr = np.atleast_2d(data).astype(np.float32, copy=False)
        if arr.size > 0 and arr.shape[1] != 4:
            raise ValueError("Data must have 4 columns")
        return arr

    def _set_layer_data(self, data: NDArray[np.float32]):
        self._backend._plt_set_data(*data.T)
        _, self._x_hint, self._y_hint = _norm_inputs(data)

    @property
    def rects(self) -> list[Rect]:
        """Return the data as a list of `Rect` objects."""
        return [Rect.with_sort(*xy) for xy in self._get_layer_data()]

    @property
    def ndata(self) -> int:
        """The number of data points"""
        return self.data.shape[0]

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a Rects from a dictionary."""
        return cls(
            d["data"], name=d["name"], color=d["face"]["color"],
            hatch=d["face"]["hatch"], backend=backend,
        ).with_edge(
            color=d["edge"]["color"], width=d["edge"]["width"], style=d["edge"]["style"]
        )  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the layer."""
        return {
            "type": "rects",
            "data": self.data,
            "name": self.name,
            "face": self.face.to_dict(),
            "edge": self.edge.to_dict(),
        }

    def with_face(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1,
    ) -> Rects[ConstFace, _Edge]:
        return super().with_face(color=color, hatch=hatch, alpha=alpha)

    def with_face_multi(
        self,
        *,
        color: ColorType | Sequence[ColorType] | _Void = _void,
        hatch: str | Hatch | Sequence[str | Hatch] | _Void = _void,
        alpha: float = 1,
    ) -> Rects[MultiFace, _Edge]:
        return super().with_face_multi(color=color, hatch=hatch, alpha=alpha)

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Rects[_Face, ConstEdge]:
        return super().with_edge(color=color, width=width, style=style, alpha=alpha)

    def with_edge_multi(
        self,
        *,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Rects[_Face, MultiEdge]:
        return super().with_edge_multi(
            color=color, width=width, style=style, alpha=alpha
        )

    def as_edge_only(
        self,
        *,
        width: float = 3.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Self:
        """
        Convert the rectangles to edge-only mode.

        This method will set the face color to transparent and the edge color to the
        current face color.

        Parameters
        ----------
        width : float, default 3.0
            Width of the edge.
        style : str or LineStyle, default LineStyle.SOLID
            Line style of the edge.
        """
        color = self.face.color
        if color.ndim == 0:
            pass
        elif color.ndim == 1:
            self.with_edge(color=color, width=width, style=style)
        elif color.ndim == 2:
            self.with_edge_multi(color=color, width=width, style=style)
        else:
            raise RuntimeError("Unreachable error.")
        self.face.update(alpha=0.0)
        return self

    def with_hover_template(self, template: str, extra: Any | None = None) -> Rects:
        coords = self.data
        if self._backend_name in ("plotly", "bokeh"):  # conversion for HTML
            template = template.replace("\n", "<br>")
        params = parse_texts(template, coords.shape[0], extra)
        # set default format keys
        params.setdefault("left", coords[:, 0])
        params.setdefault("right", coords[:, 1])
        params.setdefault("bottom", coords[:, 2])
        params.setdefault("top", coords[:, 3])
        if "i" not in params:
            params["i"] = np.arange(coords.shape[0])
        texts = [
            template.format(**{k: v[i] for k, v in params.items()})
            for i in range(coords.shape[0])
        ]
        self._backend._plt_set_hover_text(texts)
        return self

    def _as_legend_item(self) -> _legend.BarLegendItem:
        return _legend.BarLegendItem(
            self.face._as_legend_info(), self.edge._as_legend_info()
        )


def _norm_inputs(coords: ArrayLike):
    arr = np.atleast_2d(coords).astype(np.float32, copy=False)
    if arr.size == 0:
        return (np.zeros(0, dtype=np.float32),) * 4, None, None
    if arr.shape[1] != 4:
        raise ValueError("Data must have 4 columns")
    x0, x1, y0, y1 = arr.T
    if x0.size > 0:
        xhint = np.min(x0), np.max(x1)
        yhint = np.min(y0), np.max(y1)
    else:
        xhint = None
        yhint = None
    return (x0, x1, y0, y1), xhint, yhint
