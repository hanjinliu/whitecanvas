from __future__ import annotations

from typing import Iterable

import numpy as np
from cmap import Color
from numpy.typing import ArrayLike, NDArray
from psygnal import Signal

from whitecanvas.backend import Backend
from whitecanvas.layers._mixin import (
    EdgeNamespace,
    EnumArray,
    FaceNamespace,
    MonoEdge,
    MultiFace,
    _AbstractFaceEdgeMixin,
)
from whitecanvas.layers._primitive import Bars, MultiLine
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.layers.group._collections import LayerContainer, LayerContainerEvents
from whitecanvas.theme import get_theme
from whitecanvas.types import ColorType, Hatch, LineStyle, Orientation, _Void
from whitecanvas.utils.normalize import as_any_1d_array, as_color_array

_void = _Void()


class BoxPlotEvents(LayerContainerEvents):
    face = Signal(object)
    edge = Signal(object)


class BoxPlot(LayerContainer, _AbstractFaceEdgeMixin["BoxFace", "BoxEdge"]):
    """
    A group for boxplot.

    Children layers are:
    - Bars (boxes)
    - MultiLine (whiskers)
    - MultiLine (median line)
    - Markers (outliers)

     ──┬──  <-- max
       │
    ┌──┴──┐ <-- 75% quantile
    │  o  │ <-- mean
    ╞═════╡ <-- median
    └──┬──┘ <-- 25% quantile
       │
     ──┴──  <-- min
    """

    events: BoxPlotEvents
    _events_class = BoxPlotEvents

    def __init__(
        self,
        boxes: Bars[MultiFace, MonoEdge],
        whiskers: MultiLine,
        medians: MultiLine,
        # outliers: Markers | None = None,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
    ):
        super().__init__([boxes, whiskers, medians], name=name)
        _AbstractFaceEdgeMixin.__init__(self, BoxFace(self), BoxEdge(self))
        self._orient = Orientation.parse(orient)
        self._init_events()

    @property
    def boxes(self) -> Bars[MultiFace, MonoEdge]:
        """The boxes layer (Bars)."""
        return self._children[0]

    @property
    def whiskers(self) -> MultiLine:
        """The whiskers layer (MultiLine)."""
        return self._children[1]

    @property
    def medians(self) -> MultiLine:
        """The median line layer (MultiLine)."""
        return self._children[2]

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike],
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.3,
        capsize: float = 0.15,
        color: ColorType | list[ColorType] = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        backend: str | Backend | None = None,
    ):
        x, data = check_array_input(x, data)
        ori = Orientation.parse(orient)
        color = as_color_array(color, len(x))
        agg_values: list[NDArray[np.number]] = []
        for d in data:
            agg_values.append(np.quantile(d, [0, 0.25, 0.5, 0.75, 1]))
        agg_arr = np.stack(agg_values, axis=1)
        box = Bars(
            x, agg_arr[3] - agg_arr[1], agg_arr[1], name=name, orient=ori,
            bar_width=extent, backend=backend,
        ).with_face_multi(
            hatch=hatch, color=color, alpha=alpha,
        ).with_edge(color="black")  # fmt: skip
        if ori.is_vertical:
            segs = _xyy_to_segments(
                x, agg_arr[0], agg_arr[1], agg_arr[3], agg_arr[4], capsize
            )
            medsegs = [
                [(x0 - extent / 2, y0), (x0 + extent / 2, y0)]
                for x0, y0 in zip(x, agg_arr[2])
            ]
        else:
            segs = _yxx_to_segments(
                x, agg_arr[0], agg_arr[1], agg_arr[3], agg_arr[4], capsize
            )
            medsegs = [
                [(x0, y0 - extent / 2), (x0, y0 + extent / 2)]
                for x0, y0 in zip(x, agg_arr[2])
            ]
        whiskers = MultiLine(
            segs, name=name, style=LineStyle.SOLID, alpha=alpha, backend=backend,
            color="black",
        )  # fmt: skip
        medians = MultiLine(
            medsegs, name="medians", color="black", alpha=alpha, backend=backend,
        )  # fmt: skip

        return cls(box, whiskers, medians, name=name, orient=ori)

    @property
    def orient(self) -> Orientation:
        """Orientation of the boxplot."""
        return self._orient

    def with_shift(self, shift: float) -> BoxPlot:
        self.boxes.set_data(xdata=self.boxes.data.x + shift)
        if self.orient.is_vertical:
            _wdata = []
            for seg in self.whiskers.data:
                _wdata.append([seg[:, 0] + shift, seg[:, 1]])
            self.whiskers.data = _wdata
            _mdata = []
            for seg in self.medians.data:
                _mdata.append([seg[:, 0] + shift, seg[:, 1]])
            self.medians.data = _mdata
        else:
            _wdata = []
            for seg in self.whiskers.data:
                _wdata.append([seg[:, 0], seg[:, 1] + shift])
            self.whiskers.data = _wdata
            _mdata = []
            for seg in self.medians.data:
                _mdata.append([seg[:, 0], seg[:, 1] + shift])
            self.medians.data = _mdata
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def _make_sure_hatch_visible(self):
        _is_no_width = self.edge.width == 0
        if np.any(_is_no_width):
            ec = get_theme().foreground_color
            self.edge.width = np.where(_is_no_width, 1, self.edge.width)
            self.edge.color = np.where(_is_no_width, ec, self.edge.color)


def _xyy_to_segments(
    x: ArrayLike,
    y0: ArrayLike,
    y1: ArrayLike,
    y2: ArrayLike,
    y3: ArrayLike,
    capsize: float,
):
    """
    ──┬──  <-- y3
      │    <-- y2

      │    <-- y1
    ──┴──  <-- y0
      ↑
      x
    """
    v0 = np.stack([x, y0], axis=1)
    v1 = np.stack([x, y1], axis=1)
    v2 = np.stack([x, y2], axis=1)
    v3 = np.stack([x, y3], axis=1)
    segments_0 = [[s0, s1] for s0, s1 in zip(v0, v1)]
    segments_1 = [[s2, s3] for s2, s3 in zip(v2, v3)]
    if capsize > 0:
        _c = np.array([capsize / 2, 0])
        cap0 = [[s0 - _c, s0 + _c] for s0 in v0]
        cap1 = [[s3 - _c, s3 + _c] for s3 in v3]
    else:
        cap0 = []
        cap1 = []
    return segments_0 + segments_1 + cap0 + cap1


def _yxx_to_segments(
    y: ArrayLike,
    x0: ArrayLike,
    x1: ArrayLike,
    x2: ArrayLike,
    x3: ArrayLike,
    capsize: float,
):
    """
    |        |
     ───  ───  <-- y
    |        |
    ↑  ↑  ↑  ↑
    x0 x1 x2 x3
    """
    v0 = np.stack([x0, y], axis=1)
    v1 = np.stack([x1, y], axis=1)
    v2 = np.stack([x2, y], axis=1)
    v3 = np.stack([x3, y], axis=1)
    segments_0 = [[s0, s1] for s0, s1 in zip(v0, v1)]
    segments_1 = [[s2, s3] for s2, s3 in zip(v2, v3)]

    if capsize > 0:
        _c = np.array([0, capsize / 2])
        cap0 = [[s0 - _c, s0 + _c] for s0 in v0]
        cap1 = [[s3 - _c, s3 + _c] for s3 in v3]
    else:
        cap0 = []
        cap1 = []
    return segments_0 + segments_1 + cap0 + cap1


class BoxFace(FaceNamespace):
    _layer: BoxPlot

    @property
    def color(self) -> NDArray[np.floating]:
        """Face color of the bar."""
        return self._layer.boxes.face.color

    @color.setter
    def color(self, color):
        ndata = self._layer.boxes.ndata
        col = as_color_array(color, ndata)
        self._layer.boxes.face.color = col
        self.events.color.emit(col)

    @property
    def hatch(self) -> EnumArray[Hatch]:
        """Face fill hatch."""
        return self._layer.boxes.face.hatch

    @hatch.setter
    def hatch(self, hatch: str | Hatch | Iterable[str | Hatch]):
        ndata = self._layer.boxes.ndata
        hatches = as_any_1d_array(hatch, ndata, dtype=object)
        self._layer.boxes.face.hatch = hatches
        self.events.hatch.emit(hatches)

    @property
    def alpha(self) -> float:
        """Alpha value of the face."""
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> BoxPlot:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        hatch : FacePattern, optional
            Fill hatch of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if hatch is not _void:
            self.hatch = hatch
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class BoxEdge(EdgeNamespace):
    _layer: BoxPlot

    @property
    def color(self) -> NDArray[np.floating]:
        """Edge color of the box plot."""
        return self._layer.boxes.edge.color

    @color.setter
    def color(self, color: ColorType):
        col = np.array(Color(color), dtype=np.float32)  # assert a single color
        self._layer.boxes.edge.color = col
        self._layer.whiskers.color = col
        self._layer.medians.color = col
        self.events.color.emit(col)

    @property
    def width(self) -> NDArray[np.float32]:
        """Edge widths."""
        return self._layer.boxes.edge.width

    @width.setter
    def width(self, width: float):
        self._layer.boxes.edge.width = width
        self._layer.whiskers.width = width
        self._layer.medians.width = width
        self.events.width.emit(width)

    @property
    def style(self) -> EnumArray[LineStyle]:
        """Edge styles."""
        return self._layer.boxes.edge.style

    @style.setter
    def style(self, style: str | LineStyle):
        style = LineStyle(style)
        self._layer.boxes.edge.style = style
        self._layer.whiskers.style = style
        self._layer.medians.style = style
        self.events.style.emit(style)

    @property
    def alpha(self) -> float:
        return self.color[3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ) -> BoxPlot:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer
