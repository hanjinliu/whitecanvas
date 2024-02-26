from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from whitecanvas.backend import Backend
from whitecanvas.layers._mixin import (
    AbstractFaceEdgeMixin,
    EnumArray,
    MultiEdge,
    MultiFace,
    MultiPropertyEdgeBase,
    MultiPropertyFaceBase,
)
from whitecanvas.layers._primitive import Bars, MultiLine
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.layers.group._collections import LayerContainer, RichContainerEvents
from whitecanvas.theme import get_theme
from whitecanvas.types import (
    ArrayLike1D,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    _Void,
)
from whitecanvas.utils.normalize import as_any_1d_array, as_color_array

_void = _Void()


class BoxPlot(LayerContainer, AbstractFaceEdgeMixin["BoxFace", "BoxEdge"]):
    """
    A group for boxplot.

    Children layers are:
    - Bars (boxes)
    - MultiLine (whiskers)
    - MultiLine (median line)

     ──┬──  <-- max
       │
    ┌──┴──┐ <-- 75% quantile
    │  o  │ <-- mean
    ╞═════╡ <-- median
    └──┬──┘ <-- 25% quantile
       │
     ──┴──  <-- min
    """

    events: RichContainerEvents
    _events_class = RichContainerEvents

    def __init__(
        self,
        boxes: Bars[MultiFace, MultiEdge],
        whiskers: MultiLine,
        medians: MultiLine,
        *,
        name: str | None = None,
        orient: Orientation = Orientation.VERTICAL,
        capsize: float = 0.15,
    ):
        super().__init__([boxes, whiskers, medians], name=name)
        AbstractFaceEdgeMixin.__init__(self, BoxFace(self), BoxEdge(self))
        self._orient = Orientation.parse(orient)
        self._capsize = capsize
        self._init_events()

    @property
    def boxes(self) -> Bars[MultiFace, MultiEdge]:
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
        data: list[ArrayLike1D],
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
            extent=extent, backend=backend,
        ).with_face_multi(
            hatch=hatch, color=color, alpha=alpha,
        ).with_edge_multi(color="black")  # fmt: skip
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

        return cls(box, whiskers, medians, name=name, orient=ori, capsize=capsize)

    @property
    def orient(self) -> Orientation:
        """Orientation of the boxplot."""
        return self._orient

    def move(self, shift: float, autoscale: bool = True) -> BoxPlot:
        """Move the layer by the given shift."""
        self.boxes.set_data(xdata=self.boxes.data.x + shift)
        if self.orient.is_vertical:
            _wdata = []
            for seg in self.whiskers.data:
                _wdata.append(np.stack([seg[:, 0] + shift, seg[:, 1]], axis=0))
            self.whiskers.data = _wdata
            _mdata = []
            for seg in self.medians.data:
                _mdata.append(np.stack([seg[:, 0] + shift, seg[:, 1]], axis=0))
            self.medians.data = _mdata
        else:
            _wdata = []
            for seg in self.whiskers.data:
                _wdata.append(np.stack([seg[:, 0], seg[:, 1] + shift], axis=0))
            self.whiskers.data = _wdata
            _mdata = []
            for seg in self.medians.data:
                _mdata.append(np.stack([seg[:, 0], seg[:, 1] + shift], axis=0))
            self.medians.data = _mdata
        if autoscale and (canvas := self._canvas_ref()):
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def _update_data(self, agg_arr: NDArray[np.number]):
        x = self.boxes.data.x
        extent = self.boxes.bar_width
        self.boxes.set_data(ydata=agg_arr[3] - agg_arr[1], bottom=agg_arr[1])
        if self.orient.is_vertical:
            segs = _xyy_to_segments(
                x, agg_arr[0], agg_arr[1], agg_arr[3], agg_arr[4], self._capsize
            )
            medsegs = [
                np.array([(x0 - extent / 2, y0), (x0 + extent / 2, y0)])
                for x0, y0 in zip(x, agg_arr[2])
            ]
        else:
            segs = _yxx_to_segments(
                x, agg_arr[0], agg_arr[1], agg_arr[3], agg_arr[4], self._capsize
            )
            medsegs = [
                np.array([(x0, y0 - extent / 2), (x0, y0 + extent / 2)])
                for x0, y0 in zip(x, agg_arr[2])
            ]
        self.whiskers.data = segs
        self.medians.data = medsegs
        return None

    def _get_sep_values(self) -> NDArray[np.number]:
        """(5, N) array of min, 25%, 50%, 75%, max."""
        idx = 1 if self.orient.is_vertical else 0
        stop = self.boxes.ndata
        _min = [seg[0, idx] for seg in self.whiskers.data[:stop]]
        _p25 = self.boxes.bottom
        _median = [seg[0, idx] for seg in self.medians.data]
        _p75 = self.boxes.top
        _max = [seg[1, idx] for seg in self.whiskers.data[stop : stop * 2]]
        return np.stack([_min, _p25, _median, _p75, _max], axis=0)

    def _make_sure_hatch_visible(self):
        _is_no_width = self.edge.width == 0
        if np.any(_is_no_width):
            ec = get_theme().foreground_color
            self.edge.width = np.where(_is_no_width, 1, self.edge.width)
            self.edge.color = np.where(_is_no_width, ec, self.edge.color)

    def _xndata(self) -> int:
        nboxes = self.boxes.ndata
        nlines = self.whiskers.ndata
        assert nboxes * 2 == nlines or nboxes * 4 == nlines, f"{nboxes=}, {nlines=}"
        return nlines // nboxes


def _xyy_to_segments(
    x: ArrayLike1D,
    y0: ArrayLike1D,
    y1: ArrayLike1D,
    y2: ArrayLike1D,
    y3: ArrayLike1D,
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
    segments_0 = [np.stack([s0, s1]) for s0, s1 in zip(v0, v1)]
    segments_1 = [np.stack([s2, s3]) for s2, s3 in zip(v2, v3)]
    if capsize > 0:
        _c = np.array([capsize / 2, 0])
        cap0 = [np.stack([s0 - _c, s0 + _c]) for s0 in v0]
        cap1 = [np.stack([s3 - _c, s3 + _c]) for s3 in v3]
    else:
        cap0 = []
        cap1 = []
    return segments_0 + segments_1 + cap0 + cap1


def _yxx_to_segments(
    y: ArrayLike1D,
    x0: ArrayLike1D,
    x1: ArrayLike1D,
    x2: ArrayLike1D,
    x3: ArrayLike1D,
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
    segments_0 = [np.stack([s0, s1]) for s0, s1 in zip(v0, v1)]
    segments_1 = [np.stack([s2, s3]) for s2, s3 in zip(v2, v3)]

    if capsize > 0:
        _c = np.array([0, capsize / 2])
        cap0 = [np.stack([s0 - _c, s0 + _c]) for s0 in v0]
        cap1 = [np.stack([s3 - _c, s3 + _c]) for s3 in v3]
    else:
        cap0 = []
        cap1 = []
    return segments_0 + segments_1 + cap0 + cap1


class BoxFace(MultiPropertyFaceBase):
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


class BoxEdge(MultiPropertyEdgeBase):
    _layer: BoxPlot

    def _xndata(self) -> int:
        return self._layer._xndata()

    @property
    def color(self) -> NDArray[np.floating]:
        """Edge color of the box plot."""
        return self._layer.boxes.edge.color

    @color.setter
    def color(self, color: ColorType):
        ndata = self._layer.boxes.ndata
        col = as_color_array(color, ndata)
        self._layer.boxes.edge.color = col
        self._layer.whiskers.color = np.concatenate([col] * self._xndata(), axis=0)
        self._layer.medians.color = col
        self.events.color.emit(col)

    @property
    def width(self) -> NDArray[np.float32]:
        """Edge widths."""
        return self._layer.boxes.edge.width

    @width.setter
    def width(self, width: float):
        ndata = self._layer.boxes.ndata
        widths = as_any_1d_array(width, ndata, dtype=np.float32)
        self._layer.boxes.edge.width = widths
        self._layer.whiskers.width = np.tile(widths, self._xndata())
        self._layer.medians.width = widths
        self.events.width.emit(widths)

    @property
    def style(self) -> EnumArray[LineStyle]:
        """Edge styles."""
        return self._layer.boxes.edge.style

    @style.setter
    def style(self, style: str | LineStyle):
        ndata = self._layer.boxes.ndata
        if isinstance(style, (str, LineStyle)):
            styles = np.full(ndata, LineStyle(style), dtype=object)
        else:
            styles = np.array(style, dtype=object)
            if styles.shape != (ndata,):
                raise ValueError("Invalid shape of the style array.")
        self._layer.boxes.edge.style = styles
        self._layer.whiskers.style = np.tile(styles, self._xndata())
        self._layer.medians.style = styles
        self.events.style.emit(style)
