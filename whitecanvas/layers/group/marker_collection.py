from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, Symbol, LineStyle, Orientation
from whitecanvas.layers._primitive import Markers
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.utils.normalize import as_color_array


class MarkerCollection(LayerContainer):
    def __init__(
        self,
        markers: list[Markers],
        *,
        name: str | None = None,
        extent: float = 0.3,
        orient: Orientation = Orientation.VERTICAL,
    ):
        super().__init__(markers, name=name)
        self._extent = extent
        self._orient = Orientation.parse(orient)

    def nth(self, n: int) -> Markers:
        """The n-th markers layer."""
        return self._children[n]

    def with_edge(
        self,
        *,
        color: ColorType = "black",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> MarkerCollection:
        """Add edges to the strip plot."""
        for markers in self.iter_children():
            markers.with_edge(color=color, alpha=alpha, width=width, style=style)
        return self

    @classmethod
    def build_strip(
        cls,
        x: list[float],
        data: list[ArrayLike],
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.3,
        seed: int | None = 0,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        color: ColorType | Sequence[ColorType] = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: str | Backend | None = None,
    ):
        x, data = check_array_input(x, data)
        rng = np.random.default_rng(seed)
        ori = Orientation.parse(orient)
        layers: list[Markers] = []
        color = as_color_array(color, len(x))
        for ith, (x0, values, color) in enumerate(zip(x, data, color)):
            offsets = rng.uniform(-extent / 2, extent / 2, size=len(values))
            if ori.is_vertical:
                _x = np.full_like(values, x0) + offsets
                _y = values
            else:
                _x = values
                _y = np.full_like(values, x0) + offsets
            markers = Markers(
                _x, _y, name=f"markers_{ith}", symbol=symbol, size=size, color=color,
                alpha=alpha, pattern=pattern, backend=backend
            )  # fmt: skip
            layers.append(markers)

        return cls(layers, name=name, extent=extent, orient=ori)

    @classmethod
    def build_swarm(
        cls,
        x: list[float],
        data: list[ArrayLike],
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.3,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        sort: bool = False,
        color: ColorType | Sequence[ColorType] = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: str | Backend | None = None,
    ):
        x, data = check_array_input(x, data)
        ori = Orientation.parse(orient)
        layers: list[Markers] = []
        color = as_color_array(color, len(x))
        nbin = 25
        data_concat = np.concatenate(data)
        vmin, vmax = data_concat.min(), data_concat.max()
        dv = (vmax - vmin) / nbin
        for ith, (x0, values, color) in enumerate(zip(x, data, color)):
            if sort:
                values = np.sort(values)
            else:
                values = np.asarray(values)
            v_indices = np.floor((values - vmin) / dv).astype(np.int32)
            v_indices[v_indices == nbin] = nbin - 1
            offset_count = np.zeros(nbin, dtype=np.int32)
            offset_pre = np.zeros_like(values, dtype=np.int32)
            for i, idx in enumerate(v_indices):
                c = offset_count[idx]
                if c % 2 == 0:
                    offset_pre[i] = c / 2
                else:
                    offset_pre[i] = -(c + 1) / 2
                offset_count[idx] += 1
            offset_max = np.abs(offset_pre).max()
            width_default = dv * offset_max
            offsets = offset_pre / offset_max * min(extent / 2, width_default)
            if ori.is_vertical:
                _x = np.full_like(values, x0) + offsets
                _y = values
            else:
                _x = values
                _y = np.full_like(values, x0) + offsets
            if not sort:
                ...
            markers = Markers(
                _x,
                _y,
                name=f"markers_{ith}",
                symbol=symbol,
                size=size,
                color=color,
                alpha=alpha,
                pattern=pattern,
                backend=backend,
            )
            layers.append(markers)
        return cls(layers, name=name, extent=extent, orient=ori)

    @property
    def orient(self) -> Orientation:
        """Orientation of the strip plot."""
        return self._orient

    def with_edge(
        self,
        *,
        color: ColorType = "black",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
    ) -> MarkerCollection:
        """Add edges to the strip plot."""
        for markers in self.iter_children():
            markers.with_edge(color=color, alpha=alpha, width=width, style=style)
        return self

    if TYPE_CHECKING:  # fmt: skip

        def iter_children(self) -> Iterator[Markers]:
            ...
