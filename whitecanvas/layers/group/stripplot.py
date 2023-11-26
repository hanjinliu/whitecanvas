from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, Symbol, LineStyle, Orientation
from whitecanvas.layers.primitive import Markers
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.utils.normalize import as_color_array


class StripPlot(ListLayerGroup):
    def __init__(
        self,
        markers: list[Markers],
        *,
        name: str | None = None,
        strip_width: float = 0.3,
        orient: Orientation = Orientation.VERTICAL,
    ):
        super().__init__(markers, name=name)
        self._strip_width = strip_width
        self._orient = Orientation.parse(orient)

    def nth(self, n: int) -> Markers:
        """The n-th markers layer."""
        return self._children[n]

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike],
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        strip_width: float = 0.3,
        seed: int | None = None,
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
            offsets = rng.uniform(-strip_width / 2, strip_width / 2, size=len(values))
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

        return cls(layers, name=name, strip_width=strip_width, orient=ori)

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
    ) -> StripPlot:
        """Add edges to the strip plot."""
        for markers in self.iter_children():
            markers.with_edge(color=color, alpha=alpha, width=width, style=style)
        return self

    if TYPE_CHECKING:  # fmt: skip

        def iter_children(self) -> Iterator[Markers]:
            ...
