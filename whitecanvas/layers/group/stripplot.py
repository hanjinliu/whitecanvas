from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from whitecanvas.backend import Backend
from whitecanvas.types import ColorType, FacePattern, Symbol, _Void, Orientation
from whitecanvas.layers.primitive import Markers
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.layers.group._cat_utils import check_array_input


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
        self._orient = Orientation(orient)

    def nth(self, n: int) -> Markers:
        return self._children[n]

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike],
        labels: list[str] | None = None,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        strip_width: float = 0.3,
        seed: int | None = None,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float = 10,
        color: ColorType = "blue",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
        backend: str | Backend | None = None,
    ):
        x, data, labels = check_array_input(x, data, labels)
        rng = np.random.default_rng(seed)
        ori = Orientation.parse(orient)
        layers: list[Markers] = []
        for offset, values, label in zip(x, data, labels):
            offsets = rng.uniform(-strip_width, strip_width, size=len(values))
            if ori.is_vertical:
                x = np.full_like(values, offset)
                y = values + offsets
            else:
                x = values + offsets
                y = np.full_like(values, offset)
            markers = Markers(
                x, y, name=label, symbol=symbol, size=size, color=color,
                alpha=alpha, pattern=pattern, backend=backend
            )  # fmt: skip
            layers.append(markers)

        return cls(layers, name=name, strip_width=strip_width, orient=ori)

    @property
    def orient(self) -> Orientation:
        return self._orient
