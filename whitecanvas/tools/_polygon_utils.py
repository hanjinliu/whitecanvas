from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# modified from: https://qiita.com/yamadasuzaku/items/b7482131a06759731b47


def is_in_polygon(points: NDArray[np.number], poly: NDArray[np.number]):
    num = poly.shape[0]
    inside = np.full(points.shape[0], False, dtype=np.bool_)
    xs = points[:, 0]
    ys = points[:, 1]
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        mask = (
            (ys > min(p1y, p2y))
            & (ys <= max(p1y, p2y))
            & (xs <= max(p1x, p2x))
            & (p1y != p2y)
        )
        xinters = (ys[mask] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
        inside[mask] = np.logical_xor(inside[mask], xs[mask] < xinters)
        p1x, p1y = p2x, p2y

    return inside
