from __future__ import annotations

import numpy as np
from typing import Sequence
from numpy.typing import ArrayLike, NDArray
from whitecanvas.utils.normalize import as_array_1d


def check_array_input(
    x: Sequence[float],
    data: Sequence[ArrayLike],
    labels: Sequence[str] | None = None,
) -> tuple[NDArray[np.floating], list[NDArray[np.floating]], list[str]]:
    x = as_array_1d(x)
    if len(x) != len(data):
        raise ValueError(f"len(x) != len(data), got {len(x)} and {len(data)}")
    if labels is None:
        labels = [f"data_{i}" for i in range(len(data))]
    elif len(labels) != len(data):
        raise ValueError(f"len(labels) != len(data), got {len(labels)} and {len(data)}")
    return x, data, labels
