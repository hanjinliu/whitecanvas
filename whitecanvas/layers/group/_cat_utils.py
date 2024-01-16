from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from whitecanvas.utils.normalize import as_array_1d


def check_array_input(
    x: Sequence[float],
    data: Sequence[ArrayLike],
) -> tuple[NDArray[np.floating], list[NDArray[np.floating]], list[str]]:
    x = as_array_1d(x)
    if len(x) != len(data):
        raise ValueError(f"len(x) != len(data), got {len(x)} and {len(data)}")
    return x, data
