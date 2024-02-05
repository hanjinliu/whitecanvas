from __future__ import annotations

import numpy as np


def unique(arr: np.ndarray, axis=0) -> np.ndarray:
    """Return unique in the order of appearance."""
    if arr.dtype.kind in "fc":
        raise ValueError(f"Cannot handle {arr.dtype} in unique().")
    _, idx = np.unique(arr, axis=axis, return_index=True)
    return arr[np.sort(idx)]
