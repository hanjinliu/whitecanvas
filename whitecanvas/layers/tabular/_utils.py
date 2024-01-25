from __future__ import annotations

import itertools

import numpy as np


def unique(arr: np.ndarray, axis=0) -> np.ndarray:
    """Return unique in the order of appearance."""
    _, idx = np.unique(arr, axis=axis, return_index=True)
    return arr[np.sort(idx)]


def unique_product(each_unique: list[np.ndarray]) -> np.ndarray:
    """
    Return the all the unique combinations of the given arrays.

    >>> unique_product([np.array([0, 1, 2]), np.array([3, 4])])
    array([[0, 3],
           [0, 4],
           [1, 3],
           [1, 4],
           [2, 3],
           [2, 4]])
    """
    return np.array(list(itertools.product(*each_unique)))
