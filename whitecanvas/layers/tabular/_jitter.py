from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
from ._utils import unique
from ._df_compat import DataFrameWrapper
from ._plans import OffsetPlan

_DF = TypeVar("_DF")


class JitterBase(ABC):
    @abstractmethod
    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
        """Map the source data to jittered data."""


class IdentityJitter(JitterBase):
    """No jittering."""

    def __init__(self, by: str):
        self._by = by

    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
        return src[self._by]


class UniformJitter(JitterBase):
    def __init__(self, by: str, extent: float = 0.8, seed: int | None = 0):
        self._by = by
        self._rng = np.random.default_rng(seed)
        self._extent = extent

    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
        w = self._extent
        jitter = self._rng.uniform(-w / 2, w / 2, size=len(src))
        return src[self._by] + jitter


class SwarmJitter(JitterBase):
    def __init__(
        self,
        by: str,
        limits: tuple[float, float],
        extent: float = 0.8,
        sort: bool = False,
    ):
        self._by = by
        self._extent = extent
        self._sort = sort
        self._limits = limits

    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
        if self._sort:
            values = np.sort(values)
        else:
            values = np.asarray(values)
        vmin, vmax = self._limits
        nbin = 25
        dv = (vmax - vmin) / nbin
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
        offsets = offset_pre / offset_max * min(self._extent / 2, width_default)
        return src[self._by] + offsets


def _map_x(*args: np.ndarray) -> NDArray[np.floating]:
    by_all = tuple(str(i) for i in range(len(args)))
    plan = OffsetPlan.default().more_by(*by_all)
    labels = [unique(a) for a in args]
    return plan.generate(labels, by_all)
