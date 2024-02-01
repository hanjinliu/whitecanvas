from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from whitecanvas.layers.tabular._df_compat import DataFrameWrapper

_DF = TypeVar("_DF")


class JitterBase(ABC):
    @abstractmethod
    def map(self, src: DataFrameWrapper[_DF]) -> NDArray[np.floating]:
        """Map the source data to jittered data."""


class IdentityJitter(JitterBase):
    """No jittering."""

    def __init__(self, by: str):
        if not isinstance(by, str):
            raise TypeError(f"Only str is allowed, got {type(by)}")
        self._by = by

    def map(self, src: DataFrameWrapper[_DF]) -> NDArray[np.floating]:
        return src[self._by]

    def check(self, src: DataFrameWrapper[_DF]) -> IdentityJitter:
        if self._by not in src:
            raise ValueError(f"Column {self._by} not found in the data frame.")
        if src[self._by].dtype.kind not in "iufb":
            raise ValueError(f"Column {self._by} is not numeric.")
        return self


class CategoricalLikeJitter(JitterBase):
    def __init__(self, by: str | tuple[str, ...], mapping: dict[tuple, float]):
        self._by = _tuple(by)
        self._mapping = mapping

    def _map(self, src: DataFrameWrapper[_DF]) -> NDArray[np.floating]:
        # only map the categorical data to real numbers
        args = [src[b] for b in self._by]
        out = np.zeros(len(src), dtype=np.float32)
        for row, pos in self._mapping.items():
            sl = np.all(np.column_stack([a == r for a, r in zip(args, row)]), axis=1)
            out[sl] = pos
        return out


class CategoricalJitter(CategoricalLikeJitter):
    """Jitter for categorical data."""

    def map(self, src: DataFrameWrapper[_DF]) -> NDArray[np.floating]:
        return self._map(src)


class UniformJitter(CategoricalLikeJitter):
    """Jitter with uniform distribution."""

    def __init__(
        self,
        by: str | tuple[str, ...],
        mapping: dict[tuple, float],
        extent: float = 0.8,
        seed: int | None = 0,
    ):
        super().__init__(by, mapping)
        self._rng = np.random.default_rng(seed)
        self._extent = extent

    def map(self, src: DataFrameWrapper[_DF]) -> NDArray[np.floating]:
        w = self._extent
        jitter = self._rng.uniform(-w / 2, w / 2, size=len(src))
        return self._map(src) + jitter


class SwarmJitter(CategoricalLikeJitter):
    """Jitter for swarm plot."""

    def __init__(
        self,
        by: str | tuple[str, ...],
        mapping: dict[tuple, float],
        value: str,
        limits: tuple[float, float],
        extent: float = 0.8,
    ):
        super().__init__(by, mapping)
        self._value = value
        self._extent = extent
        self._limits = limits

    def map(self, src: DataFrameWrapper[_DF]) -> NDArray[np.floating]:
        values = src[self._value]
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
        out = self._map(src) + offsets
        return out


def _tuple(x) -> tuple[str, ...]:
    if isinstance(x, str):
        return (x,)
    return tuple(x)
