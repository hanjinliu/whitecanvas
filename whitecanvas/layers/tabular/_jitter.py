from __future__ import annotations
from abc import ABC, abstractmethod
import itertools

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
        if not isinstance(by, str):
            raise TypeError(f"Only str is allowed, got {type(by)}")
        self._by = by

    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
        return src[self._by]


class CategoricalJitter(JitterBase):
    """Jitter for categorical data."""

    def __init__(self, by: tuple[str, ...]):
        if isinstance(by, str):
            raise TypeError(f"Only tuple is allowed, got {type(by)}")
        self._by = by

    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
        # only map the categorical data to real numbers
        return _map_x([src[b] for b in self._by])


def identity_or_categorical(
    df: DataFrameWrapper[_DF],
    by: str | tuple[str, ...],
) -> JitterBase:
    """
    Return either IdentityJitter or CategoricalJitter depending on the data type.

    Parameters
    ----------
    df : DataFrameWrapper
        The source data.
    by : str | tuple[str, ...]
        Column(s) to be used for the x-axis.
    """
    if isinstance(by, str):
        series = df[by]
        if series.dtype.kind in "iuf":
            return IdentityJitter(by)
        else:
            return CategoricalJitter((by,))
    else:
        if len(by) == 1:
            return identity_or_categorical(df, by[0])
        else:
            return CategoricalJitter(by)


class UniformJitter(JitterBase):
    """Jitter with uniform distribution."""

    def __init__(
        self,
        by: str | tuple[str, ...],
        extent: float = 0.8,
        seed: int | None = 0,
    ):
        self._by = _tuple(by)
        self._rng = np.random.default_rng(seed)
        self._extent = extent

    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
        w = self._extent
        jitter = self._rng.uniform(-w / 2, w / 2, size=len(src))
        return _map_x([src[b] for b in self._by]) + jitter


class SwarmJitter(JitterBase):
    """Jitter for swarm plot."""

    def __init__(
        self,
        by: str | tuple[str, ...],
        value: str,
        limits: tuple[float, float],
        extent: float = 0.8,
    ):
        self._by = _tuple(by)
        self._value = value
        self._extent = extent
        self._limits = limits

    def map(self, src: DataFrameWrapper[_DF]) -> np.ndarray:
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
        out = _map_x([src[b] for b in self._by]) + offsets
        return out


def _tuple(x) -> tuple[str, ...]:
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def _map_x(args: list[np.ndarray]) -> NDArray[np.floating]:
    """
    Map the input data to x-axis values.

    >>> _map_x([["a", "a", "b", "b"], ["u", "v", "u", "v"]])  # [0, 1, 2, 3]
    >>> _map_x([["p", "q", "r", "r", "q"]])  # [0, 1, 2]
    """
    by_all = tuple(str(i) for i in range(len(args)))
    plan = OffsetPlan.default().more_by(*by_all)
    each_unique = [unique(a, axis=None) for a in args]
    labels = list(itertools.product(*each_unique))
    offsets = np.asarray(plan.generate(labels, by_all))
    out = np.zeros_like(args[0], dtype=np.float32)
    for i, row in enumerate(labels):
        sl = np.all(np.column_stack([a == r for a, r in zip(args, row)]), axis=1)
        out[sl] = offsets[i]
    return out
