from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Sequence, TYPE_CHECKING, TypeVar
from itertools import product
from cmap import Color
import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import FacePattern
from whitecanvas.canvas._palette import ColorPalette

if TYPE_CHECKING:
    from typing_extensions import Self

_T = TypeVar("_T", bound=tuple)


class Plan(ABC):
    def __init__(self, by: tuple[str, ...]):
        self._by = by

    @property
    def by(self) -> tuple[str, ...]:
        return self._by

    @abstractclassmethod
    def default(cls, by: Sequence[str]) -> Self:
        """Create a default plan."""

    @abstractmethod
    def generate(
        self,
        vals: np.ndarray,
        by_all: tuple[str, ...],
    ) -> list[Any]:
        """
        Generate values for given labels.

        Length of the returned list must be equal to vals.shape[0].
        """


class OffsetPolicy(ABC):
    """Class that defines how to define offsets for given data."""

    @abstractmethod
    def get(self, interval: int) -> float:
        """Get 1D array for offsets"""


class ConstMarginPolicy(OffsetPolicy):
    def __init__(self, margin: float) -> None:
        self._margin = margin

    def get(self, interval: int) -> float:
        return interval + self._margin


class ConstPolicy(OffsetPolicy):
    def __init__(self, incr: float) -> None:
        self._incr = incr

    def get(self, interval: int) -> float:
        return self._incr


class NoMarginPolicy(ConstMarginPolicy):
    def __init__(self) -> None:
        super().__init__(0.0)


class OverlayPolicy(OffsetPolicy):
    def get(self, interval: int) -> float:
        return 0.0


class CompositeOffsetPolicy(OffsetPolicy):
    def __init__(self, policies: list[OffsetPolicy]) -> None:
        self._policies = policies

    def get(self, interval: int) -> float:
        return sum(policy.get(interval) for policy in self._policies)


class ConstOffset(OffsetPolicy):
    def __init__(self, val: float) -> None:
        self._val = val

    def get(self, idx: int, size: int):
        return self._val


class NoOffset(ConstOffset):
    def __init__(self) -> None:
        super().__init__(0.0)


class ManualOffset(OffsetPolicy):
    def __init__(self, offsets: list[float]) -> None:
        self._offsets = offsets

    def get(self, idx: int, size: int):
        return self._offsets[idx]


class OffsetPlan(Plan):
    """Plan of how to define x-offsets for each category."""

    def __init__(self, by: tuple[str, ...], offsets: list[OffsetPolicy]):
        super().__init__(by)
        self._offsets = offsets

    @classmethod
    def default(cls, by: tuple[str, ...]) -> OffsetPlan:
        offsets = [NoMarginPolicy() for _ in by]
        return cls(by, offsets)

    def more_by(self, *by: str, margin: float) -> OffsetPlan:
        """Add more offsets."""
        not_found = [b for b in by if b not in self._by]
        if not_found:
            raise ValueError(f"{not_found} is already in the plan.")
        new = [ConstMarginPolicy(margin) for _ in by]
        return OffsetPlan(self._by + (by,), self._offsets + new)

    def generate(
        self,
        values: np.ndarray,
        by_all: tuple[str, ...],
        blank: bool = True,
    ) -> list[float]:
        """
        Generate x coordinates for plotting.

        This method is a generarization of the jitter.
        `by_all` must be a superset of `self.by`.

        >>> val = [("A", "p"), ("A", "q"), ("A", "r"),
        ...        ("B", "p"), ("B", "q"), ("B", "r")]

        """
        indices = [by_all.index(b) for b in self.by]
        filt = _filter_unique(values, indices)
        ncols = len(indices)
        each_uniques = [_unique_ordered(filt[:, i]) for i in range(ncols)]
        out_lookup = {}
        if blank:
            # make a full mesh where all combinations are included
            _mesh = np.meshgrid(*[each_uniques[i] for i in range(ncols)])
            full = np.stack(_mesh, axis=-1).reshape(-1, ncols)
            # existing = set(tuple(row) for row in arr)  # make a set to speed up
            last_row = None
            last_x = np.zeros(ncols, dtype=np.float32)
            for row in full:
                if last_row is None:
                    last_row = row
                    x = 0
                else:
                    i: int = np.where(row != last_row)[0][0]
                    x0 = last_x[i]
                    if i + 1 < ncols:
                        interval = each_uniques[i + 1].size
                    else:
                        interval = 1
                    x = x0 + self._offsets[i].get(interval)
                    last_x[i] = x
                out_lookup[tuple(row)] = x
                last_row = row
        else:
            x = 0.0
            last_row = None
            last_x = np.zeros(ncols, dtype=np.float32)
            last_idx = np.zeros(ncols, dtype=np.int32)
            for idx, row in enumerate(filt):
                if last_row is None:
                    last_row = row
                else:
                    i: int = np.where(row != last_row)[0][0]
                    interval = idx - last_idx[i]
                    offset = self._offsets[i].get(interval)
                    last_idx[i:] = idx
                    x += offset
                out_lookup[tuple(row)] = x
        out = [out_lookup[tuple(row_all[indices])] for row_all in values]
        return out


class ColorPlan(Plan):
    def __init__(self, by: tuple[str, ...], palette: ColorPalette):
        super().__init__(by)
        self._palette = palette

    @classmethod
    def default(cls, by: Sequence[str]) -> ColorPlan:
        return cls(tuple(by), ColorPalette(["gray"]))

    def with_palette(self, palette: ColorPalette) -> ColorPlan:
        return ColorPlan(self._by, palette)

    def more_by(self, by: str) -> ColorPlan:
        """Add more offsets."""
        if by in self._by:
            raise ValueError(f"{by} is already in the plan.")
        return ColorPlan(self._by + (by,), self._palette)

    def generate(
        self,
        values: np.ndarray,
        by_all: tuple[str, ...],
    ) -> list[Color]:
        """Generate colors for each row of values."""
        indices = [by_all.index(b) for b in self.by]
        filt = _filter_unique(values, indices)
        out_lookup = {tuple(row): self._palette.next() for row in filt}
        return [out_lookup[tuple(row_all[indices])] for row_all in values]


class HatchPlan(Plan):
    def __init__(self, by: tuple[str, ...], hatches: list[FacePattern]):
        super().__init__(by)
        self._hatches = hatches

    @classmethod
    def default(cls, by: Sequence[str]) -> HatchPlan:
        return cls(
            tuple(by),
            [
                FacePattern.SOLID,
                FacePattern.DIAGONAL_BACK,
                FacePattern.HORIZONTAL,
                FacePattern.CROSS,
                FacePattern.VERTICAL,
                FacePattern.DIAGONAL_CROSS,
                FacePattern.DOTS,
                FacePattern.DIAGONAL_FORWARD,
            ],
        )

    def more_by(self, by: str) -> HatchPlan:
        """Add more offsets."""
        if by in self._by:
            raise ValueError(f"{by} is already in the plan.")
        return HatchPlan(self._by + (by,), self._hatches)

    def generate(
        self,
        values: np.ndarray,
        by_all: tuple[str, ...],
    ) -> list[FacePattern]:
        indices = [by_all.index(b) for b in self.by]
        size = len(self._hatches)
        filt = _filter_unique(values, indices)
        out_lookup = {tuple(row): self._hatches[i % size] for i, row in enumerate(filt)}
        return [out_lookup[tuple(row_all[indices])] for row_all in values]


def _filter_unique(values, indices: list[int]) -> np.ndarray:
    return _unique_ordered(values[:, indices])


def _unique_ordered(arr: np.ndarray) -> np.ndarray:
    """Return unique in the order of appearance."""
    _, idx = np.unique(arr, axis=0, return_index=True)
    return arr[np.sort(idx)]
