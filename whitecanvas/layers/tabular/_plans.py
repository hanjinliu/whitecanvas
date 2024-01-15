from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, Sequence, TypeVar

import numpy as np
from cmap import Color
from numpy.typing import NDArray

from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.layers.tabular._utils import unique
from whitecanvas.types import Hatch, LineStyle, Symbol

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers.tabular._df_compat import DataFrameWrapper

_V = TypeVar("_V", bound=Any)


class Plan(ABC, Generic[_V]):
    @abstractmethod
    def generate(
        self,
        vals: np.ndarray,
        by_all: tuple[str, ...],
    ) -> list[_V]:
        """
        Generate values for given labels.

        Length of the returned list must be equal to vals.shape[0].
        """


class CategoricalPlan(Plan[_V]):
    def __init__(self, by: tuple[str, ...]):
        self._by = by

    @property
    def by(self) -> tuple[str, ...]:
        return self._by

    @classmethod
    @abstractmethod
    def default(cls, by: Sequence[str]) -> Self:
        """Create a default plan."""


class OffsetPolicy(ABC):
    """Class that defines how to define offsets for given data."""

    @abstractmethod
    def get(self, interval: int) -> float:
        """Get 1D array for offsets"""

    def with_shift(self, val: float) -> CompositeOffsetPolicy:
        return CompositeOffsetPolicy([self, ConstOffset(val)])


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

    def with_shift(self, val: float) -> CompositeOffsetPolicy:
        return CompositeOffsetPolicy([*self._policies, ConstOffset(val)])


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


class OffsetPlan(CategoricalPlan[float]):
    """Plan of how to define x-offsets for each category."""

    def __init__(self, by: tuple[str, ...], offsets: list[OffsetPolicy]):
        super().__init__(by)
        self._offsets = offsets

    @classmethod
    def default(cls) -> OffsetPlan:
        return cls((), [])

    def more_by(self, *by: str, margin: float = 0.0) -> OffsetPlan:
        """Add more offsets."""
        found = [b for b in by if b in self._by]
        if found:
            raise ValueError(f"{found} is already in the plan.")
        new = [ConstMarginPolicy(margin) for _ in by]
        return OffsetPlan(self._by + tuple(by), self._offsets + new)

    def shift(self, val: float) -> OffsetPlan:
        new = [offset.with_shift(val) for offset in self._offsets]
        return OffsetPlan(self._by, new)

    def iter_ticks(
        self,
        labels: list[tuple[Any, ...]],
        by_all: tuple[str, ...],
    ) -> Iterator[tuple[float, list[str]]]:
        indices = [by_all.index(b) for b in self.by]
        ncols = len(indices)
        out_lookup: dict[tuple, float] = {}
        # make a full mesh where all combinations are included
        each_uniques = [
            unique(np.array([row[i] for row in labels]), axis=None) for i in indices
        ]

        last_row: tuple | None = None
        last_x = np.zeros(ncols, dtype=np.float32)
        # intervals will be like:
        # each_uniques --> intervals
        # [[1, 2], [4, 5, 6]] --> [3, 1]
        # [[1, 2], [4, 5, 6], [7, 8]] --> [6, 2, 1]
        intervals = _to_intervals(each_uniques)
        for row in itertools.product(*each_uniques):
            row_arr = np.array(row)
            if last_row is None:
                last_row = row_arr
                x = 0.0
            else:
                i = int(np.where(row_arr != last_row)[0][0])
                x0 = last_x[i]
                x = x0 + self._offsets[i].get(intervals[i])
                last_x[i:] = x
            out_lookup[row] = x
            last_row = row_arr
        # yield like this:
        # 0.0 ['Female', 'Dinner', 'No']
        # 1.0 ['Female', 'Dinner', 'Yes']
        # 2.0 ['Female', 'Lunch', 'No']
        # 3.0 ['Female', 'Lunch', 'Yes']
        # 4.0 ['Male', 'Dinner', 'No']
        # 5.0 ['Male', 'Dinner', 'Yes']
        # 6.0 ['Male', 'Lunch', 'No']
        # 7.0 ['Male', 'Lunch', 'Yes']
        for _r in labels:
            _pos = out_lookup[tuple(_r[i] for i in indices)]
            _labels = [str(_r[i]) for i in indices]
            yield _pos, _labels

    def generate(
        self,
        labels: list[tuple[Any, ...]],
        by_all: tuple[str, ...],
    ) -> list[float]:
        return [pos for pos, _ in self.iter_ticks(labels, by_all)]


# class TightOffsetPlan(OffsetPlan):

#     def generate(
#         self,
#         labels: np.ndarray,
#         by_all: tuple[str, ...],
#     ) -> list[float]:
#         indices = [by_all.index(b) for b in self.by]
#         ncols = len(indices)
#         out_lookup = {}
#         filt = _filter_unique(labels, indices)
#         x = 0.0
#         last_row = None
#         last_idx = np.zeros(ncols, dtype=np.int32)
#         for idx, row in enumerate(filt):
#             if last_row is None:
#                 last_row = row
#             else:
#                 i: int = np.where(row != last_row)[0][0]
#                 interval = idx - last_idx[i]
#                 offset = self._offsets[i].get(interval)
#                 last_idx[i:] = idx
#                 x += offset
#             out_lookup[tuple(row)] = x
#         out = [out_lookup[tuple(row_all[indices])] for row_all in labels]
#         return out


class CyclicPlan(CategoricalPlan[_V]):
    """
    Plan of cyclic values.

    This object is used to generate values for each category.
    """

    def __init__(self, by: tuple[str, ...], values: list[_V]):
        super().__init__(by)
        self._values = values

    def __repr__(self) -> str:
        cname = type(self).__name__
        return f"{cname}(by={self._by!r}, values={self._values})"

    @property
    def values(self) -> list[_V]:
        return self._values

    @classmethod
    @abstractmethod
    def _default_values(cls) -> list[_V]:
        """Default values for the plan."""

    @classmethod
    @abstractmethod
    def _norm_value(cls, v: Any) -> _V:
        """Normalize the value."""

    @classmethod
    def default(cls) -> Self:
        return cls((), cls._default_values())

    def update(self, *by: str, values: list[Any] | None = None) -> Self:
        """Return an updated plan."""
        cls = type(self)
        _by = tuple(by)
        if values is None:
            values = self.values
        else:
            values = [cls._norm_value(val) for val in values]
        return cls(_by, values)

    def with_const(self, value: _V) -> Self:
        return self.update(*self.by, values=[value])

    def generate(
        self,
        labels: list[tuple[Any, ...]],
        by_all: tuple[str, ...],
    ) -> list[_V]:
        # labels = [("a",), ("b",)] ... the all set of unique labels
        # by_all = ("column-0", "column-1")
        indices = [by_all.index(b) for b in self.by]
        size = len(self.values)
        # filt = _filter_unique(labels, indices)
        out_lookup: dict[tuple[Any, ...], _V] = {}
        i = 0
        for row in labels:
            row_filt = tuple(row[i] for i in indices)
            if row_filt not in out_lookup:
                out_lookup[row_filt] = self.values[i % size]
                i += 1

        ret = [out_lookup[tuple(_r[i] for i in indices)] for _r in labels]
        return ret

    def map(
        self,
        values: DataFrameWrapper,  # the data frame
    ) -> Sequence[_V]:
        if self._by:
            series = [values[k] for k in self._by]
        else:
            # constant, no key filter
            return self.values[0]
        uniques = [unique(ar, axis=None) for ar in series]
        out = np.empty(series[0].shape, dtype=object)
        i = 0
        for row in itertools.product(*uniques):
            sl: NDArray[np.bool_] = np.all(
                np.column_stack([a == b for a, b in zip(series, row)]), axis=1
            )
            ntrue = sl.sum()
            if ntrue > 0:
                val = self.values[i % len(self.values)]
                out[sl] = np.full(ntrue, val, dtype=object)
                i += 1
        return out


class MapPlan(ABC, Generic[_V]):
    def __init__(
        self,
        on: tuple[str, ...],
        mapper: Callable[[dict[str, np.ndarray]], Sequence[_V]],
    ):
        self._on = on
        self._mapper = mapper

    def __repr__(self) -> str:
        cname = type(self).__name__
        if self._on:
            return f"{cname}(on={self._on!r}, mapper={self._mapper})"
        else:
            return f"{cname}(mapper={self._mapper!r})"

    def with_map(
        self,
        on: tuple[str, ...],
        mapper: Callable[[dict[str, np.ndarray]], Sequence[_V]],
    ) -> Self:
        return type(self)(self._on + on, mapper)

    def with_range(self, on: str, limits: tuple[float, float] | None = None) -> Self:
        """Add a mapper that maps a range to a value."""

        def mapper(values: dict[str, np.ndarray]) -> Sequence[_V]:
            arr = values[on]
            valid = np.isfinite(arr)
            amin, amax = arr[valid].min(), arr[valid].max()
            if amin == amax:
                if limits is not None:
                    w0, w1 = limits
                    return np.full(arr.shape, (w0 + w1) / 2)
                else:
                    return np.full(arr.shape, amin)
            _arr = arr.clip(amin, amax)
            _arr[np.isnan(_arr)] = amin
            if limits is not None:
                w0, w1 = limits
                return (_arr - amin) / (amax - amin) * (w1 - w0) + w0
            else:
                return _arr

        return self.with_map((on,), mapper)

    def with_const(self, value: _V) -> Self:
        return self.with_map((), ConstMap(value))

    @classmethod
    def default(cls) -> Self:
        return cls((), cls._default_mapper())

    @classmethod
    @abstractmethod
    def _default_mapper(cls) -> Callable[[dict[str, np.ndarray]], Sequence[_V]]:
        """Default mapper for the plan."""

    def map(
        self,
        values: DataFrameWrapper,  # the data frame
    ) -> Sequence[_V]:
        """Calculate the values for the input data frame."""
        if self._on:
            input_dict = {k: values[k] for k in self._on}
        else:
            # constant or default size, no key filter
            try:
                k, v = next(iter(values.iter_items()))
                input_dict = {k: v}
            except StopIteration:
                input_dict = {}
        return self._mapper(input_dict)


class ColorPlan(CyclicPlan[Color]):
    @classmethod
    def _default_values(cls) -> list[Color]:
        return [Color("black")]  # NOTE: will be updated by the canvas palette

    @classmethod
    def _norm_value(cls, v: Any) -> Color:
        return Color(v)

    def with_palette(self, palette: ColorPalette) -> ColorPlan:
        palette = ColorPalette(palette)
        colors = palette.nextn(palette.ncolors, update=False)
        return ColorPlan(self._by, colors)

    # NOTE: Color instance is detected as a sequence of 4 floats
    #       so we need to override the default mapper
    def map(
        self,
        values: dict[str, np.ndarray],  # the data frame
    ) -> Sequence[_V]:
        if self._by:
            series = [values[k] for k in self._by]
        else:
            # constant, no key filter
            return self.values[0]
        uniques = [unique(ar, axis=None) for ar in series]
        out = np.empty((series[0].size, 4), dtype=np.float32)
        i = 0
        for row in itertools.product(*uniques):
            sl: NDArray[np.bool_] = np.all(
                np.column_stack([a == b for a, b in zip(series, row)]), axis=1
            )
            ntrue = sl.sum()
            if ntrue > 0:
                val = self.values[i % len(self.values)]
                out[sl] = [val] * ntrue
                i += 1
        return out


class StylePlan(CyclicPlan[LineStyle]):
    @classmethod
    def _default_values(cls) -> list[LineStyle]:
        return [
            LineStyle.SOLID,
            LineStyle.DASH,
            LineStyle.DOT,
            LineStyle.DASH_DOT,
        ]

    @classmethod
    def _norm_value(cls, v: Any) -> LineStyle:
        return LineStyle(v)

    def with_choices(self, choices: Sequence[LineStyle]) -> ColorPlan:
        choices = [LineStyle(s) for s in choices]
        return ColorPlan(self._by, choices)


class HatchPlan(CyclicPlan[Hatch]):
    @classmethod
    def _default_values(cls) -> list[Hatch]:
        return [
            Hatch.SOLID,
            Hatch.DIAGONAL_BACK,
            Hatch.HORIZONTAL,
            Hatch.CROSS,
            Hatch.VERTICAL,
            Hatch.DIAGONAL_CROSS,
            Hatch.DOTS,
            Hatch.DIAGONAL_FORWARD,
        ]

    @classmethod
    def _norm_value(cls, v: Any) -> Hatch:
        return Hatch(v)


class SymbolPlan(CyclicPlan[Symbol]):
    @classmethod
    def _default_values(cls) -> list[Symbol]:
        return [
            Symbol.CIRCLE,
            Symbol.SQUARE,
            Symbol.TRIANGLE_UP,
            Symbol.DIAMOND,
            Symbol.TRIANGLE_DOWN,
        ]

    @classmethod
    def _norm_value(cls, v: Any) -> Symbol:
        return Symbol(v)


class ColormapPlan(MapPlan[Color]):
    @classmethod
    def _default_mapper(cls):
        return lambda x: np.ones((x.size, 4))


class WidthPlan(MapPlan[float]):
    @classmethod
    def _default_mapper(cls):
        return lambda _: 1.0


class SizePlan(MapPlan[float]):
    @classmethod
    def _default_mapper(cls):
        return lambda _: 12.0


def _to_intervals(each_uniques: list[np.ndarray]):
    each_size = [un.size for un in each_uniques] + [1]
    return np.cumprod(each_size[1:][::-1])[::-1]


class ConstMap:
    def __init__(self, value):
        self._value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self._value!r}>"

    def __call__(self, values: dict[str, np.ndarray]) -> Sequence[_V]:
        series = next(iter(values.values()), np.zeros(0))
        return [self._value] * series.size
